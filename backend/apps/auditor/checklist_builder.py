from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from django.conf import settings

from apps.search.services import SearchResult
from core.neo4j import neo4j_client

_LATIN_TO_PERSIAN = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
_PERSIAN_TO_LATIN = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
_MAX_FEATURE_EVIDENCE = 8
_MAX_RULING_EVIDENCE = 5
_MAX_GROUNDED_FEATURES_PER_ARTICLE = 10

# See the priority-order comment above _rank_article_refs (unchanged) for
# why article candidates are capped and ranked the way they are.
_DEFAULT_MAX_ARTICLES_PER_QUERY = 25

_BATCH_GROUNDED_FEATURES_QUERY = """
UNWIND $articles AS a
CALL {
  WITH a
  MATCH (feature)-[g:GROUNDED_IN]->(article:Article {article_number: a.number, law: a.law})
  WHERE g.npmi IS NOT NULL
  RETURN labels(feature)[0] AS feature_label, feature.name AS feature_value
  ORDER BY g.npmi DESC
  LIMIT $limit
}
RETURN a.law AS law, a.number AS number, collect({label: feature_label, value: feature_value}) AS top_features
"""

_BATCH_ARTICLE_TEXT_QUERY = """
UNWIND $refs AS ref
MATCH (law:Law {name: ref.law_name})-[:CONTAINS]->(article:Article {article_number: ref.article_number})
RETURN ref.law_name AS law_name, ref.article_number AS article_number, article.content AS text
"""


def _parse_article_ref(ref: str) -> tuple[str, int] | None:
    """
    Reverses the "<law> - ماده <persian digits>" format used everywhere
    in this codebase back into (law, article_number) as stored on the
    Article node -- law and article_number (Latin int), per
    database/law_loader.py. Returns None for a ref that doesn't match
    the expected shape; callers skip GROUNDED_IN lookup for it rather
    than guessing.
    """
    if " - ماده " not in ref:
        return None
    law_name, num_part = ref.rsplit(" - ماده ", 1)
    latin_num = num_part.translate(_PERSIAN_TO_LATIN)
    if not latin_num.isdigit():
        return None
    return law_name, int(latin_num)


def _fetch_grounded_features(refs: list[str]) -> dict[str, set[tuple[str, str]]]:
    """
    Batched lookup against the GROUNDED_IN edges built by
    database/grounded_in_builder.py: for each article_ref, returns the
    set of (feature_label, feature_value) pairs statistically grounded
    to it (ranked by NPMI, capped). A ref that fails to parse, or whose
    article has no GROUNDED_IN edges (e.g. excluded procedural-law
    articles -- see that file's _PROCEDURAL_DOMAINS), is simply absent
    from the returned dict; build_evidence_bundles falls back to the
    shared feature list for those.
    """
    parsed_by_ref = {ref: _parse_article_ref(ref) for ref in refs}
    articles_param = [
        {"law": law, "number": number}
        for parsed in parsed_by_ref.values() if parsed
        for law, number in [parsed]
    ]
    if not articles_param:
        return {}

    with neo4j_client.session() as session:
        rows = session.run(
            _BATCH_GROUNDED_FEATURES_QUERY,
            articles=articles_param,
            limit=_MAX_GROUNDED_FEATURES_PER_ARTICLE,
        )
        grounded_by_law_number = {
            (row["law"], row["number"]): {
                (f["label"], f["value"]) for f in row["top_features"]
            }
            for row in rows
        }

    return {
        ref: grounded_by_law_number.get(parsed, set())
        for ref, parsed in parsed_by_ref.items() if parsed
    }


def _fetch_article_texts(refs: list[str]) -> dict[str, str]:
    """
    Batched, deterministic fetch of article text by (law_name,
    article_number) for every ref in `refs` -- independent of whether
    the article came from the embedding-based article_channel or only
    from a ruling's citation. Fixes a real bug: citation-only articles
    (e.g. ماده ۴۷۴, reached only via ruling citations) previously had
    empty article_text, because the old article_text_by_ref was built
    solely from search_result.article_results. Without the article's own
    text, the verifier prompt's requirement to ground every condition in
    the article's actual wording (see verifier.py's Golden Rule 1) was
    impossible to satisfy for those articles.

    Uses the same Law/CONTAINS/Article pattern already proven working by
    apps/search/services.py's article_channel query -- not the separate
    Article.law string property used by GROUNDED_IN edges above.
    """
    parsed_by_ref = {ref: _parse_article_ref(ref) for ref in refs}
    refs_param = [
        {"law_name": law, "article_number": number}
        for parsed in parsed_by_ref.values() if parsed
        for law, number in [parsed]
    ]
    if not refs_param:
        return {}

    with neo4j_client.session() as session:
        rows = session.run(_BATCH_ARTICLE_TEXT_QUERY, refs=refs_param)
        text_by_law_number = {
            (row["law_name"], row["article_number"]): row["text"] or ""
            for row in rows
        }

    return {
        ref: text_by_law_number.get(parsed, "")
        for ref, parsed in parsed_by_ref.items() if parsed
    }


@dataclass
class ArticleEvidenceBundle:
    """All evidence gathered for a single article_ref, assembled from the
    three retrieval channels, to be handed to the verifier prompt."""
    article_ref: str
    article_text: str
    supporting_texts: list[str]


def _normalized_citations(cited_articles: list[str] | None) -> list[str]:
    """
    e.cited_articles comes straight from the Neo4j query with Latin
    digits (toString(article.article_number)), while search_result's own
    article_refs/article_number formatting is Persian-digit. Every
    comparison against a ref string in this file MUST go through this
    first, or citation matching silently returns nothing -- this was a
    real bug: citation_weight in debug_rank_article_refs was always
    0.0000 for every ref, and build_evidence_bundles' ruling_texts was
    always empty (only the shared feature texts ever reached the LLM),
    both from this exact mismatch.
    """
    return [ref.translate(_LATIN_TO_PERSIAN) for ref in (cited_articles or [])]


def _rank_article_refs(search_result: SearchResult) -> list[str]:
    """See the priority order explained in the module-level comment above."""
    return [ref for ref, _tier, _weight in debug_rank_article_refs(search_result)]


def debug_rank_article_refs(search_result: SearchResult) -> list[tuple[str, str, float]]:
    """
    Same ranking as _rank_article_refs, but returns (ref, tier, weight)
    for every candidate instead of just the ordered ref list -- lets a
    debug script show exactly why a given ref landed where it did:
    'direct' tier + the article_channel score, or 'citation' tier + the
    summed score of every retrieved ruling that cites it.
    """
    direct_refs_ordered: list[tuple[str, str, float]] = []
    seen = set()
    for e in search_result.article_results:
        if not e.law_name or not e.article_number:
            continue
        ref = f"{e.law_name} - ماده {str(e.article_number).translate(_LATIN_TO_PERSIAN)}"
        if ref not in seen:
            seen.add(ref)
            direct_refs_ordered.append((ref, "direct", e.score))

    citation_weight: dict[str, float] = defaultdict(float)
    for e in search_result.ruling_results:
        for ref in _normalized_citations(e.cited_articles):
            citation_weight[ref] += e.score

    original_order = {ref: i for i, ref in enumerate(search_result.article_refs)}
    citation_only_refs = [ref for ref in search_result.article_refs if ref not in seen]
    citation_only_refs.sort(key=lambda ref: (-citation_weight[ref], original_order[ref]))

    citation_ranked = [(ref, "citation", citation_weight[ref]) for ref in citation_only_refs]
    return direct_refs_ordered + citation_ranked


def build_evidence_bundles(search_result: SearchResult) -> list[ArticleEvidenceBundle]:
    """
    Groups evidence in `search_result` by article_ref so each article can
    be verified independently. An article's supporting evidence is:
      - its own article text, fetched deterministically by (law,
        article_number) for every ranked ref (see _fetch_article_texts) --
        not dependent on whether article_channel happened to surface it
      - text of ruling evidence that cites it (cited_articles match)
      - feature evidence GROUNDED_IN this specific article (via the
        NPMI-scored GROUNDED_IN edges from database/grounded_in_builder.py),
        intersected with the features actually retrieved for this query
        -- NOT a generic shared list. If an article has no GROUNDED_IN
        edges (e.g. excluded procedural-law articles) or none of its
        grounded features were retrieved for this query, falls back to
        the top retrieved features generally, same as before.

    Only the top `AUDITOR_MAX_ARTICLES_PER_QUERY` refs (article_channel
    hits first, then citation-score-weighted, see _rank_article_refs)
    get a bundle -- see module docstring above for why capping is
    necessary and why this priority order.
    """
    max_articles = getattr(
        settings, "AUDITOR_MAX_ARTICLES_PER_QUERY", _DEFAULT_MAX_ARTICLES_PER_QUERY
    )
    ranked_refs = _rank_article_refs(search_result)[:max_articles]

    article_text_by_ref = _fetch_article_texts(ranked_refs)

    grounded_features_by_ref = _fetch_grounded_features(ranked_refs)

    fallback_feature_texts = [
        e.text for e in search_result.feature_results[:_MAX_FEATURE_EVIDENCE] if e.text
    ]

    bundles: list[ArticleEvidenceBundle] = []
    for ref in ranked_refs:
        ruling_texts = [
            e.text
            for e in search_result.ruling_results
            if e.text and ref in _normalized_citations(e.cited_articles)
        ][:_MAX_RULING_EVIDENCE]

        # Grounded features are used as-is (their names), not filtered down
        # to only the ones a retrieved quote happens to match verbatim --
        # exact-match intersection was tried and made evidence sparser,
        # not richer, since the graph's top-NPMI features for an article
        # (computed corpus-wide) rarely match a specific case's extracted
        # feature values by exact string equality. Presented as a distinct
        # "known relevant factors" line so the LLM can use it as prior
        # knowledge about the article even without a case-specific quote.
        grounded_pairs = grounded_features_by_ref.get(ref, set())
        grounded_values = [value for _label, value in grounded_pairs]
        grounded_context = (
            "عوامل حقوقی که طبق آمار پرونده‌های مشابه با این ماده مرتبط‌اند: "
            + "، ".join(grounded_values)
        ) if grounded_values else None

        feature_texts = fallback_feature_texts
        supporting_texts = ruling_texts + feature_texts
        if grounded_context:
            supporting_texts = [grounded_context] + supporting_texts

        bundles.append(ArticleEvidenceBundle(
            article_ref=ref,
            article_text=article_text_by_ref.get(ref, ""),
            supporting_texts=supporting_texts,
        ))

    return bundles