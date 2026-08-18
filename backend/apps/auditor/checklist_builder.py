from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from django.conf import settings

from apps.search.services import SearchResult



_LATIN_TO_PERSIAN = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
_MAX_FEATURE_EVIDENCE = 8
_MAX_RULING_EVIDENCE = 5
_DEFAULT_MAX_ARTICLES_PER_QUERY = 25


# search_result.article_refs is a UNION of every article cited by every
# retrieved ruling, which routinely runs to 50-100+ unique refs for a
# single query -- most appearing in only one ruling's citation list.
# Verifying every single one is impractical (LLM calls, latency, free-tier
# rate limits) and unnecessary, so we cap it.
#
# We tried two rankings that both backfired on real cases:
#   - raw citation frequency: generic articles that several unrelated,
#     low-relevance rulings happen to co-cite outrank a gold article
#     that only one HIGHLY relevant retrieved ruling cites (observed:
#     ruling 34630 -- گold procedural article ماده ۴۷۴ ranked ~43rd of
#     75 purely because few rulings happened to cite it).
#   - article_channel-first, frequency-second: still leaves the same
#     citation-only ties broken by raw count, same problem for refs the
#     embedding channel never surfaces directly.
#
# Correct priority now:
#   1. article_results (article_channel) hits, in their OWN score order --
#      already a query-specific relevance ranking, trust it first.
#   2. remaining budget filled by ruling-citation refs, ranked by the SUM
#      of the .score of every retrieved ruling that cites them -- so one
#      citation from a highly-relevant ruling can outrank several
#      citations from low-relevance ones, instead of raw count treating
#      every citing ruling as equally informative regardless of how well
#      it actually matched the query.







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
      - its own article text, from article_results
      - text of ruling evidence that cites it (cited_articles match)
      - a capped sample of feature evidence (case-fact context; features
        are not linked to specific articles in the graph, so the same
        top feature facts are shared across all article bundles for a
        given query)

    Only the top `AUDITOR_MAX_ARTICLES_PER_QUERY` refs (article_channel
    hits first, then citation-score-weighted, see _rank_article_refs)
    get a bundle -- see module docstring above for why capping is
    necessary and why this priority order.
    """
    max_articles = getattr(
        settings, "AUDITOR_MAX_ARTICLES_PER_QUERY", _DEFAULT_MAX_ARTICLES_PER_QUERY
    )
    ranked_refs = _rank_article_refs(search_result)[:max_articles]

    article_text_by_ref: dict[str, str] = {}
    for e in search_result.article_results:
        if not e.law_name or not e.article_number:
            continue
        ref = f"{e.law_name} - ماده {str(e.article_number).translate(_LATIN_TO_PERSIAN)}"
        article_text_by_ref.setdefault(ref, e.text)

    shared_feature_texts = [
        e.text for e in search_result.feature_results[:_MAX_FEATURE_EVIDENCE] if e.text
    ]

    bundles: list[ArticleEvidenceBundle] = []
    for ref in ranked_refs:
        ruling_texts = [
            e.text
            for e in search_result.ruling_results
            if e.text and ref in _normalized_citations(e.cited_articles)
        ][:_MAX_RULING_EVIDENCE]

        bundles.append(ArticleEvidenceBundle(
            article_ref=ref,
            article_text=article_text_by_ref.get(ref, ""),
            supporting_texts=ruling_texts + shared_feature_texts,
        ))

    return bundles