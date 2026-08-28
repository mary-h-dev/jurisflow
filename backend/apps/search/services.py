from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from core.neo4j import neo4j_client
from .embedder import embed_text
from .router_cache import route_query_cached
from .router import RoutingResult
from .confidence import (
    UncertaintyVector,
    ConfidenceResult,
    build_uncertainty_vector,
    compute_confidence,
)

logger = logging.getLogger(__name__)

_RRF_K = 60
_LATIN_TO_PERSIAN = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


# -- Evidence dataclass ------------------------------------------------------

@dataclass
class Evidence:
    source_type:      str
    text:             str
    score:            float
    rrf_score:        float         = 0.0
    ruling_id:        Optional[str] = None
    feature_value:    Optional[str] = None
    feature_category: Optional[str] = None
    article_number:   Optional[int] = None
    law_name:         Optional[str] = None
    cited_articles:   list[str]     = field(default_factory=list)
    confidence:       Optional[float] = None


@dataclass
class SearchResult:
    query:            str
    evidences:        list[Evidence]             = field(default_factory=list)
    feature_results:  list[Evidence]             = field(default_factory=list)
    ruling_results:   list[Evidence]             = field(default_factory=list)
    article_results:  list[Evidence]             = field(default_factory=list)
    confidence:       Optional[ConfidenceResult] = None
    routing:          Optional[RoutingResult]    = None

    @property
    def ruling_ids(self) -> list[str]:
        return list({e.ruling_id for e in self.evidences if e.ruling_id})

    @property
    def article_refs(self) -> list[str]:
        return _build_article_refs(self.article_results, self.ruling_results)


# -- RRF ----------------------------------------------------------------------

def _rrf_score(rank: int) -> float:
    return 1.0 / (_RRF_K + rank)


def _fuse_with_rrf(*ranked_lists: list[Evidence], top_k: int = 20) -> list[Evidence]:
    scores: dict[str, float]    = {}
    items:  dict[str, Evidence] = {}

    for ranked_list in ranked_lists:
        for rank, ev in enumerate(ranked_list, start=1):
            key = f"{ev.source_type}:{ev.ruling_id}:{ev.text[:50]}"
            scores[key] = scores.get(key, 0.0) + _rrf_score(rank)
            if key not in items:
                items[key] = ev

    sorted_keys = sorted(scores, key=lambda k: -scores[k])[:top_k]
    results = []
    for key in sorted_keys:
        ev           = items[key]
        ev.rrf_score = scores[key]
        results.append(ev)
    return results


# -- article_refs helper -------------------------------------------------------

def _build_article_refs(
    article_results: list[Evidence],
    ruling_results: list[Evidence],
) -> list[str]:
    """
    Built directly from article_results and ruling citations (each with
    its own top_k), NOT from the RRF-fused `evidences` list -- the fused
    list caps at final_top_k across all channels competing together and
    can drop an article that ranked well within its own channel.
    Format uses Persian digits to match gold_articles annotation format.
    """
    refs: list[str] = []
    seen: set[str] = set()

    for e in article_results:
        if not e.law_name or not e.article_number:
            continue
        ref = f"{e.law_name} - ماده {str(e.article_number).translate(_LATIN_TO_PERSIAN)}"
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)

    for e in ruling_results:
        for ref in (e.cited_articles or []):
            normalized = ref.translate(_LATIN_TO_PERSIAN)
            if normalized not in seen:
                seen.add(normalized)
                refs.append(normalized)

    return refs


# -- Channel queries ------------------------------------------------------------

_FEATURE_QUERY = """
CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
YIELD node AS feature, score
MATCH (r:Ruling)-[rel]->(feature)
WHERE type(rel) IN ['HAS_CONCEPT','HAS_ROLE','HAS_ACTION','HAS_OBJECT','HAS_FACT']
RETURN
    feature.name       AS value,
    labels(feature)[0] AS category,
    rel.evidence       AS quote,
    rel.confidence     AS confidence,
    r.ruling_id        AS ruling_id,
    score
ORDER BY score DESC
LIMIT $top_k
"""

_FEATURE_LABELS = ["LegalConcept", "LegalRole", "LegalAction", "LegalObject", "LegalFact"]

_EXACT_FEATURE_QUERY = """
MATCH (r:Ruling)-[rel]->(feature)
WHERE type(rel) IN ['HAS_CONCEPT','HAS_ROLE','HAS_ACTION','HAS_OBJECT']
  AND feature.name = $canonical_value
RETURN
    feature.name       AS value,
    labels(feature)[0] AS category,
    rel.evidence       AS quote,
    rel.confidence     AS confidence,
    r.ruling_id        AS ruling_id,
    1.0                AS score
LIMIT $top_k
"""

_RULING_SUMMARY_QUERY = """
CALL db.index.vector.queryNodes('ruling_summary_embedding', $top_k, $embedding)
YIELD node AS ruling, score
OPTIONAL MATCH (ruling)-[cites:CITES]->(article:Article)
WITH ruling, score, collect(DISTINCT (article.law + ' - ماده ' + toString(article.article_number))) AS cited
RETURN
    ruling.ruling_id             AS ruling_id,
    ruling.legal_factual_summary AS text,
    cited                        AS cited_articles,
    score
ORDER BY score DESC
LIMIT $top_k
"""

_RULING_SECTION_QUERY = """
CALL db.index.vector.queryNodes('ruling_section_embedding', $top_k, $embedding)
YIELD node AS section, score
MATCH (r:Ruling)-[:HAS_SECTION]->(section)
OPTIONAL MATCH (r)-[cites:CITES]->(article:Article)
WITH r, section, score, collect(DISTINCT (article.law + ' - ماده ' + toString(article.article_number))) AS cited
RETURN
    r.ruling_id  AS ruling_id,
    section.text AS text,
    cited        AS cited_articles,
    score
ORDER BY score DESC
LIMIT $top_k
"""

_ARTICLE_QUERY = """
CALL db.index.vector.queryNodes('article_embedding', $top_k, $embedding)
YIELD node AS article, score
MATCH (law:Law)-[:CONTAINS]->(article)
RETURN
    article.article_number AS article_number,
    article.content         AS text,
    law.name                AS law_name,
    score
ORDER BY score DESC
LIMIT $top_k
"""


def _search_features(session, query: str, embedding: list[float], top_k: int) -> list[Evidence]:
    """Deterministic exact/token match against the closed vocabulary first
    (see exact_feature_matcher.py); falls back to embedding similarity
    only if no vocabulary term matched -- short legal terms don't
    separate reliably in embedding space."""
    from .exact_feature_matcher import exact_feature_matcher

    exact_matches = exact_feature_matcher.find_matches(query)

    if exact_matches:
        results = []
        seen: set[str] = set()
        for canonical_value, _category_key in exact_matches:
            try:
                records = session.run(_EXACT_FEATURE_QUERY, canonical_value=canonical_value, top_k=top_k)
                for r in records:
                    key = f"{r['ruling_id']}:{r['value']}"
                    if key in seen:
                        continue
                    seen.add(key)
                    text = r["quote"] or r["value"] or ""
                    results.append(Evidence(
                        source_type="feature",
                        text=text,
                        score=float(r["score"]),
                        ruling_id=str(r["ruling_id"]) if r["ruling_id"] else None,
                        feature_value=r["value"],
                        feature_category=r["category"],
                        confidence=float(r["confidence"]) if r["confidence"] else None,
                    ))
            except Exception as e:
                logger.warning(f"Exact feature match query failed for {canonical_value}: {e}")
        if results:
            return results[:top_k]

    return _search_features_embedding_fallback(session, embedding, top_k)


def _search_features_embedding_fallback(session, embedding: list[float], top_k: int) -> list[Evidence]:
    results = []
    seen: set[str] = set()
    for label in _FEATURE_LABELS:
        index_name = f"{label.lower()}_embedding"
        try:
            records = session.run(_FEATURE_QUERY, index_name=index_name, embedding=embedding, top_k=top_k)
            for r in records:
                key = f"{r['ruling_id']}:{r['value']}"
                if key in seen:
                    continue
                seen.add(key)
                text = r["quote"] or r["value"] or ""
                results.append(Evidence(
                    source_type="feature",
                    text=text,
                    score=float(r["score"]),
                    ruling_id=str(r["ruling_id"]) if r["ruling_id"] else None,
                    feature_value=r["value"],
                    feature_category=r["category"],
                    confidence=float(r["confidence"]) if r["confidence"] else None,
                ))
        except Exception as e:
            logger.warning(f"Feature search failed for {label}: {e}")
    return sorted(results, key=lambda e: -e.score)[:top_k]


def _search_ruling_sections(session, embedding: list[float], top_k: int) -> list[Evidence]:
    try:
        records = session.run(_RULING_SECTION_QUERY, embedding=embedding, top_k=top_k)
        return [
            Evidence(
                source_type="ruling", text=r["text"] or "", score=float(r["score"]),
                ruling_id=str(r["ruling_id"]) if r["ruling_id"] else None,
                cited_articles=[c.translate(_LATIN_TO_PERSIAN) for c in (r["cited_articles"] or [])],
            )
            for r in records
        ]
    except Exception as e:
        logger.warning(f"Ruling section search failed: {e}")
        return []


def _search_ruling_summaries(session, embedding: list[float], top_k: int) -> list[Evidence]:
    try:
        records = session.run(_RULING_SUMMARY_QUERY, embedding=embedding, top_k=top_k)
        return [
            Evidence(
                source_type="ruling", text=r["text"] or "", score=float(r["score"]),
                ruling_id=str(r["ruling_id"]) if r["ruling_id"] else None,
                cited_articles=[c.translate(_LATIN_TO_PERSIAN) for c in (r["cited_articles"] or [])],
            )
            for r in records
        ]
    except Exception as e:
        logger.warning(f"Ruling summary search failed: {e}")
        return []


def _search_rulings(session, embedding: list[float], top_k: int) -> list[Evidence]:
    summary_hits = _search_ruling_summaries(session, embedding, top_k)
    section_hits = _search_ruling_sections(session, embedding, top_k)
    return _fuse_with_rrf(summary_hits, section_hits, top_k=top_k)


def _search_articles(session, embedding: list[float], top_k: int) -> list[Evidence]:
    try:
        records = session.run(_ARTICLE_QUERY, embedding=embedding, top_k=top_k)
        return [
            Evidence(
                source_type="article", text=r["text"] or "", score=float(r["score"]),
                article_number=r["article_number"], law_name=r["law_name"],
            )
            for r in records
        ]
    except Exception as e:
        logger.warning(f"Article search failed: {e}")
        return []


# -- Search Service -------------------------------------------------------------

class SearchService:
    """
    Channel gating (LLM deciding which channels to query) was tested via
    ablation and REMOVED: it reduced AUROC from 0.756 to 0.682 and
    Spearman from 0.401 to 0.151 (see apps/search/calibration/). All
    three channels are now always queried. The router's rewrite +
    case_type_hint are still used (rewrite improved article ranking in
    manual tests); case_type_hint is metadata only and does not filter
    retrieval.
    """

    def search(
        self,
        query:         str,
        feature_top_k: int = 20,
        ruling_top_k:  int = 20,
        article_top_k: int = 20,
        final_top_k:   int = 20,
    ) -> SearchResult:
        routing = route_query_cached(query)
        embedding = embed_text(routing.rewritten_query)

        with neo4j_client.session() as session:
            feature_results = _search_features(session, routing.rewritten_query, embedding, feature_top_k)
            ruling_results  = _search_rulings(session, embedding, ruling_top_k)
            article_results = _search_articles(session, embedding, article_top_k)

        fused = _fuse_with_rrf(
            feature_results, ruling_results, article_results, top_k=final_top_k,
        )

        # graph_support disabled: ablation showed it reduced AUROC
        # (0.756 -> 0.675). graph_support.py is kept intact for reference
        # and possible future use with more annotation data.

        vec = build_uncertainty_vector(
            feature_scores=[e.score for e in feature_results],
            ruling_scores=[e.score for e in ruling_results],
            article_scores=[e.score for e in article_results],
        )
        confidence = compute_confidence(vec)

        return SearchResult(
            query=query,
            evidences=fused,
            feature_results=feature_results,
            ruling_results=ruling_results,
            article_results=article_results,
            confidence=confidence,
            routing=routing,
        )


search_service = SearchService()