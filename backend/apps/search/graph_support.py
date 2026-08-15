from __future__ import annotations

import logging
from typing import Optional

from core.neo4j import neo4j_client

logger = logging.getLogger(__name__)

# Matches the threshold used in database/grounded_in_builder.py. Kept as a
# separate constant here (not imported) since this module runs in the web
# process, not the offline batch job — duplication is intentional decoupling,
# not an oversight. Both must be recalibrated together.
_MIN_CO_COUNT = 3

_GRAPH_SUPPORT_QUERY = """
UNWIND $pairs AS pair
MATCH (feature) WHERE feature.name = pair.feature_value
MATCH (feature)-[g:GROUNDED_IN]->(article:Article)
WHERE article.article_number = pair.article_number
  AND article.law = pair.law_name
  AND (g.official_count + g.text_count) >= $min_co_count
  AND g.npmi IS NOT NULL
RETURN pair.feature_value AS feature_value,
       pair.article_number AS article_number,
       g.npmi AS npmi
"""


def compute_graph_support(
    feature_evidences: list,
    article_evidences: list,
    min_co_count: int = _MIN_CO_COUNT,
) -> Optional[float]:
    """
    Cross-channel signal: do the feature and article evidences returned for
    this query actually co-occur more than chance, per the offline
    GROUNDED_IN/NPMI statistics computed in database/grounded_in_builder.py?

    Returns:
        None  -> not applicable (feature or article channel was not queried,
                 or returned nothing) — caller should treat this as skipped,
                 not as zero support.
        float in [0, 1] -> applicable; 0 means queried but no statistically
                 supported feature-article pair was found among the results.
    """
    if not feature_evidences or not article_evidences:
        return None

    feature_values = list({
        e.feature_value for e in feature_evidences
        if e.feature_value and e.feature_category != "LegalFact"
    })
    
    article_pairs = list({
        (e.article_number, e.law_name)
        for e in article_evidences if e.article_number and e.law_name
    })
    if not feature_values or not article_pairs:
        return None

    pairs = [
        {"feature_value": fv, "article_number": num, "law_name": law}
        for fv in feature_values
        for (num, law) in article_pairs
    ]

    try:
        with neo4j_client.session() as session:
            records = session.run(
                _GRAPH_SUPPORT_QUERY,
                pairs=pairs,
                min_co_count=min_co_count,
            )
            npmis = [float(r["npmi"]) for r in records]
    except Exception as e:
        logger.warning(f"Graph support query failed: {e}")
        return None

    if not npmis:
        return 0.0

    avg_npmi = sum(npmis) / len(npmis)
    # NPMI in [-1, 1]; negative or absent support is floored at 0 for the
    # quality scale, since negative co-occurrence isn't "anti-support" we
    # want to reward penalizing further beyond missing-support already does.
    return max(0.0, avg_npmi)