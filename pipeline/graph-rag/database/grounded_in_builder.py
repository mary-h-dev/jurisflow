"""Builds the GROUNDED_IN relationship between ontology features and articles,
and computes PMI/NPMI scores to normalize co-occurrence counts.

1. GROUNDED_IN Building:
   Standalone batch step, run after feature extraction and CITES loading are
   complete. Deterministic and statistical by design (no LLM), so the retrieval
   uncertainty signal it feeds does not share a source with the feature
   extraction step itself. Both 'official' and 'text' CITES sources are kept
   (official citations are a small minority of the corpus), with separate
   counters so downstream scoring can weight them differently. Procedural-law
   articles are excluded to avoid meaningless feature-article edges from a
   naive cartesian join.

2. PMI/NPMI Scoring:
   Adds PMI/NPMI scores to GROUNDED_IN edges to normalize raw co-occurrence
   counts against base rate. Without this, high-frequency generic features
   (e.g. plaintiff/defendant roles present in nearly every case) dominate
   scoring even though they carry no discriminative signal.

Property names follow database/law_loader.py and database/feature_loader.py:
Article uses article_number and law (not law_name); Concept/Action/Role/Object
use name. LegalFact is intentionally excluded (HAS_FACT is not in
_FEATURE_REL_TYPES) and stays free text, outside the ontology/GROUNDED_IN layer.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import dataclass

from dotenv import load_dotenv

from database.connection import Neo4jConnection

load_dotenv()

# Procedural-law domains, excluded from GROUNDED_IN because they are not
# substantive. Values must match Article.domain exactly, as produced by
# LawParseConfig.domain in laws/configs.py.
_PROCEDURAL_DOMAINS = {
    "دادرسی مدنی",
    "دادرسی کیفری",
}

_FEATURE_REL_TYPES = ["HAS_CONCEPT", "HAS_ROLE", "HAS_ACTION", "HAS_OBJECT"]
# HAS_FACT intentionally excluded — see module docstring.





_BUILD_QUERY = """
MATCH (r:Ruling)-[rel]->(feature)
WHERE type(rel) IN $feature_rel_types
MATCH (r)-[cites:CITES]->(article:Article)
WHERE NOT article.domain IN $procedural_domains
WITH feature, article,
     sum(CASE WHEN cites.source = 'official' THEN 1 ELSE 0 END) AS off_count,
     sum(CASE WHEN cites.source = 'text'     THEN 1 ELSE 0 END) AS txt_count
MERGE (feature)-[g:GROUNDED_IN]->(article)
SET g.official_count = off_count,
    g.text_count     = txt_count
"""


_CITES_SOURCE_STATS_QUERY = """
MATCH ()-[c:CITES]->()
RETURN c.source AS source, count(*) AS total
ORDER BY total DESC
"""

_VERIFY_SAMPLE_QUERY = """
MATCH (feature)-[g:GROUNDED_IN]->(article:Article)
RETURN labels(feature)[0] AS feature_label,
       feature.name AS feature_value,
       article.article_number AS article_number,
       article.law AS law_name,
       g.official_count AS official_count,
       g.text_count AS text_count
ORDER BY (g.official_count + g.text_count) DESC
LIMIT $limit
"""

_FETCH_EDGES_QUERY = """
MATCH (feature)-[g:GROUNDED_IN]->(article:Article)
RETURN elementId(g) AS rel_id,
       elementId(feature) AS feature_id,
       elementId(article) AS article_id,
       g.official_count + g.text_count AS co_count
"""

_UPDATE_PMI_QUERY = """
UNWIND $rows AS row
MATCH ()-[g:GROUNDED_IN]->() WHERE elementId(g) = row.rel_id
SET g.pmi = row.pmi, g.npmi = row.npmi
"""


@dataclass
class GroundedInBuildStats:
    edges_before: int
    edges_after: int
    edges_created: int

    def __str__(self) -> str:
        return (
            f"GROUNDED_IN: {self.edges_before} -> {self.edges_after} "
            f"(+{self.edges_created} new/updated edges)"
        )


@dataclass
class PmiStats:
    edges_scored: int
    total_co_occurrences: int


def build_grounded_in(connection: Neo4jConnection) -> GroundedInBuildStats:
    """Idempotent: re-running recomputes counters from CITES, does not duplicate edges."""
    with connection.session() as session:
        before = session.run(
            "MATCH ()-[g:GROUNDED_IN]->() RETURN count(g) AS c"
        ).single()["c"]

        session.run(
            _BUILD_QUERY,
            feature_rel_types=_FEATURE_REL_TYPES,
            procedural_domains=list(_PROCEDURAL_DOMAINS),
        )

        after = session.run(
            "MATCH ()-[g:GROUNDED_IN]->() RETURN count(g) AS c"
        ).single()["c"]

    return GroundedInBuildStats(
        edges_before=before,
        edges_after=after,
        edges_created=after - before,
    )


def print_cites_source_stats(connection: Neo4jConnection) -> None:
    """Run before build() as a sanity check on CITES source distribution."""
    with connection.session() as session:
        rows = session.run(_CITES_SOURCE_STATS_QUERY)
        print("CITES edge source distribution:")
        for row in rows:
            print(f"  {row['source']!r}: {row['total']}")


def sample_grounded_in(connection: Neo4jConnection, limit: int = 15) -> None:
    """Prints top edges by combined count for manual inspection."""
    with connection.session() as session:
        rows = session.run(_VERIFY_SAMPLE_QUERY, limit=limit)
        print(f"Top {limit} GROUNDED_IN edges by total count:")
        for row in rows:
            print(
                f"  [{row['feature_label']}] {row['feature_value']!r} "
                f"--GROUNDED_IN--> article {row['article_number']} ({row['law_name']}) "
                f"| official={row['official_count']} text={row['text_count']}"
            )


def compute_pmi_scores(connection: Neo4jConnection) -> PmiStats:
    """Reads all GROUNDED_IN edges, computes PMI/NPMI in Python (graph is
    small enough — tens of thousands of edges), writes scores back in one
    batched UNWIND. NPMI is bounded in [-1, 1]; use it, not raw PMI, for
    any downstream thresholding since raw PMI is unbounded and sensitive
    to low co_count noise.
    """
    with connection.session() as session:
        rows = list(session.run(_FETCH_EDGES_QUERY))

    edges = [
        (r["rel_id"], r["feature_id"], r["article_id"], r["co_count"])
        for r in rows
    ]

    feature_totals: Counter[str] = Counter()
    article_totals: Counter[str] = Counter()
    total_co = 0

    for _, feature_id, article_id, co_count in edges:
        feature_totals[feature_id] += co_count
        article_totals[article_id] += co_count
        total_co += co_count

    updates = []
    for rel_id, feature_id, article_id, co_count in edges:
        p_joint = co_count / total_co
        p_feature = feature_totals[feature_id] / total_co
        p_article = article_totals[article_id] / total_co

        pmi = math.log(p_joint / (p_feature * p_article))
        # NPMI: normalize to [-1, 1] using -log(p_joint) so common pairs
        # (like خواهان/خوانده -> high-frequency article) don't get an
        # inflated score just because co_count is large in absolute terms.
        npmi = pmi / (-math.log(p_joint))

        updates.append({"rel_id": rel_id, "pmi": pmi, "npmi": npmi})

    with connection.session() as session:
        session.run(_UPDATE_PMI_QUERY, rows=updates)

    return PmiStats(edges_scored=len(updates), total_co_occurrences=total_co)




_MIN_CO_COUNT_FOR_RANKING = 3  

def sample_pmi(connection: Neo4jConnection, limit: int = 15, min_co_count: int = _MIN_CO_COUNT_FOR_RANKING) -> None:
    query = """
    MATCH (feature)-[g:GROUNDED_IN]->(article:Article)
    WHERE g.npmi IS NOT NULL
      AND (g.official_count + g.text_count) >= $min_co_count
    RETURN labels(feature)[0] AS feature_label,
           feature.name AS feature_value,
           article.article_number AS article_number,
           article.law AS law_name,
           g.npmi AS npmi,
           g.official_count + g.text_count AS co_count
    ORDER BY g.npmi DESC
    LIMIT $limit
    """
    with connection.session() as session:
        rows = session.run(query, limit=limit, min_co_count=min_co_count)
        print(f"Top {limit} GROUNDED_IN edges by NPMI (co_count >= {min_co_count}):")
        for row in rows:
            print(
                f"  [{row['feature_label']}] {row['feature_value']!r} "
                f"--GROUNDED_IN--> article {row['article_number']} ({row['law_name']}) "
                f"| npmi={row['npmi']:.3f} co_count={row['co_count']}"
            )




if __name__ == "__main__":
    connection = Neo4jConnection(
        os.getenv("NEO4J_URI"),
        os.getenv("NEO4J_USERNAME"),
        os.getenv("NEO4J_PASSWORD"),
    )
    try:
        print_cites_source_stats(connection)
        print()
        stats = build_grounded_in(connection)
        print(stats)
        print()
        sample_grounded_in(connection)

        print()
        pmi_stats = compute_pmi_scores(connection)
        print(
            f"PMI scored: {pmi_stats.edges_scored} edges "
            f"(total co-occurrences: {pmi_stats.total_co_occurrences})"
        )
        print()
        sample_pmi(connection)
    finally:
        connection.close()