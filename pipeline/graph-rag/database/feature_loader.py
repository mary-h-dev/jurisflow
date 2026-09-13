"""
features/database/feature_loader.py — loads the third graph layer
(Feature Graph) into Neo4j

Uses the same shared Neo4jConnection (database/connection.py), exactly
like database/case_loader.py and database/law_loader.py.

Graph structure:
    (:Ruling)-[:HAS_CONCEPT   {evidence, start_char, end_char, confidence}]->(:LegalConcept   {name})
    (:Ruling)-[:HAS_ACTION    {...}]->(:LegalAction    {name})
    (:Ruling)-[:HAS_ROLE      {...}]->(:LegalRole      {name})
    (:Ruling)-[:HAS_OBJECT    {...}]->(:LegalObject    {name})
    (:Ruling)-[:HAS_FACT      {...}]->(:LegalFact      {text})

Why are closed-vocab nodes MERGEd but LegalFact isn't?
    Because Concept/Action/Role/Object come from a closed, deduplicated
    list (after canonicalization in vocabulary_categorizer.py) — a single
    node for "بیع" across the whole graph is enough, and MERGE means all
    related rulings connect to that one node (which is exactly what makes
    querying powerful). Facts, however, are free and highly varied; if we
    MERGEd them too, similar-but-not-identical Facts (very common in
    legal text) would be incorrectly merged into one node and each case's
    specific information would be lost. So every Fact gets its own
    independent node via CREATE, even if it looks similar to a Fact in
    another case.

Why are the relationships themselves (not just the nodes) created with
CREATE, not MERGE?
    It used to be `MERGE (r)-[rel:...]->(n)`, which had a real bug: if a
    concept (e.g. "فسخ") was mentioned twice in different parts of one
    ruling, MERGE would overwrite the first evidence with the second, and
    the first piece of evidence was lost forever. With CREATE, every time
    a feature is seen in a ruling it gets its own separate relationship
    with its own evidence; Neo4j allows multiple relationships of the
    same type between two nodes, so this causes no problem, and querying
    is unaffected (only the evidence stays more complete).

    This change has one side effect: if load_result() is called twice for
    the same ruling (e.g. re-running main_features.py load), relationships
    would be duplicated. That's why is_ruling_loaded/mark_ruling_loaded
    was added — before loading each ruling, it's checked that it hasn't
    already been loaded (idempotency at the ruling level, not at the
    individual-relationship level).

Why retry around Neo4j writes?
    Because main_features.py is meant to run across hundreds/thousands of
    rulings in sequence; a momentary network or Neo4j Aura outage
    shouldn't crash the whole run and lose the work done so far.
"""


import time

from database.connection import Neo4jConnection
from features.configs import FACT_CATEGORY, VOCAB_CATEGORIES
from features.schemas import ExtractedFeature, FeatureExtractionResult

# category key → list-field name mapping in FeatureExtractionResult
_FIELD_NAMES = {
    "concept": "concepts",
    "action": "actions",
    "role": "roles",
    "object": "objects",
}

_MAX_RETRIES = 3
_RETRY_DELAY_SECONDS = 5


def _with_retry(fn, *args, **kwargs):
    last_error = None
    for attempt in range(_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 — network/Neo4j outages should be caught here too
            last_error = e
            if attempt < _MAX_RETRIES - 1:
                print(f"   ⏳ Temporary Neo4j error, retrying "
                      f"({attempt + 1}/{_MAX_RETRIES}): {e}")
                time.sleep(_RETRY_DELAY_SECONDS)
    raise RuntimeError(f"❌ Neo4j write failed after {_MAX_RETRIES} attempts: {last_error}")


class FeatureGraphLoader:
    def __init__(self, connection: Neo4jConnection):
        self.connection = connection

    def create_indexes(self):
        queries = [
            f"CREATE INDEX IF NOT EXISTS FOR (n:{cat.node_label}) ON (n.name)"
            for cat in VOCAB_CATEGORIES.values()
        ]
        with self.connection.session() as session:
            for q in queries:
                _with_retry(session.run, q)
        print("✅ Feature Graph indexes are ready.")

    def ruling_exists(self, ruling_id: str) -> bool:
        """
        Checks whether a :Ruling node exists in the database, to prevent a
        silent failure on orphan cases.
        """
        ruling_id_str = str(ruling_id)
        with self.connection.session() as session:
            result = _with_retry(
                session.run,
                "MATCH (r:Ruling {ruling_id: $ruling_id}) RETURN count(r) AS cnt",
                ruling_id=ruling_id_str,
            )
            record = result.single()
            return bool(record and record["cnt"] > 0)

    def is_ruling_loaded(self, ruling_id: str) -> bool:
        """
        Idempotency check: has this ruling already been fully loaded?
        """
        ruling_id_str = str(ruling_id)
        with self.connection.session() as session:
            result = _with_retry(
                session.run,
                "MATCH (r:Ruling {ruling_id: $ruling_id}) RETURN r.features_loaded AS loaded",
                ruling_id=ruling_id_str,
            )
            record = result.single()
            return bool(record and record["loaded"])

    def load_result(self, result: FeatureExtractionResult):
        ruling_id_str = str(result.ruling_id)

        # 1. verify the Ruling node exists before loading (prevents silent failure)
        if not self.ruling_exists(ruling_id_str):
            raise ValueError(
                f"❌ :Ruling node not found for ruling_id='{ruling_id_str}' in Neo4j (orphan case)."
            )

        # 2. load features and mark success
        with self.connection.session() as session:
            for category_key, field_name in _FIELD_NAMES.items():
                cat = VOCAB_CATEGORIES[category_key]
                for item in getattr(result, field_name):
                    _with_retry(session.execute_write, self._link_closed_vocab, ruling_id_str, item, cat)
            for fact in result.facts:
                _with_retry(session.execute_write, self._link_fact, ruling_id_str, fact)
            _with_retry(
                session.run,
                "MATCH (r:Ruling {ruling_id: $ruling_id}) SET r.features_loaded = true",
                ruling_id=ruling_id_str,
            )

    @staticmethod
    def _link_closed_vocab(tx, ruling_id: str, item: ExtractedFeature, cat):
        tx.run(
            f"""
            MATCH (r:Ruling {{ruling_id: $ruling_id}})
            MERGE (n:{cat.node_label} {{name: $value}})
            CREATE (r)-[rel:{cat.relation_type}]->(n)
            SET rel.evidence = $quote, rel.start_char = $start, rel.end_char = $end,
                rel.confidence = $confidence
            """,
            ruling_id=ruling_id,
            value=item.value,
            quote=item.evidence.quote,
            start=item.evidence.start_char,
            end=item.evidence.end_char,
            confidence=item.evidence.confidence,
        )

    @staticmethod
    def _link_fact(tx, ruling_id: str, item: ExtractedFeature):
        tx.run(
            f"""
            MATCH (r:Ruling {{ruling_id: $ruling_id}})
            CREATE (f:{FACT_CATEGORY.node_label} {{text: $value}})
            CREATE (r)-[rel:{FACT_CATEGORY.relation_type}]->(f)
            SET rel.evidence = $quote, rel.start_char = $start, rel.end_char = $end,
                rel.confidence = $confidence
            """,
            ruling_id=ruling_id,
            value=item.value,
            quote=item.evidence.quote,
            start=item.evidence.start_char,
            end=item.evidence.end_char,
            confidence=item.evidence.confidence,
        )