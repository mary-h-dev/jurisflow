"""
database/case_loader.py — loads the case graph (Fact Graph) into Neo4j

Uses the same shared Neo4jConnection (see database/connection.py) —
exactly like law_loader.py, except this one is responsible for the
Ruling/RulingSection nodes.

Graph structure:
    (:Ruling {ruling_id, title, summary, verdict_number, verdict_date,
               case_type, url})
        -[:HAS_SECTION]->
    (:RulingSection {court_level, text})

    (:Ruling)-[:CITES {source: "official"}]->(:Article)   ← from the official "citations" field
    (:Ruling)-[:CITES {source: "text"}]->(:Article)        ← extracted from free text

    (:Ruling)-[:RELATED_TO]->(:Ruling)   ← other tiers of the same case

Note: the Article node must already have been created by laws/database
(Rule Graph). If an article number cannot be found in the Rule Graph
(e.g. an article cited by a very old case that refers to a now-abolished
statute), MERGE is used instead of MATCH so the relationship isn't lost —
but such an Article will remain without content (a signal that it needs
to be reviewed later).
"""

from database.connection import Neo4jConnection
from cases.schemas import Ruling, RulingSection, CitedArticle


def dict_to_ruling(data: dict) -> Ruling:
    def to_section(d):
        return RulingSection(**d)

    def to_cited(d):
        return CitedArticle(**d)

    data["sections"] = [to_section(s) for s in data.get("sections", [])]
    data["cited_articles"] = [to_cited(c) for c in data.get("cited_articles", [])]
    data["text_cited_articles"] = [to_cited(c) for c in data.get("text_cited_articles", [])]
    return Ruling(**data)


class CaseGraphLoader:
    def __init__(self, connection: Neo4jConnection):
        self.connection = connection

    def create_indexes(self):
        queries = [
            "CREATE INDEX IF NOT EXISTS FOR (r:Ruling) ON (r.ruling_id)",
            "CREATE INDEX IF NOT EXISTS FOR (r:Ruling) ON (r.case_type)",
        ]
        with self.connection.session() as session:
            for q in queries:
                session.run(q)
        print("✅ Fact Graph indexes are ready.")

    def load_ruling(self, ruling: Ruling):
        with self.connection.session() as session:
            session.execute_write(self._create_ruling, ruling)
            for section in ruling.sections:
                session.execute_write(self._create_section, ruling.ruling_id, section)
            for article in ruling.cited_articles:
                session.execute_write(self._link_article, ruling.ruling_id, article, "official")
            for article in ruling.text_cited_articles:
                session.execute_write(self._link_article, ruling.ruling_id, article, "text")
            for related_id in ruling.related_ruling_ids:
                if related_id != ruling.ruling_id:
                    session.execute_write(self._link_related, ruling.ruling_id, related_id)

    @staticmethod
    def _create_ruling(tx, ruling: Ruling):
        tx.run(
            """
            MERGE (r:Ruling {ruling_id: $id})
            SET r.title = $title, r.legal_factual_summary = $summary,
                r.verdict_number = $verdict_number, r.verdict_date = $verdict_date,
                r.case_type = $case_type, r.url = $url
            """,
            id=ruling.ruling_id, title=ruling.title, summary=ruling.legal_factual_summary,
            verdict_number=ruling.verdict_number, verdict_date=ruling.verdict_date,
            case_type=ruling.case_type, url=ruling.url,
        )

    @staticmethod
    def _create_section(tx, ruling_id: str, section: RulingSection):
        tx.run(
            """
            MATCH (r:Ruling {ruling_id: $ruling_id})
            CREATE (s:RulingSection {court_level: $level, text: $text})
            CREATE (r)-[:HAS_SECTION]->(s)
            """,
            ruling_id=ruling_id, level=section.court_level, text=section.text,
        )

    @staticmethod
    def _link_article(tx, ruling_id: str, article: CitedArticle, source: str):
        tx.run(
            """
            MATCH (r:Ruling {ruling_id: $ruling_id})
            MERGE (a:Article {article_number: $num, law: $law})
            MERGE (r)-[c:CITES]->(a)
            SET c.source = $source
            """,
            ruling_id=ruling_id, num=article.article_number,
            law=article.law_name, source=source,
        )

    @staticmethod
    def _link_related(tx, ruling_id: str, related_id: str):
        tx.run(
            """
            MATCH (r1:Ruling {ruling_id: $id1})
            MERGE (r2:Ruling {ruling_id: $id2})
            MERGE (r1)-[:RELATED_TO]->(r2)
            """,
            id1=ruling_id, id2=related_id,
        )