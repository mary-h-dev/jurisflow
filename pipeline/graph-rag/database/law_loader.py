from database.connection import Neo4jConnection
from laws.schemas import Law, Article, Note


def dict_to_law(data: dict) -> Law:
    """Converts JSON read from a file into a Law object."""

    def to_note(d):
        return Note(**d)

    def to_article(d):
        d["notes"] = [to_note(n) for n in d.get("notes", [])]
        return Article(**d)

    data["articles"] = [to_article(a) for a in data.get("articles", [])]
    return Law(**data)


class LawGraphLoader:
    def __init__(self, connection: Neo4jConnection):
        self.connection = connection

    def create_indexes(self):
        queries = [
            "CREATE INDEX IF NOT EXISTS FOR (a:Article) ON (a.article_number)",
            "CREATE INDEX IF NOT EXISTS FOR (a:Article) ON (a.domain)",
            "CREATE INDEX IF NOT EXISTS FOR (a:Article) ON (a.case_type)",
            "CREATE INDEX IF NOT EXISTS FOR (a:Article) ON (a.law)",
            "CREATE INDEX IF NOT EXISTS FOR (l:Law) ON (l.name)",
        ]
        with self.connection.session() as session:
            for q in queries:
                session.run(q)
        print("✅ Rule Graph indexes are ready.")

    def load_law(self, law: Law):
        print(f"📥 Starting load: {law.name} ({law.domain}) — {len(law.articles)} articles")
        with self.connection.session() as session:
            session.execute_write(self._create_law, law)
            for article in law.articles:
                session.execute_write(self._load_article, article, law.name)
            session.execute_write(self._create_references, law.name)
        print("✅ Loading finished.")

    @staticmethod
    def _create_law(tx, law: Law):
        tx.run(
            "MERGE (l:Law {name: $name}) SET l.url = $url, l.domain = $domain, l.case_type = $case_type",
            name=law.name, url=law.url, domain=law.domain, case_type=law.case_type,
        )

    @staticmethod
    def _load_article(tx, article: Article, law_name: str):
        tx.run(
            """
            MERGE (a:Article {article_number: $num, law: $law})
            SET a.content    = $content,
                a.status     = $status,
                a.domain     = $domain,
                a.case_type  = $case_type,
                a.references = $refs
            """,
            num=article.article_number, law=law_name,
            content=article.content, status=article.status,
            domain=article.domain, case_type=article.case_type, refs=article.references,
        )
        tx.run(
            """
            MATCH (l:Law {name: $law})
            MATCH (a:Article {article_number: $num, law: $law})
            MERGE (l)-[:CONTAINS]->(a)
            """,
            law=law_name, num=article.article_number,
        )
        for note in article.notes:
            tx.run(
                """
                MATCH (a:Article {article_number: $num, law: $law})
                MERGE (n:Note {note_number: $note_num, article_number: $num, law: $law})
                SET n.content = $content, n.status = $status
                MERGE (a)-[:HAS_NOTE]->(n)
                """,
                num=article.article_number, law=law_name,
                note_num=note.note_number, content=note.content,
                status=note.status,
            )

    @staticmethod
    def _create_references(tx, law_name: str):
        tx.run(
            """
            MATCH (a:Article {law: $law})
            WHERE size(a.references) > 0
            UNWIND a.references AS ref_num
            MATCH (b:Article {article_number: ref_num, law: $law})
            MERGE (a)-[:REFERENCES]->(b)
            """,
            law=law_name,
        )