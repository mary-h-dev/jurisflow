"""
database/embedding_store.py — ذخیره‌ی embedding روی گره‌های Rule Graph و Fact Graph

نکته‌ی مهم درباره‌ی طراحی: هر واحد قابل‌جستجو (ماده، تبصره، عنوان پرونده،
خلاصه‌ی پرونده، هر بخش رأی) روی فیلد embedding *جدای خودش* ذخیره می‌شه —
نه یک embedding ترکیبی از همه‌چیز با هم. این عمداً اینطوریه: وقتی تابع
retrieval رو نوشتیم، باید بتونیم اثر هرکدوم (مثلاً فقط عنوان در برابر فقط
خلاصه) رو جدا جدا بسنجیم، نه یک عدد قاطی که معلوم نیست کدوم فیلد باعث
تطابق شده.

واحدهای قابل embedding:
    Rule Graph:
        - Article.embedding        ← متن اصلی ماده
        - Note.embedding            ← متن هر تبصره (جدا از ماده — چون یک
                                       تبصره می‌تونه استثنا/شرط کاملاً
                                       متفاوتی از متن اصلی ماده باشه)
    Fact Graph:
        - Ruling.title_embedding    ← عنوان (کوتاه، پرمعنا، برای جستجوی سریع)
        - Ruling.summary_embedding  ← «پیام» / legal_factual_summary
        - RulingSection.embedding   ← متن کامل هر لایه‌ی رأی

⚠️ تغییر تاریخ [رفع باگ]: در get_articles_without_embedding، شرط قبلی
   `a.status <> 'abolished'` وقتی a.status اصلاً NULL بود (نه رشته‌ی
   'abolished')، نتیجه‌ی مقایسه‌ش در Cypher به‌جای true/false، خودش NULL
   می‌شد — و هر شرط NULL در WHERE مثل false رفتار می‌کنه، یعنی اون ماده
   بی‌صدا از نتیجه حذف می‌شد. با coalesce(a.status, '') این مشکل رفع شده:
   اگه status نال باشه، به‌جاش رشته‌ی خالی در نظر گرفته می‌شه که مطمئناً
   با 'abolished' برابر نیست، پس ماده دیگه به‌اشتباه رد نمی‌شه.
"""

from database.connection import Neo4jConnection


class EmbeddingStore:
    def __init__(self, connection: Neo4jConnection):
        self.connection = connection

    def create_vector_indexes(self):
        index_specs = [
            ("article_embedding", "Article", "embedding"),
            ("note_embedding", "Note", "embedding"),
            ("ruling_title_embedding", "Ruling", "title_embedding"),
            ("ruling_summary_embedding", "Ruling", "summary_embedding"),
            ("ruling_section_embedding", "RulingSection", "embedding"),
        ]
        with self.connection.session() as session:
            for index_name, label, prop in index_specs:
                session.run(f"""
                    CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                    FOR (n:{label}) ON n.{prop}
                    OPTIONS {{indexConfig: {{
                        `vector.dimensions`: 1024,
                        `vector.similarity_function`: 'cosine'
                    }}}}
                """)
        print("✅ Vector index های هر دو گراف آماده‌اند (Article, Note, Ruling×۲, RulingSection).")

    # ── Rule Graph: مواد ────────────────────────────────────────────────

    def get_articles_without_embedding(self, law: str) -> list[dict]:
        with self.connection.session() as session:
            result = session.run(
                """
                MATCH (a:Article {law: $law})
                WHERE a.embedding IS NULL
                  AND coalesce(a.status, '') <> 'abolished'
                  AND a.content IS NOT NULL AND a.content <> ''
                RETURN a.article_number AS num, a.content AS content
                ORDER BY a.article_number
                """,
                law=law,
            )
            return [{"num": r["num"], "content": r["content"]} for r in result]

    def save_article_embedding(self, article_number: int, law: str, embedding: list[float]):
        with self.connection.session() as session:
            session.run(
                """
                MATCH (a:Article {article_number: $num, law: $law})
                SET a.embedding = $embedding
                """,
                num=article_number, law=law, embedding=embedding,
            )

    # ── Rule Graph: تبصره‌ها ────────────────────────────────────────────
    # نکته: چرا این مهمه؟ چون در cases/parser.py وقتی یک رأی به «تبصره N
    # از ماده M» استناد می‌کنه، ما اون رو در CitedArticle.note_number ثبت
    # کردیم. بدون embedding خودِ متن تبصره، هیچ‌وقت نمی‌تونیم موقع retrieval
    # مستقیم محتوای همون تبصره‌ی خاص رو (نه کل ماده رو) پیدا کنیم.

    def get_notes_without_embedding(self, law: str) -> list[dict]:
        with self.connection.session() as session:
            result = session.run(
                """
                MATCH (a:Article {law: $law})-[:HAS_NOTE]->(n:Note)
                WHERE n.embedding IS NULL
                  AND coalesce(n.status, '') <> 'abolished'
                  AND n.content IS NOT NULL AND n.content <> ''
                RETURN n.note_number AS note_num, n.article_number AS article_num, n.content AS content
                """,
                law=law,
            )
            return [
                {"note_num": r["note_num"], "article_num": r["article_num"], "content": r["content"]}
                for r in result
            ]

    def save_note_embedding(self, note_number: int, article_number: int, law: str, embedding: list[float]):
        with self.connection.session() as session:
            session.run(
                """
                MATCH (n:Note {note_number: $note_num, article_number: $article_num, law: $law})
                SET n.embedding = $embedding
                """,
                note_num=note_number, article_num=article_number, law=law, embedding=embedding,
            )

    # ── Fact Graph: عنوان پرونده ────────────────────────────────────────

    def get_ruling_titles_without_embedding(self) -> list[dict]:
        with self.connection.session() as session:
            result = session.run(
                """
                MATCH (r:Ruling)
                WHERE r.title_embedding IS NULL
                  AND r.title IS NOT NULL AND r.title <> ''
                RETURN r.ruling_id AS id, r.title AS text
                """
            )
            return [{"id": r["id"], "text": r["text"]} for r in result]

    def save_ruling_title_embedding(self, ruling_id: str, embedding: list[float]):
        with self.connection.session() as session:
            session.run(
                """
                MATCH (r:Ruling {ruling_id: $id})
                SET r.title_embedding = $embedding
                """,
                id=ruling_id, embedding=embedding,
            )

    # ── Fact Graph: خلاصه‌ی پرونده ──────────────────────────────────────

    def get_rulings_without_summary_embedding(self) -> list[dict]:
        with self.connection.session() as session:
            result = session.run(
                """
                MATCH (r:Ruling)
                WHERE r.summary_embedding IS NULL
                  AND r.legal_factual_summary IS NOT NULL
                  AND r.legal_factual_summary <> ''
                RETURN r.ruling_id AS id, r.legal_factual_summary AS text
                """
            )
            return [{"id": r["id"], "text": r["text"]} for r in result]

    def save_ruling_summary_embedding(self, ruling_id: str, embedding: list[float]):
        with self.connection.session() as session:
            session.run(
                """
                MATCH (r:Ruling {ruling_id: $id})
                SET r.summary_embedding = $embedding
                """,
                id=ruling_id, embedding=embedding,
            )

    # ── Fact Graph: بخش‌های رأی ─────────────────────────────────────────

    def get_sections_without_embedding(self) -> list[dict]:
        with self.connection.session() as session:
            result = session.run(
                """
                MATCH (r:Ruling)-[:HAS_SECTION]->(s:RulingSection)
                WHERE s.embedding IS NULL AND s.text IS NOT NULL AND s.text <> ''
                RETURN elementId(s) AS section_id, s.text AS text
                """
            )
            return [{"section_id": r["section_id"], "text": r["text"]} for r in result]

    def save_section_embedding(self, section_id: str, embedding: list[float]):
        with self.connection.session() as session:
            session.run(
                """
                MATCH (s:RulingSection) WHERE elementId(s) = $section_id
                SET s.embedding = $embedding
                """,
                section_id=section_id, embedding=embedding,
            )

    def get_facts_without_embedding(self) -> list[dict]:
        with self.connection.session() as session:
            result = session.run(
                """
                MATCH (f:LegalFact)
                WHERE f.embedding IS NULL AND f.text IS NOT NULL
                RETURN elementId(f) AS fact_id, f.text AS text
                """
            )
            return [{"fact_id": r["fact_id"], "text": r["text"]} for r in result]



    def save_fact_embedding(self, fact_id: str, embedding: list[float]):
        with self.connection.session() as session:
            session.run(
                "MATCH (f:LegalFact) WHERE elementId(f) = $id SET f.embedding = $embedding",
                id=fact_id, embedding=embedding,
            )



    def attach_vocab_embeddings(self):
        import json
        cache = json.load(open("data/legal-vocabulary/gap_audit/vocab_embeddings_cache.json", encoding="utf-8"))
        label_map = {
            "concept": "LegalConcept",
            "action": "LegalAction",
            "role": "LegalRole",
            "object": "LegalObject"
        }
        updated = 0
        with self.connection.session() as session:
            for entry in cache.values():
                label = label_map.get(entry["category"])
                if not label:
                    continue
                result = session.run(
                    f"MATCH (n:{label} {{name: $name}}) SET n.embedding = $embedding RETURN count(n) AS cnt",
                    name=entry["word"],    
                    embedding=entry["vector"],
                )
                updated += result.single()["cnt"]
        print(f"✅ {updated} نود واژگانی embedding گرفت.")





    # def attach_vocab_embeddings(self):
    #     """
    #     برای concept/action/role/object: بردار موجود در
    #     vocab_embeddings_cache.json را مستقیم به نودهای مربوطه وصل می‌کند —
    #     بدون embed کردن دوباره.
    #     """
    #     import json
    #     cache = json.load(open("data/legal-vocabulary/gap_audit/vocab_embeddings_cache.json", encoding="utf-8"))
    #     label_map = {"concept": "LegalConcept", "action": "LegalAction", "role": "LegalRole", "object": "LegalObject"}

    #     with self.connection.session() as session:
    #         for entry in cache.values():
    #             label = label_map.get(entry["category"])
    #             if not label:
    #                 continue
    #             session.run(
    #                 f"MATCH (n:{label} {{name: $value}}) SET n.embedding = $embedding",
    #                 value=entry["word"], embedding=entry["vector"],
    #             )