"""
embed_features.py — embedding نودهای Feature graph (لایه‌ی سوم)
    ۱. concept/action/role/object: کپی بردار از vocab_embeddings_cache.json
    ۲. fact: embed تازه (متن آزاد)
    ۳. ساخت vector index برای هر ۵ label
اجرا فقط بعد از این‌ها لازمه:
    - main_features.py load <domain> --limit 0 (برای همه‌ی domain ها)
    - بازسازی vocab_embeddings_cache.json
"""

import os
from dotenv import load_dotenv

from common.embedder import embed_text
from database.connection import Neo4jConnection
from database.embedding_store import EmbeddingStore

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")

FEATURE_LABELS = ["LegalConcept", "LegalAction", "LegalRole", "LegalObject", "LegalFact"]


def create_feature_vector_indexes(connection):
    with connection.session() as session:
        for label in FEATURE_LABELS:
            session.run(f"""
                CREATE VECTOR INDEX {label.lower()}_embedding IF NOT EXISTS
                FOR (n:{label}) ON n.embedding
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: 1024,
                    `vector.similarity_function`: 'cosine'
                }}}}
            """)
    print("✅ vector index های Feature graph آماده شدن.")


def run():
    connection = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASS)
    store = EmbeddingStore(connection)

    try:
        create_feature_vector_indexes(connection)

        print("📚 چسباندن embedding واژگان (concept/action/role/object)...")
        store.attach_vocab_embeddings()

        print("📝 embedding factها...")
        facts = store.get_facts_without_embedding()
        print(f"   {len(facts)} fact نیاز به embedding دارند")
        for i, fact in enumerate(facts, start=1):
            vector = embed_text(fact["text"])
            store.save_fact_embedding(fact["fact_id"], vector)
            if i % 50 == 0:
                print(f"   ... {i}/{len(facts)}")

    finally:
        connection.close()

    print("\n🎉 embedding کامل Feature graph تمام شد!")


if __name__ == "__main__":
    run()