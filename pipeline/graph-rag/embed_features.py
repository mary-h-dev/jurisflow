"""
embed_features.py — embeds Feature graph nodes (third graph layer)
    1. concept/action/role/object: copies the vector from vocab_embeddings_cache.json
    2. fact: embedded fresh (free text)
    3. builds a vector index for all 5 labels
Only needs to run after:
    - main_features.py load <domain> --limit 0 (for all domains)
    - vocab_embeddings_cache.json has been rebuilt
    uv run embed_features.py
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
    print("✅ Feature graph vector indexes are ready.")


def run():
    connection = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASS)
    store = EmbeddingStore(connection)

    try:
        create_feature_vector_indexes(connection)

        print("📚 Attaching vocabulary embeddings (concept/action/role/object)...")
        store.attach_vocab_embeddings()

        print("📝 Embedding facts...")
        facts = store.get_facts_without_embedding()
        print(f"   {len(facts)} facts need embedding")
        for i, fact in enumerate(facts, start=1):
            vector = embed_text(fact["text"])
            store.save_fact_embedding(fact["fact_id"], vector)
            if i % 50 == 0:
                print(f"   ... {i}/{len(facts)}")

    finally:
        connection.close()

    print("\n🎉 Feature graph embedding complete!")


if __name__ == "__main__":
    run()