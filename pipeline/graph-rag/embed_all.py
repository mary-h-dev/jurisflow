"""
embed_all.py — generates embeddings for the Rule Graph and Fact Graph

Dynamic with respect to statutes: like main.py and main_cases.py, this
file also loops directly over laws.configs.LAW_CONFIGS — not a manual list.

⚠️ New addition — rules-extra: besides the 5 core statutes in
LAW_CONFIGS, dozens of other values for a.law were found in practice in
the graph (e.g. "قانون حمایت خانواده" [Family Protection Act], "قانون
امور حسبی", and even messy strings like "و ۲۳۰ قانون مدنی" which are
actually a fragment of a reference, not an independent statute name).
Decision: we embed all of these too — because embedding runs on the
article's `content` itself, not on the statute name; even if the "which
statute" label is dirty, the article's own content is still real and
usable. Cleaning up the `law` label itself (fixing the reference-
extraction parser) is a separate task, for later.

Usage:
    uv run embed_all.py rules            ← only the 5 core statutes, all of them
    uv run embed_all.py rules civil      ← only one specific core statute
    uv run embed_all.py rules-extra      ← every a.law value outside the 5 core statutes
    uv run embed_all.py cases            ← ruling summaries + sections
    uv run embed_all.py all              ← rules + rules-extra + cases
"""

import os
import sys
import time
from dotenv import load_dotenv
from neo4j.exceptions import Neo4jError, ServiceUnavailable, SessionExpired

from common.embedder import embed_batch
from database.connection import Neo4jConnection
from database.embedding_store import EmbeddingStore
from laws.configs import LAW_CONFIGS

load_dotenv()

NEO4J_URI  = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")

BATCH_SIZE = 12   # how many texts to process together per model call


def _save_with_retry(save_fn, item_id, embedding, retries: int = 3, delay: float = 5.0):
    """
    The connection to Neo4j Aura (or any other cloud server) sometimes
    drops due to a transient network outage / idle timeout. Since each
    save_xxx_embedding opens a fresh session every time, a simple retry is
    usually enough — the driver creates a new connection on its own. If it
    still fails after a few attempts, we skip this one item (rather than
    losing the entire remaining 1000+ items).
    """
    for attempt in range(retries):
        try:
            save_fn(item_id, embedding)
            return True
        except (ServiceUnavailable, SessionExpired, Neo4jError) as e:
            if attempt < retries - 1:
                print(f"  ⏳ Neo4j connection outage, retrying ({attempt + 1}/{retries}) after {delay}s...")
                time.sleep(delay)
            else:
                print(f"  ❌ Saving {item_id} failed after {retries} attempts — skipped: {e}")
                return False


def _embed_and_save(items: list[dict], text_key: str, save_fn, id_key: str, max_length: int | None = None):
    """
    Shared pattern: build a batch, embed it, save it. Used for all three
    node types (Article/Ruling/RulingSection).
    """
    total = len(items)
    for i in range(0, total, BATCH_SIZE):
        batch = items[i:i + BATCH_SIZE]
        texts = [it[text_key] for it in batch]
        embeddings = embed_batch(texts, batch_size=BATCH_SIZE, max_length=max_length)

        for item, emb in zip(batch, embeddings):
            _save_with_retry(save_fn, item[id_key], emb)

        print(f"  ✅ {min(i + BATCH_SIZE, total)}/{total}")


def _embed_one_law(store: EmbeddingStore, law_name: str):
    """
    Logic shared between embed_rules (the 5 core statutes) and
    embed_extra_laws (everything else) — written once so both behave
    exactly the same way.
    """
    articles = store.get_articles_without_embedding(law_name)
    print(f"\n📊 {law_name[:80]}: {len(articles)} articles need embedding")

    if articles:
        _embed_and_save(
            items=[{"id": (a["num"], law_name), "content": a["content"]} for a in articles],
            text_key="content",
            save_fn=lambda id_pair, emb: store.save_article_embedding(id_pair[0], id_pair[1], emb),
            id_key="id",
            max_length=512,
        )

    notes = store.get_notes_without_embedding(law_name)
    if notes:
        print(f"📊 {law_name[:80]}: {len(notes)} notes need embedding")
        _embed_and_save(
            items=[
                {"id": (n["note_num"], n["article_num"], law_name), "content": n["content"]}
                for n in notes
            ],
            text_key="content",
            save_fn=lambda id_triple, emb: store.save_note_embedding(
                id_triple[0], id_triple[1], id_triple[2], emb
            ),
            id_key="id",
            max_length=512,
        )


def embed_rules(law_key: str | None = None):
    connection = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASS)
    store = EmbeddingStore(connection)

    try:
        store.create_vector_indexes()
        keys = [law_key] if law_key else list(LAW_CONFIGS.keys())

        for key in keys:
            if key not in LAW_CONFIGS:
                print(f"❌ Unknown key: {key}")
                continue
            _embed_one_law(store, LAW_CONFIGS[key].law_name)
    finally:
        connection.close()

    print("\n🎉 Core statute embedding finished!")


def get_extra_law_values(connection: Neo4jConnection) -> list[str]:
    """
    Every a.law value that isn't one of the 5 core statutes in
    LAW_CONFIGS — including both genuine secondary statutes (like "قانون
    حمایت خانواده") and messy strings caused by parser errors
    (deliberately not filtered out, see the note at the top of this file).
    """
    known = {cfg.law_name for cfg in LAW_CONFIGS.values()}
    with connection.session() as session:
        result = session.run("MATCH (a:Article) WHERE a.law IS NOT NULL RETURN DISTINCT a.law AS law")
        return sorted({r["law"] for r in result if r["law"] not in known})


def embed_extra_laws():
    connection = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASS)
    store = EmbeddingStore(connection)

    try:
        store.create_vector_indexes()
        extra_laws = get_extra_law_values(connection)
        print(f"📚 {len(extra_laws)} 'law' values found outside the 5 core statutes")

        for law_name in extra_laws:
            _embed_one_law(store, law_name)
    finally:
        connection.close()

    print("\n🎉 Secondary/miscellaneous statute embedding finished too!")


def embed_cases():
    connection = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASS)
    store = EmbeddingStore(connection)

    try:
        store.create_vector_indexes()

        titles = store.get_ruling_titles_without_embedding()
        print(f"\n📊 {len(titles)} ruling titles need embedding")
        if titles:
            _embed_and_save(
                items=titles, text_key="text",
                save_fn=store.save_ruling_title_embedding, id_key="id",
                max_length=512,
            )

        rulings = store.get_rulings_without_summary_embedding()
        print(f"\n📊 {len(rulings)} ruling summaries need embedding")
        if rulings:
            _embed_and_save(
                items=rulings, text_key="text",
                save_fn=store.save_ruling_summary_embedding, id_key="id",
                max_length=2048,
            )

        sections = store.get_sections_without_embedding()
        print(f"\n📊 {len(sections)} ruling sections need embedding")
        if sections:
            _embed_and_save(
                items=sections, text_key="text",
                save_fn=store.save_section_embedding, id_key="section_id",
                max_length=8192,
            )
    finally:
        connection.close()

    print("\n🎉 Case embedding finished!")


def _usage():
    keys = "|".join(LAW_CONFIGS.keys())
    print("Usage:")
    print(f"  uv run embed_all.py rules [<{keys}>]")
    print("  uv run embed_all.py rules-extra")
    print("  uv run embed_all.py cases")
    print("  uv run embed_all.py all")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _usage()
        sys.exit(1)

    mode = sys.argv[1]
    extra = sys.argv[2] if len(sys.argv) > 2 else None

    if mode == "rules":
        embed_rules(extra)
    elif mode == "rules-extra":
        embed_extra_laws()
    elif mode == "cases":
        embed_cases()
    elif mode == "all":
        embed_rules()
        embed_extra_laws()
        embed_cases()
    else:
        _usage()