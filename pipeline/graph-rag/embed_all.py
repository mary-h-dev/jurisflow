"""
embed_all.py — تولید embedding برای Rule Graph و Fact Graph

پویا بودن نسبت به قوانین: مثل main.py و main_cases.py، این فایل هم
مستقیم از laws.configs.LAW_CONFIGS حلقه می‌زنه — نه یک لیست دستی.

نحوه‌ی استفاده:
    uv run embed_all.py rules            ← فقط مواد قانونی، هر ۵ قانون
    uv run embed_all.py rules civil      ← فقط یک قانون خاص
    uv run embed_all.py cases            ← خلاصه + بخش‌های رأی‌ها
    uv run embed_all.py all              ← هر دو
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

BATCH_SIZE = 12   # چند متن در هر فراخوانی مدل با هم پردازش بشن


def _save_with_retry(save_fn, item_id, embedding, retries: int = 3, delay: float = 5.0):
    """
    اتصال به Neo4j Aura (یا هر سرور ابری دیگه) گاهی به‌خاطر قطعی موقت
    شبکه/idle-timeout قطع می‌شه. چون هر save_xxx_embedding خودش هر بار
    یک session جدید باز می‌کنه، صرفاً تلاش دوباره معمولاً کافیه — درایور
    خودش کانکشن جدید می‌سازه. اگه بعد از چند تلاش هم جواب نداد، این یک
    آیتم رو رد می‌کنیم (به‌جای این‌که کل باقی‌مونده‌ی ۱۰۰۰+ آیتم از بین بره).
    """
    for attempt in range(retries):
        try:
            save_fn(item_id, embedding)
            return True
        except (ServiceUnavailable, SessionExpired, Neo4jError) as e:
            if attempt < retries - 1:
                print(f"  ⏳ قطعی اتصال Neo4j، تلاش دوباره ({attempt + 1}/{retries}) بعد از {delay} ثانیه...")
                time.sleep(delay)
            else:
                print(f"  ❌ ذخیره‌ی {item_id} بعد از {retries} تلاش شکست خورد — رد شد: {e}")
                return False


def _embed_and_save(items: list[dict], text_key: str, save_fn, id_key: str, max_length: int | None = None):
    """
    الگوی مشترک: batch بساز، embed کن، ذخیره کن. برای هر سه نوع گره
    (Article/Ruling/RulingSection) همین تابع استفاده می‌شه.
    """
    total = len(items)
    for i in range(0, total, BATCH_SIZE):
        batch = items[i:i + BATCH_SIZE]
        texts = [it[text_key] for it in batch]
        embeddings = embed_batch(texts, batch_size=BATCH_SIZE, max_length=max_length)

        for item, emb in zip(batch, embeddings):
            _save_with_retry(save_fn, item[id_key], emb)

        print(f"  ✅ {min(i + BATCH_SIZE, total)}/{total}")


def embed_rules(law_key: str | None = None):
    connection = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASS)
    store = EmbeddingStore(connection)

    try:
        store.create_vector_indexes()
        keys = [law_key] if law_key else list(LAW_CONFIGS.keys())

        for key in keys:
            if key not in LAW_CONFIGS:
                print(f"❌ کلید ناشناخته: {key}")
                continue

            law_name = LAW_CONFIGS[key].law_name
            articles = store.get_articles_without_embedding(law_name)
            print(f"\n📊 {law_name}: {len(articles)} ماده نیاز به embedding دارند")

            if not articles:
                continue

            _embed_and_save(
                items=[{"id": (a["num"], law_name), "content": a["content"]} for a in articles],
                text_key="content",
                save_fn=lambda id_pair, emb: store.save_article_embedding(id_pair[0], id_pair[1], emb),
                id_key="id",
                max_length=512,
            )

            notes = store.get_notes_without_embedding(law_name)
            print(f"📊 {law_name}: {len(notes)} تبصره نیاز به embedding دارند")
            if notes:
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
    finally:
        connection.close()

    print("\n🎉 embedding قوانین تمام شد!")


def embed_cases():
    connection = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASS)
    store = EmbeddingStore(connection)

    try:
        store.create_vector_indexes()

        titles = store.get_ruling_titles_without_embedding()
        print(f"\n📊 {len(titles)} عنوان پرونده نیاز به embedding دارند")
        if titles:
            _embed_and_save(
                items=titles, text_key="text",
                save_fn=store.save_ruling_title_embedding, id_key="id",
                max_length=512,
            )

        rulings = store.get_rulings_without_summary_embedding()
        print(f"\n📊 {len(rulings)} خلاصه‌ی رأی نیاز به embedding دارند")
        if rulings:
            _embed_and_save(
                items=rulings, text_key="text",
                save_fn=store.save_ruling_summary_embedding, id_key="id",
                max_length=2048,
            )

        sections = store.get_sections_without_embedding()
        print(f"\n📊 {len(sections)} بخش رأی نیاز به embedding دارند")
        if sections:
            _embed_and_save(
                items=sections, text_key="text",
                save_fn=store.save_section_embedding, id_key="section_id",
                max_length=8192,
            )
    finally:
        connection.close()

    print("\n🎉 embedding پرونده‌ها تمام شد!")


def _usage():
    keys = "|".join(LAW_CONFIGS.keys())
    print("استفاده:")
    print(f"  uv run embed_all.py rules [<{keys}>]")
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
    elif mode == "cases":
        embed_cases()
    elif mode == "all":
        embed_rules()
        embed_cases()
    else:
        _usage()