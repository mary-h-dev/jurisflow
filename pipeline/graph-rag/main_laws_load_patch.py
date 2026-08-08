"""
main_laws_load_patch.py — نسخه‌ی resumable و مقاوم در برابر قطعی شبکه
                           برای main.py (بارگذاری Rule Graph / قوانین)

چرا شکلش کمی با main_cases_load_patch.py فرق دارد؟
    در Fact Graph (پرونده‌ها) هر رأی یک فایل JSON جدا بود، پس رزومه‌پذیری
    در سطح «فایل» معنا داشت. اینجا برعکس است: هر قانون (civil, penal, ...)
    فقط یک فایل JSON دارد که می‌تواند صدها ماده داخلش باشد. پس رزومه‌پذیری
    را در سطح «هر ماده» پیاده کرده‌ایم — قبل از نوشتن هر Article، چک
    می‌کنیم آیا از قبل با همان content کامل ذخیره شده یا نه.

نحوه‌ی استفاده: دقیقاً مثل main.py —
    uv run main_laws_load_patch.py load civil
    uv run main_laws_load_patch.py load all

اگر وسط کار اینترنت قطع شد، لازم نیست کاری کنی — اسکریپت خودش صبر
می‌کند و ادامه می‌دهد. اگر کامل قطع شد (Codespace ری‌استارت شد)، فقط
دوباره همین دستور را بزن؛ مواد قبلاً لودشده رد می‌شوند.

⚠️ توجه: مرحله‌ی scrape دست‌نخورده باقی مانده — این فایل فقط load را
جایگزین می‌کند. برای scrape همچنان از main.py اصلی استفاده کن:
    uv run main.py scrape <key>
"""

import os
import sys
import json
import time
import traceback

from dotenv import load_dotenv
from neo4j.exceptions import Neo4jError, ServiceUnavailable, SessionExpired

from laws.configs import LAW_CONFIGS
from laws.schemas import Law, Article
from database.connection import Neo4jConnection
from database.law_loader import LawGraphLoader, dict_to_law

load_dotenv()

NEO4J_URI  = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")

DATA_DIR = "data/laws"
ERRORS_DIR = "data/features"

MAX_RETRIES = 5
RETRY_BASE_DELAY = 5   # ثانیه — 5, 10, 20, 40, 80

TRANSIENT_EXCEPTIONS = (
    ServiceUnavailable,
    SessionExpired,
    ConnectionError,
    TimeoutError,
    OSError,
)


def _raw_json_path(key: str) -> str:
    return f"{DATA_DIR}/{key}_law.json"


def _errors_path(key: str) -> str:
    os.makedirs(ERRORS_DIR, exist_ok=True)
    return f"{ERRORS_DIR}/{key}_law_load_errors.json"


# ─────────────────────────────────────────────────────────────────────────
# چک کردن این‌که آیا یک Article از قبل «کامل» ذخیره شده یا نه
# ─────────────────────────────────────────────────────────────────────────

def _article_already_loaded(loader: LawGraphLoader, article: Article, law_name: str) -> bool:
    """
    True اگر Article با همین content از قبل موجود است *و* تعداد Note
    هایش با تعداد note های همین article در فایل JSON برابر است — یعنی
    قبلاً کامل لود شده، نه این‌که وسط کار قطع شده باشد.
    """
    query = """
    MATCH (a:Article {article_number: $num, law: $law})
    WHERE a.content = $content
    OPTIONAL MATCH (a)-[:HAS_NOTE]->(n:Note)
    RETURN count(n) AS note_count
    """
    with loader.connection.session() as session:
        result = session.run(
            query, num=article.article_number, law=law_name, content=article.content
        )
        record = result.single()
        if record is None:
            return False
        return record["note_count"] >= len(article.notes)


def _connect_with_retry() -> Neo4jConnection:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASS)
        except TRANSIENT_EXCEPTIONS as e:
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"   ⚠️ اتصال ناموفق (تلاش {attempt}/{MAX_RETRIES}): {e}")
            print(f"      {delay} ثانیه صبر می‌کنیم و دوباره تلاش می‌کنیم...")
            time.sleep(delay)
    return Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASS)


def _run_with_retry(fn, *args, **kwargs):
    """
    اجرای یک عملیات (مثلاً _create_law یا لود یک ماده) با retry در
    برابر خطاهای شبکه‌ای موقت. اگر بعد از MAX_RETRIES هم نشد،
    (False, پیام‌خطا) برمی‌گرداند؛ در غیر این‌صورت (True, None).
    """
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            fn(*args, **kwargs)
            return True, None
        except TRANSIENT_EXCEPTIONS as e:
            last_error = f"{type(e).__name__}: {e}"
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(f"      ⏳ خطای شبکه‌ای موقت (تلاش {attempt}/{MAX_RETRIES}) — "
                      f"{delay} ثانیه صبر و تلاش دوباره... ({last_error})")
                time.sleep(delay)
            else:
                return False, last_error
        except Neo4jError as e:
            # خطای منطقی/داده‌ای — retry فایده ندارد
            return False, f"{type(e).__name__}: {e}"
    return False, last_error


def _load_article_with_retry(loader: LawGraphLoader, article: Article, law_name: str) -> tuple[bool, str | None]:
    """
    معادل _load_article داخل LawGraphLoader، ولی به‌جای یک session.execute_write
    که همه‌چیز (article + رابطه با Law + همه‌ی note ها) را در یک تراکنش
    می‌گذارد، همان منطق را با retry اجرا می‌کند.
    """
    def _do_load():
        with loader.connection.session() as session:
            session.execute_write(loader._load_article, article, law_name)

    return _run_with_retry(_do_load)


# ─────────────────────────────────────────────────────────────────────────
# نسخه‌ی resumable + مقاوم load_one
# ─────────────────────────────────────────────────────────────────────────

def load_one_safe(key: str, loader: LawGraphLoader):
    raw_json = _raw_json_path(key)

    if not os.path.exists(raw_json):
        print(f"❌ فایل JSON یافت نشد: {raw_json} — اول scrape کن.")
        return

    with open(raw_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    law: Law = dict_to_law(data)
    total = len(law.articles)
    print(f"\n📂 {law.name} ({law.domain}): {total} ماده پیدا شد. "
          f"در حال چک کردن مواد قبلاً لودشده...")

    # ── مرحله‌ی Law node — با retry ──
    ok, err = _run_with_retry(_create_law_tx, loader, law)
    if not ok:
        print(f"❌ ساخت گره‌ی Law برای {law.name} شکست خورد: {err}")
        print("   بدون گره‌ی Law، ادامه دادن معنا ندارد — این قانون رد شد.")
        return

    loaded = 0
    already_done = 0
    errors: list[dict] = []

    start_time = time.time()

    for i, article in enumerate(law.articles, 1):
        try:
            if _article_already_loaded(loader, article, law.name):
                already_done += 1
                if i % 50 == 0:
                    elapsed = time.time() - start_time
                    print(f"   ... {i}/{total} بررسی شد "
                          f"({loaded} تازه لود شد، {already_done} قبلاً بوده، {len(errors)} خطا) "
                          f"— {elapsed:.0f} ثانیه")
                continue
        except TRANSIENT_EXCEPTIONS:
            print(f"   ⚠️ قطعی موقت هنگام چک کردن ماده‌ی {article.article_number} — ۱۰ ثانیه صبر...")
            time.sleep(10)

        ok, err = _load_article_with_retry(loader, article, law.name)

        if ok:
            loaded += 1
        else:
            errors.append({
                "article_number": article.article_number,
                "law": law.name,
                "error_message": err,
            })
            print(f"   🔴 [{i}/{total}] لود ماده‌ی {article.article_number} "
                  f"بعد از {MAX_RETRIES} تلاش شکست خورد: {err}")

        if i % 50 == 0:
            elapsed = time.time() - start_time
            print(f"   ... {i}/{total} پردازش شد "
                  f"({loaded} تازه لود شد، {already_done} قبلاً بوده، {len(errors)} خطا) "
                  f"— {elapsed:.0f} ثانیه")

    # ── مرحله‌ی آخر: ساخت رابطه‌های REFERENCES بین مواد همین قانون ──
    ok, err = _run_with_retry(_create_references_tx, loader, law.name)
    if not ok:
        errors.append({
            "article_number": None,
            "law": law.name,
            "error_message": f"ساخت روابط REFERENCES شکست خورد: {err}",
        })
        print(f"   🔴 ساخت رابطه‌های REFERENCES برای {law.name} شکست خورد: {err}")

    elapsed = time.time() - start_time

    if errors:
        err_path = _errors_path(key)
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"\n⚠️  {len(errors)} مورد با خطا مواجه شدند — جزئیات در: {err_path}")
    else:
        print(f"\n✅ هیچ خطایی رخ نداد.")

    print(f"✅ {law.name}: {loaded} تازه لود شد، {already_done} از قبل موجود بود، "
          f"{len(errors)} خطا، از مجموع {total} ماده.")
    print(f"   زمان کل: {elapsed:.0f} ثانیه")


def _create_law_tx(loader: LawGraphLoader, law: Law):
    with loader.connection.session() as session:
        session.execute_write(loader._create_law, law)


def _create_references_tx(loader: LawGraphLoader, law_name: str):
    with loader.connection.session() as session:
        session.execute_write(loader._create_references, law_name)


def load_all_safe():
    print("💾 اتصال به Neo4j...")
    connection = _connect_with_retry()
    loader = LawGraphLoader(connection)

    try:
        loader.create_indexes()
        for key in LAW_CONFIGS:
            load_one_safe(key, loader)
    finally:
        connection.close()

    print("\n✅ بارگذاری Rule Graph (نسخه‌ی resumable) تمام شد!")


def load_single_safe(key: str):
    if key not in LAW_CONFIGS:
        print(f"❌ کلید ناشناخته: {key}")
        print(f"   کلیدهای معتبر: {', '.join(LAW_CONFIGS.keys())}")
        return

    print("💾 اتصال به Neo4j...")
    connection = _connect_with_retry()
    loader = LawGraphLoader(connection)

    try:
        loader.create_indexes()
        load_one_safe(key, loader)
    finally:
        connection.close()

    print("\n✅ بارگذاری (نسخه‌ی resumable) تمام شد!")


def _usage():
    keys = "|".join(LAW_CONFIGS.keys())
    print("استفاده:")
    print(f"  uv run main_laws_load_patch.py load <{keys}|all>")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "load":
        _usage()
        sys.exit(1)

    target = sys.argv[2]

    while True:
        try:
            load_all_safe() if target == "all" else load_single_safe(target)
            break
        except TRANSIENT_EXCEPTIONS as e:
            print(f"\n⚠️ قطعی شبکه در سطح کلی برنامه: {e}")
            print("   ۱۵ ثانیه صبر می‌کنیم و کل عملیات را دوباره شروع می‌کنیم "
                  "(مواد قبلاً لودشده رد خواهند شد)...")
            time.sleep(15)