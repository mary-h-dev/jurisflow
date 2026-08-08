"""
main_cases_load_patch.py — نسخه‌ی resumable و مقاوم در برابر قطعی شبکه

تفاوت با نسخه‌ی قبلی:
  ۱. Resumable: قبل از لود هر پرونده، چک می‌کند آیا آن Ruling از قبل در
     Neo4j هست یا نه (و «کامل» است — یعنی HAS_SECTION هم دارد). اگر بود،
     رد می‌شود. یعنی اگر اینترنت وسط کار قطع شد، دفعه‌ی بعد از همان‌جا
     ادامه می‌دهد، نه از اول.

  ۲. مقاوم در برابر قطعی لحظه‌ای: اگر موقع لود یک پرونده خطای شبکه/اتصال
     رخ بدهد (قطعی موقت، timeout، connection reset)، به‌جای رد شدن فوری
     یا متوقف شدن کل اسکریپت، تا ۵ بار با فاصله‌ی افزایشی (۵، ۱۰، ۲۰،
     ۴۰، ۸۰ ثانیه) دوباره تلاش می‌کند. اگر بعد از ۵ بار هم نشد، همان
     پرونده را در گزارش خطا ثبت می‌کند و می‌رود سراغ بعدی (کل اسکریپت
     متوقف نمی‌شود).

  ۳. حتی اگر قطعی در سطح کل برنامه رخ بدهد (نه فقط یک پرونده)، برنامه
     خودش صبر می‌کند و کل عملیات را از نو صدا می‌زند — و چون رزومه‌پذیر
     است، پرونده‌های قبلاً لودشده را رد می‌کند و عملاً از همان‌جا که
     قطع شده بود ادامه می‌دهد.

نحوه‌ی استفاده: دقیقاً مثل قبل —
    uv run main_cases_load_patch.py load civil
    uv run main_cases_load_patch.py load all

اگر وسط کار اینترنت قطع شد، لازم نیست کاری کنی — اسکریپت خودش صبر
می‌کند و ادامه می‌دهد. اگر کامل بسته شد (مثلاً Codespace ری‌استارت شد)،
فقط دوباره همین دستور را بزن.
"""

import os
import sys
import json
import time
import traceback

from dotenv import load_dotenv
from neo4j.exceptions import Neo4jError, ServiceUnavailable, SessionExpired

from laws.configs import LAW_CONFIGS
from database.connection import Neo4jConnection
from database.case_loader import CaseGraphLoader, dict_to_ruling

load_dotenv()

NEO4J_URI  = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")

DATA_DIR = "data/cases"
ERRORS_DIR = "data/features"

# تنظیمات retry برای قطعی موقت شبکه
MAX_RETRIES = 5
RETRY_BASE_DELAY = 5   # ثانیه — هر بار دو برابر می‌شود (5, 10, 20, 40, 80)

# خطاهایی که «موقتی» و مربوط به شبکه/اتصال هستند و باید retry شوند
TRANSIENT_EXCEPTIONS = (
    ServiceUnavailable,
    SessionExpired,
    ConnectionError,
    TimeoutError,
    OSError,   # شامل قطعی DNS، reset شدن سوکت و مشابه
)


def _law_dir(law_key: str) -> str:
    return f"{DATA_DIR}/{law_key}"


def _errors_path(law_key: str) -> str:
    os.makedirs(ERRORS_DIR, exist_ok=True)
    return f"{ERRORS_DIR}/{law_key}_load_errors.json"


# ─────────────────────────────────────────────────────────────────────────
# چک کردن این‌که آیا یک Ruling از قبل «کامل» در Neo4j هست یا نه
# ─────────────────────────────────────────────────────────────────────────

def _already_loaded(loader: CaseGraphLoader, ruling_id: str) -> bool:
    """
    True برمی‌گرداند اگر این ruling_id از قبل در گراف هست *و* حداقل یک
    RulingSection هم دارد (یعنی load قبلی‌اش تا انتها کامل شده بوده،
    نه این‌که وسط کار قطع شده و فقط گره‌ی خالی Ruling ساخته شده باشد).
    """
    query = """
    MATCH (r:Ruling {ruling_id: $ruling_id})
    OPTIONAL MATCH (r)-[:HAS_SECTION]->(s:RulingSection)
    RETURN r IS NOT NULL AS has_ruling, count(s) AS section_count
    """
    with loader.connection.session() as session:
        result = session.run(query, ruling_id=ruling_id)
        record = result.single()
        if record is None:
            return False
        return bool(record["has_ruling"]) and record["section_count"] > 0


def _connect_with_retry() -> Neo4jConnection:
    """اتصال اولیه هم اگر با قطعی مواجه شد، چند بار تلاش می‌کند."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASS)
        except TRANSIENT_EXCEPTIONS as e:
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"   ⚠️ اتصال ناموفق (تلاش {attempt}/{MAX_RETRIES}): {e}")
            print(f"      {delay} ثانیه صبر می‌کنیم و دوباره تلاش می‌کنیم...")
            time.sleep(delay)
    # آخرین تلاش بدون گرفتن exception، اگر بازم fail شد بالا می‌رود
    return Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASS)


def _load_with_retry(loader: CaseGraphLoader, ruling) -> tuple[bool, str | None]:
    """
    تلاش برای لود یک ruling با retry در برابر خطاهای شبکه‌ای موقت.
    خروجی: (موفق بود یا نه, پیام خطا در صورت شکست نهایی)
    """
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            loader.load_ruling(ruling)
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
            # خطای منطقی/داده‌ای (نه شبکه‌ای) — retry فایده ندارد، فوراً رد شو
            return False, f"{type(e).__name__}: {e}"

    return False, last_error


# ─────────────────────────────────────────────────────────────────────────
# نسخه‌ی resumable + مقاوم load_one_law
# ─────────────────────────────────────────────────────────────────────────

def load_one_law_safe(law_key: str, loader: CaseGraphLoader):
    law_dir = _law_dir(law_key)

    if not os.path.isdir(law_dir):
        print(f"❌ پوشه‌ی {law_dir} پیدا نشد — اول scrape کن.")
        return

    files = [f for f in os.listdir(law_dir) if f.endswith(".json")]

    if not files:
        print(f"❌ هیچ فایلی برای {law_key} پیدا نشد — اول scrape کن.")
        return

    total = len(files)
    print(f"\n📂 {law_key}: {total} فایل پیدا شد. در حال چک کردن پرونده‌های قبلاً لودشده...")

    case_type_counts: dict[str, int] = {}
    succeeded = 0
    already_done = 0
    errors: list[dict] = []

    start_time = time.time()

    for i, filename in enumerate(files, 1):
        ruling_id = filename.replace(".json", "")
        path = os.path.join(law_dir, filename)

        # ── چک resumability: اگه قبلاً کامل لود شده، رد شو ──
        try:
            if _already_loaded(loader, ruling_id):
                already_done += 1
                if i % 50 == 0:
                    elapsed = time.time() - start_time
                    print(f"   ... {i}/{total} بررسی شد "
                          f"({succeeded} تازه لود شد، {already_done} قبلاً بوده، {len(errors)} خطا) "
                          f"— {elapsed:.0f} ثانیه")
                continue
        except TRANSIENT_EXCEPTIONS:
            # حتی چک کردن هم اگه با قطعی مواجه شد، صبر کن و دوباره سعی کن
            print(f"   ⚠️ قطعی موقت هنگام چک کردن {ruling_id} — ۱۰ ثانیه صبر...")
            time.sleep(10)

        # ── خواندن فایل ──
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()

            if not raw.strip():
                raise ValueError("فایل خالی است (۰ بایت یا فقط whitespace)")

            data = json.loads(raw)
            ruling = dict_to_ruling(data)

        except json.JSONDecodeError as e:
            errors.append({
                "ruling_id": ruling_id, "file": path,
                "error_type": "JSONDecodeError", "error_message": str(e),
            })
            print(f"   🔴 [{i}/{total}] JSON خراب: {ruling_id} — {e}")
            continue

        except Exception as e:
            errors.append({
                "ruling_id": ruling_id, "file": path,
                "error_type": type(e).__name__, "error_message": str(e),
                "traceback": traceback.format_exc(),
            })
            print(f"   🔴 [{i}/{total}] خطا در خواندن فایل {ruling_id}: {type(e).__name__}: {e}")
            continue

        # ── لود با retry در برابر قطعی موقت ──
        ok, err_msg = _load_with_retry(loader, ruling)

        if ok:
            case_type_counts[ruling.case_type] = case_type_counts.get(ruling.case_type, 0) + 1
            succeeded += 1
        else:
            errors.append({
                "ruling_id": ruling_id, "file": path,
                "error_type": "LoadFailedAfterRetries", "error_message": err_msg,
            })
            print(f"   🔴 [{i}/{total}] لود {ruling_id} بعد از {MAX_RETRIES} تلاش شکست خورد: {err_msg}")

        if i % 50 == 0:
            elapsed = time.time() - start_time
            print(f"   ... {i}/{total} پردازش شد "
                  f"({succeeded} تازه لود شد، {already_done} قبلاً بوده، {len(errors)} خطا) "
                  f"— {elapsed:.0f} ثانیه")

    elapsed = time.time() - start_time

    if errors:
        err_path = _errors_path(law_key)
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"\n⚠️  {len(errors)} پرونده با خطا مواجه شدند — جزئیات در: {err_path}")
    else:
        print(f"\n✅ هیچ خطایی رخ نداد.")

    print(f"✅ {law_key}: {succeeded} تازه لود شد، {already_done} از قبل موجود بود، "
          f"{len(errors)} خطا، از مجموع {total} فایل.")
    print(f"   توزیع نوع پرونده (تازه‌لودشده‌ها): {case_type_counts}")
    print(f"   زمان کل: {elapsed:.0f} ثانیه")


def load_all_safe():
    print("💾 اتصال به Neo4j...")
    connection = _connect_with_retry()
    loader = CaseGraphLoader(connection)

    try:
        loader.create_indexes()
        for law_key in LAW_CONFIGS:
            load_one_law_safe(law_key, loader)
    finally:
        connection.close()

    print("\n✅ بارگذاری Fact Graph (نسخه‌ی resumable) تمام شد!")


def load_single_safe(law_key: str):
    if law_key not in LAW_CONFIGS:
        print(f"❌ کلید قانون ناشناخته: {law_key}")
        print(f"   کلیدهای معتبر: {', '.join(LAW_CONFIGS.keys())}")
        return

    print("💾 اتصال به Neo4j...")
    connection = _connect_with_retry()
    loader = CaseGraphLoader(connection)

    try:
        loader.create_indexes()
        load_one_law_safe(law_key, loader)
    finally:
        connection.close()

    print("\n✅ بارگذاری (نسخه‌ی resumable) تمام شد!")


def _usage():
    keys = "|".join(LAW_CONFIGS.keys())
    print("استفاده:")
    print(f"  uv run main_cases_load_patch.py load <{keys}|all>")


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
                  "(پرونده‌های قبلاً لودشده رد خواهند شد)...")
            time.sleep(15)