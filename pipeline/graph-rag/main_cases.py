"""
main_cases.py — اجرای دو-مرحله‌ای پایپ‌لاین پرونده‌ها (Fact Graph)

⚠️ این فایل مستقل از main.py (که فقط مخصوص ۵ قانون مادر است) است — دقیقاً
مثل main.py دو مرحله‌ی جدا دارد (scrape / load) و از همون common/fetcher.py
و database/connection.py مشترک استفاده می‌کند.

پویا بودن نسبت به قوانین:
    این فایل به‌جای یک لیست دستی از ۵ قانون، مستقیم از روی
    laws.configs.LAW_CONFIGS حلقه می‌زند. یعنی اگر بعداً قانون ششمی به
    laws/configs.py اضافه شود، این پایپ‌لاین خودکار پرونده‌های همان
    قانون را هم اسکرپ می‌کند — به شرطی که شناسه‌ی سامانه‌ی آراء آن قانون
    هم در cases/configs.py ثبت شده باشد (وگرنه با خطای واضح متوقف می‌شود،
    نگاه کن به cases/configs.py).

درباره‌ی case_type:
    هر Ruling مستقل از قانونی که با آن پیدا شده، فیلد case_type
    (حقوقی/کیفری) خودش را از «گروه رأی» می‌گیرد — چون ممکن است یک رأی
    کیفری به یک ماده‌ی حقوقی هم استناد کند (مثلاً یک رأی کیفری چک به
    قانون تجارت). در گزارش پایانی این توزیع را جدا نشان می‌دهیم.

نحوه‌ی استفاده:
    uv run main_cases.py scrape civil
    uv run main_cases.py scrape all

    uv run main_cases.py load civil
    uv run main_cases.py load all
"""

import os
import sys
import json
import time
import dataclasses
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from laws.configs import LAW_CONFIGS
from cases.configs import get_case_search_id
from cases.parser import parse_ruling
from database.connection import Neo4jConnection
from database.case_loader import CaseGraphLoader, dict_to_ruling

load_dotenv()

NEO4J_URI  = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")

DATA_DIR = "data/cases"
REQUEST_DELAY = 1.5   # ثانیه — احترام به نرخ درخواست سایت

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fa-IR,fa;q=0.9,en;q=0.8",
}

LARGE_PAGE_SIZE = 2500   # بزرگ‌تر از بزرگ‌ترین قانون‌مون (مدنی: ۱۹۳۴)


def _law_dir(law_key: str) -> str:
    path = f"{DATA_DIR}/{law_key}"
    os.makedirs(path, exist_ok=True)
    return path


def _ruling_path(law_key: str, ruling_id: str) -> str:
    return f"{_law_dir(law_key)}/{ruling_id}.json"


# ─────────────────────────────────────────────────────────────────────────
# گام ۱: گرفتن لیست شناسه‌ی پرونده‌های هر قانون
# ─────────────────────────────────────────────────────────────────────────

def _extract_ruling_ids(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    links = soup.find_all("a", href=lambda h: h and "/Judge/Text/" in h)
    ids = [a.get("href").rstrip("/").split("/")[-1] for a in links]
    # ترتیب رو حفظ کن ولی تکراری‌ها رو حذف کن
    seen = set()
    unique_ids = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            unique_ids.append(i)
    return unique_ids


def fetch_all_ruling_ids(law_search_id: int) -> list[str]:
    """
    ابتدا با PageSize بزرگ سعی می‌کنه همه رو یک‌جا بگیره (سریع‌تر).
    اگه به هر دلیلی جواب نداد (نتایج کمتر از حد انتظار بود)، خودکار
    برمی‌گرده به حالت صفحه‌به‌صفحه با PageSize پیش‌فرض.
    """
    base_url = f"https://ara.jri.ac.ir/Judge/Index?Laws={law_search_id}"

    # تلاش اول: یک درخواست با PageSize بزرگ
    r = requests.get(f"{base_url}&PageSize={LARGE_PAGE_SIZE}", headers=HEADERS, timeout=20)
    r.encoding = "utf-8"
    ids = _extract_ruling_ids(r.text)
    time.sleep(REQUEST_DELAY)

    if ids:
        return ids

    # روش پشتیبان: صفحه‌به‌صفحه با اندازه‌ی پیش‌فرض (۲۵)
    print("   ⚠️ PageSize بزرگ جواب نداد — برگشت به حالت صفحه‌به‌صفحه...")
    all_ids: list[str] = []
    seen = set()
    page = 1
    while True:
        r = requests.get(f"{base_url}&PageNumber={page}", headers=HEADERS, timeout=20)
        r.encoding = "utf-8"
        page_ids = _extract_ruling_ids(r.text)
        new_ids = [i for i in page_ids if i not in seen]

        if not new_ids:
            break

        for i in new_ids:
            seen.add(i)
            all_ids.append(i)

        page += 1
        time.sleep(REQUEST_DELAY)

        if page > 500:   # محافظ در برابر حلقه‌ی بی‌نهایت
            print("   ⚠️ به سقف ۵۰۰ صفحه رسیدیم — متوقف شد (چیزی غیرعادیه).")
            break

    return all_ids


# ─────────────────────────────────────────────────────────────────────────
# گام ۲: اسکرپ — دریافت + پارس + ذخیره‌ی هر پرونده (resumable)
# ─────────────────────────────────────────────────────────────────────────

def scrape_one_law(law_key: str):
    if law_key not in LAW_CONFIGS:
        print(f"❌ کلید قانون ناشناخته: {law_key}")
        return

    law_search_id = get_case_search_id(law_key)   # اگه ثبت نشده باشه، اینجا خطای واضح می‌ده
    law_name = LAW_CONFIGS[law_key].law_name

    print(f"\n🔎 {law_name} (شناسه سامانه آراء: {law_search_id})")
    ruling_ids = fetch_all_ruling_ids(law_search_id)
    print(f"   {len(ruling_ids)} پرونده پیدا شد.")

    fetched, skipped, failed = 0, 0, 0

    for i, ruling_id in enumerate(ruling_ids, 1):
        path = _ruling_path(law_key, ruling_id)

        if os.path.exists(path) and os.path.getsize(path) > 0:
            skipped += 1
            continue

        url = f"https://ara.jri.ac.ir/Judge/Text/{ruling_id}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.encoding = "utf-8"
            soup = BeautifulSoup(r.text, "lxml")
            ruling = parse_ruling(soup, ruling_id, url)

            with open(path, "w", encoding="utf-8") as f:
                json.dump(dataclasses.asdict(ruling), f, ensure_ascii=False, indent=2)

            fetched += 1
            if fetched % 20 == 0:
                print(f"   ... {i}/{len(ruling_ids)} پردازش شد ({fetched} جدید، {skipped} از قبل)")

        except Exception as e:
            print(f"   ❌ خطا در {ruling_id}: {e}")
            failed += 1

        time.sleep(REQUEST_DELAY)

    print(f"✅ {law_name}: {fetched} جدید، {skipped} قبلاً موجود، {failed} خطا")


def scrape_all():
    for law_key in LAW_CONFIGS:
        scrape_one_law(law_key)


# ─────────────────────────────────────────────────────────────────────────
# گام ۳: Load  (فقط با IP خارج / Codespace اجرا شود)
# ─────────────────────────────────────────────────────────────────────────

def load_one_law(law_key: str, loader: CaseGraphLoader):
    law_dir = _law_dir(law_key)
    files = [f for f in os.listdir(law_dir) if f.endswith(".json")]

    if not files:
        print(f"❌ هیچ فایلی برای {law_key} پیدا نشد — اول scrape کن.")
        return

    case_type_counts: dict[str, int] = {}

    for filename in files:
        with open(os.path.join(law_dir, filename), "r", encoding="utf-8") as f:
            data = json.load(f)

        ruling = dict_to_ruling(data)
        loader.load_ruling(ruling)

        case_type_counts[ruling.case_type] = case_type_counts.get(ruling.case_type, 0) + 1

    print(f"✅ {law_key}: {len(files)} پرونده بارگذاری شد.  توزیع: {case_type_counts}")


def load_all():
    print("💾 اتصال به Neo4j...")
    connection = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASS)
    loader = CaseGraphLoader(connection)

    try:
        loader.create_indexes()
        for law_key in LAW_CONFIGS:
            load_one_law(law_key, loader)
    finally:
        connection.close()

    print("✅ بارگذاری Fact Graph تمام شد!")


def load_single(law_key: str):
    print("💾 اتصال به Neo4j...")
    connection = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASS)
    loader = CaseGraphLoader(connection)

    try:
        loader.create_indexes()
        load_one_law(law_key, loader)
    finally:
        connection.close()

    print("✅ بارگذاری تمام شد!")


# ─────────────────────────────────────────────────────────────────────────

def _usage():
    keys = "|".join(LAW_CONFIGS.keys())
    print("استفاده:")
    print(f"  uv run main_cases.py scrape <{keys}|all>")
    print(f"  uv run main_cases.py load   <{keys}|all>")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        _usage()
        sys.exit(1)

    mode, target = sys.argv[1], sys.argv[2]

    if mode == "scrape":
        scrape_all() if target == "all" else scrape_one_law(target)

    elif mode == "load":
        load_all() if target == "all" else load_single(target)

    else:
        _usage()