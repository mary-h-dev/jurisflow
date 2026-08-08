 
import os
import sys
import json
import dataclasses
from dotenv import load_dotenv
 
from common.fetcher import load_local_html
from laws.parser import parse_law
from laws.configs import LAW_CONFIGS
from database.connection import Neo4jConnection
from database.law_loader import LawGraphLoader, dict_to_law
 

 
load_dotenv()
 
NEO4J_URI  = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")
 
DATA_DIR = "data/laws"   
 
LAW_URLS = {
    "civil":               "https://qavanin.ir/Law/TreeText/?IDS=12021850837713548188",
    "penal":                "https://qavanin.ir/Law/TreeText/?IDS=4693937366938194803",             
    "commercial":           "https://qavanin.ir/Law/TreeText/?IDS=12145533825531226090",      
    "criminal_procedure":   "https://qavanin.ir/Law/TreeText/?IDS=2433638803151753757",     
    "civil_procedure":      "https://qavanin.ir/Law/TreeText/?IDS=502404499976366661",        
}


 
def _paths(key: str) -> tuple[str, str]:
    raw_html = f"{DATA_DIR}/{key}_law.html"
    raw_json = f"{DATA_DIR}/{key}_law.json"
    return raw_html, raw_json
 
 
# ─────────────────────────────────────────────────────────────────────────
# مرحله ۱ — Scrape  (فقط با IP ایران؛ HTML باید از قبل دستی ذخیره شده باشد)
# ─────────────────────────────────────────────────────────────────────────
 
def scrape_one(key: str):
    if key not in LAW_CONFIGS:
        print(f"❌ کلید ناشناخته: {key}")
        return
 
    config = LAW_CONFIGS[key]
    raw_html, raw_json = _paths(key)
 
    os.makedirs(DATA_DIR, exist_ok=True)
 
    if not os.path.exists(raw_html) or os.path.getsize(raw_html) == 0:
        print(f"⚠️ فایل HTML یافت نشد یا خالی است: {raw_html}")
        print("   qavanin.ir پشت چالش ضد-ربات JS است و به‌صورت خودکار قابل دریافت نیست.")
        print("   لطفاً این مراحل را انجام بده:")
        print(f"   ۱) در مرورگر باز کن: {LAW_URLS.get(key, '(آدرس تعریف نشده)')}")
        print("   ۲) صبر کن صفحه کامل لود شود.")
        print(f"   ۳) Ctrl+S → «Webpage, Complete» → دقیقاً در این مسیر ذخیره کن: {raw_html}")
        print(f"   ۴) دوباره اجرا کن: python main.py scrape {key}")
        return
 
    soup = load_local_html(raw_html)
    if not soup:
        print(f"❌ خطا در خواندن HTML برای {config.law_name}")
        return
 
    print(f"🔍 در حال parse کردن: {config.law_name} ...")
    law = parse_law(soup, config, LAW_URLS.get(key, ""))
 
    if len(law.articles) == 0:
        print(f"⚠️ هیچ ماده‌ای پیدا نشد. احتمالاً article_class در laws/configs.py")
        print(f"   با HTML واقعی این قانون مطابقت ندارد — با debug_check.py بررسی کن:")
        print(f"   python debug_check.py {raw_html}")
        return
 
    with open(raw_json, "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(law), f, ensure_ascii=False, indent=2)
 
    print(f"✅ {len(law.articles)} ماده ذخیره شد → {raw_json}")
 
 
def scrape_all():
    for key in LAW_CONFIGS:
        scrape_one(key)
 
 
# ─────────────────────────────────────────────────────────────────────────
# مرحله ۲ — Load  (فقط با IP خارج / Codespace اجرا شود)
# ─────────────────────────────────────────────────────────────────────────
 
def load_one(key: str, loader: LawGraphLoader):
    _, raw_json = _paths(key)
 
    if not os.path.exists(raw_json):
        print(f"❌ فایل JSON یافت نشد: {raw_json} — اول scrape کن.")
        return
 
    with open(raw_json, "r", encoding="utf-8") as f:
        data = json.load(f)
 
    law = dict_to_law(data)
    loader.load_law(law)
 
 
def load_all():
    print("💾 اتصال به Neo4j...")
    connection = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASS)
    loader = LawGraphLoader(connection)
 
    try:
        loader.create_indexes()
        for key in LAW_CONFIGS:
            load_one(key, loader)
    finally:
        connection.close()
 
    print("✅ بارگذاری در Neo4j تمام شد!")
 
 
def load_single(key: str):
    print("💾 اتصال به Neo4j...")
    connection = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASS)
    loader = LawGraphLoader(connection)
 
    try:
        loader.create_indexes()
        load_one(key, loader)
    finally:
        connection.close()
 
    print("✅ بارگذاری تمام شد!")
 
 
# ─────────────────────────────────────────────────────────────────────────
 
def _usage():
    print("استفاده:")
    print("  uv run main.py scrape <civil|penal|commercial|criminal_procedure|civil_procedure|all>")
    print("  uv run main.py load   <civil|penal|commercial|criminal_procedure|civil_procedure|all>")
 
 
if __name__ == "__main__":
    if len(sys.argv) < 3:
        _usage()
        sys.exit(1)
 
    mode, target = sys.argv[1], sys.argv[2]
 
    if mode == "scrape":
        scrape_all() if target == "all" else scrape_one(target)
 
    elif mode == "load":
        load_all() if target == "all" else load_single(target)
 
    else:
        _usage()
 