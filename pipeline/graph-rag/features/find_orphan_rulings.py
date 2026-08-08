"""
features/find_orphan_rulings.py — پیدا کردن دقیقِ ruling_id هایی که در
data/features/<domain>/*.json هستن ولی node Ruling متناظرشون در Neo4j
وجود نداره.

چرا این اسکریپت لازم بود؟
    دو تشخیص متفاوت مطرح شد (type mismatch از یک AI، بازمانده‌ی migration
    ناقص از AI دیگر) — به‌جای انتخاب بین حدس‌ها، این اسکریپت مستقیماً
    برای *هر* ruling_id چک می‌کند که آیا node متناظرش (با هر دو نوع
    رشته/عدد) وجود دارد یا نه، و لیست دقیق یتیم‌ها را گزارش می‌دهد.

نحوه‌ی اجرا:
    uv run -m features.find_orphan_rulings civil
"""

import glob
import json
import os
import sys

from dotenv import load_dotenv

from database.connection import Neo4jConnection

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")

FEATURES_DIR = "data/features"


def get_all_extracted_ruling_ids(domain: str) -> list[str]:
    paths = glob.glob(f"{FEATURES_DIR}/{domain}/*.json")
    ids = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        ids.append(str(data["ruling_id"]))
    return sorted(ids)


def find_orphans(ruling_ids: list[str], connection: Neo4jConnection) -> dict:
    """
    برای هر ruling_id چک می‌کند: آیا با نوع رشته وجود دارد؟ با نوع عدد؟
    خروجی: {"missing_both": [...], "found_as_string": n, "found_as_int": n}
    """
    missing_both = []
    found_as_string = 0
    found_as_int = 0

    with connection.session() as session:
        for rid in ruling_ids:
            result_str = session.run(
                "MATCH (r:Ruling {ruling_id: $rid}) RETURN r.ruling_id AS id LIMIT 1",
                rid=rid,
            ).single()
            if result_str:
                found_as_string += 1
                continue

            try:
                rid_int = int(rid)
            except ValueError:
                rid_int = None

            if rid_int is not None:
                result_int = session.run(
                    "MATCH (r:Ruling {ruling_id: $rid}) RETURN r.ruling_id AS id LIMIT 1",
                    rid=rid_int,
                ).single()
                if result_int:
                    found_as_int += 1
                    continue

            missing_both.append(rid)

    return {
        "missing_both": missing_both,
        "found_as_string": found_as_string,
        "found_as_int": found_as_int,
    }


def run(domain: str):
    ruling_ids = get_all_extracted_ruling_ids(domain)
    print(f"📂 {len(ruling_ids)} ruling_id از فایل‌های استخراج‌شده‌ی «{domain}» خوانده شد")

    connection = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASS)
    try:
        stats = find_orphans(ruling_ids, connection)
    finally:
        connection.close()

    print(f"\n📊 نتیجه:")
    print(f"  ✅ پیدا شد با نوع رشته : {stats['found_as_string']}")
    print(f"  ✅ پیدا شد با نوع عدد  : {stats['found_as_int']}  "
          f"(اگه این عدد > 0 بود، یعنی type mismatch واقعاً بخشی از مشکل بوده)")
    print(f"  🔴 اصلاً پیدا نشد      : {len(stats['missing_both'])}  "
          f"(یعنی Ruling node این‌ها اصلاً در گراف Case ساخته نشده)")

    if stats["missing_both"]:
        out_path = f"data/features/{domain}_orphan_ruling_ids.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(stats["missing_both"]))
        print(f"\n💾 لیست کامل ruling_id های یتیم ذخیره شد: {out_path}")
        print(f"   نمونه (۱۰ تای اول): {stats['missing_both'][:10]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("استفاده: uv run -m features.find_orphan_rulings <domain>")
        sys.exit(1)
    run(sys.argv[1])