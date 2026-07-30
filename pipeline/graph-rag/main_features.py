"""
main_features.py — اجرای دو-مرحله‌ای پایپ‌لاین Feature Extraction (گراف سوم)

⚠️ چرا یک فایل جدا از main_cases.py؟
    چون این مرحله LLM-heavy است (نه فقط اسکرپ/regex ارزون)، و طبق
    تصمیمی که گرفتیم باید با یک نمونه‌ی کوچک (پایلوت) شروع بشه، نه کل
    دیتاست. برای همین limit پیش‌فرض این اسکریپت عمداً ۵۰ گذاشته شده —
    تا کسی به‌اشتباه کل ~۲۴۰۰ پرونده رو یک‌جا اجرا نکنه و هزینه/زمان
    غیرمنتظره نده.

چرا extract و load دو مرحله‌ی جدا هستن (مثل main_cases.py)؟
    extract فقط LLM صدا می‌زنه و نتیجه رو به‌صورت JSON روی دیسک
    می‌ریزه (resumable: اگه یک پرونده قبلاً استخراج شده، رد می‌شه).
    load مستقل بعداً همون JSON ها رو می‌خونه و توی Neo4j می‌ریزه.
    این جدایی یعنی اگه یک روز schema گراف عوض شد، لازم نیست دوباره
    هزینه‌ی LLM بدیم — فقط load رو دوباره اجرا می‌کنیم.

نحوه‌ی استفاده:
    uv run main_features.py extract civil --limit 50
    uv run main_features.py load civil --limit 50
    PYTHONUNBUFFERED=1 uv run main_features.py extract civil --limit 0 2>&1 | tee -a pipeline.log
"""

import dataclasses
import glob
import json
import os
import sys

from dotenv import load_dotenv

from database.connection import Neo4jConnection
from database.feature_loader import FeatureGraphLoader
from features.extractor import extract_ruling
from features.schemas import Evidence, ExtractedFeature, FeatureExtractionResult
from features.vocab_resolver import VocabResolver



load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")

CASES_DIR = "data/cases"
FEATURES_DIR = "data/features"  # خروجی extract اینجا ذخیره می‌شود (ورودی load)

DEFAULT_PILOT_LIMIT = 50


def _ruling_full_text(ruling: dict) -> str:
    """متن کامل رأی را از سکشن‌های ذخیره‌شده بازمی‌سازد (نگاه کن به cases/schemas.py)"""
    return "\n\n".join(s["text"] for s in ruling.get("sections", []))


def extract_domain(domain: str, limit: int | None):
    out_dir = f"{FEATURES_DIR}/{domain}"
    os.makedirs(out_dir, exist_ok=True)

    paths = sorted(glob.glob(f"{CASES_DIR}/{domain}/*.json"))
    if limit:
        paths = paths[:limit]

    print(f"🔎 استخراج Feature برای {len(paths)} پرونده از «{domain}» "
          f"(limit={limit or 'بدون محدودیت'})...")

    resolver = VocabResolver()
    print("📖 واژه‌نامه و کش embedding برای resolver بارگذاری شد.")

    done, skipped, failed = 0, 0, 0
    for i, path in enumerate(paths, start=1):
        try:
            with open(path, encoding="utf-8") as f:
                ruling = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ⚠️ [{i}/{len(paths)}] فایل خراب/خالی رد شد ({path}): {e}")
            failed += 1
            continue

        out_path = f"{out_dir}/{ruling['ruling_id']}.json"
        if os.path.exists(out_path):
            skipped += 1
            continue  # resumable — قبلاً استخراج شده

        text = _ruling_full_text(ruling)
        if not text.strip():
            print(f"  ⚠️ [{i}/{len(paths)}] {ruling['ruling_id']}: متن خالی، رد شد")
            continue

        try:
            result = extract_ruling(ruling["ruling_id"], text, resolver)
        except RuntimeError as e:
            print(f"  ❌ [{i}/{len(paths)}] {ruling['ruling_id']}: {e}")
            failed += 1
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(result), f, ensure_ascii=False, indent=2)
        done += 1
        n_features = len(result.all_features())
        print(f"  ✅ [{i}/{len(paths)}] {ruling['ruling_id']}: {n_features} feature استخراج شد")

    print(f"\n📊 نتیجه: {done} استخراج‌شده، {skipped} از قبل موجود (رد شد)، {failed} شکست‌خورده")


def _dict_to_result(data: dict) -> FeatureExtractionResult:
    def to_feature(d: dict) -> ExtractedFeature:
        return ExtractedFeature(
            category=d["category"], value=d["value"], evidence=Evidence(**d["evidence"])
        )

    kwargs = {"ruling_id": data["ruling_id"]}
    for key in ["concepts", "actions", "roles", "objects", "facts"]:
        kwargs[key] = [to_feature(d) for d in data.get(key, [])]
    return FeatureExtractionResult(**kwargs)


def load_domain(domain: str, limit: int | None):
    paths = sorted(glob.glob(f"{FEATURES_DIR}/{domain}/*.json"))
    if limit:
        paths = paths[:limit]

    if not paths:
        print(f"⚠️ هیچ فایل استخراج‌شده‌ای برای «{domain}» پیدا نشد. "
              f"اول extract را اجرا کن.")
        return

    connection = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASS)
    loader = FeatureGraphLoader(connection)
    loader.create_indexes()

    loaded_count, skipped_count = 0, 0
    for i, path in enumerate(paths, start=1):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        result = _dict_to_result(data)

        if loader.is_ruling_loaded(result.ruling_id):
            skipped_count += 1
            continue  # قبلاً بارگذاری شده -- resumable، نگاه کن به توضیح feature_loader.py

        loader.load_result(result)
        loaded_count += 1
        print(f"  ✅ [{i}/{len(paths)}] {result.ruling_id} بارگذاری شد")

    connection.close()
    print(f"\n✅ {loaded_count} نتیجه‌ی جدید بارگذاری شد، {skipped_count} از قبل موجود بود (رد شد).")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("استفاده: uv run main_features.py extract|load <domain> [--limit N]")
        print(f"         (پیش‌فرض limit={DEFAULT_PILOT_LIMIT}؛ برای کل دیتاست: --limit 0)")
        sys.exit(1)

    action, domain = sys.argv[1], sys.argv[2]
    limit = DEFAULT_PILOT_LIMIT
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    limit = limit or None  # --limit 0 یعنی بدون محدودیت

    if action == "extract":
        extract_domain(domain, limit)
    elif action == "load":
        load_domain(domain, limit)
    else:
        print("❌ عمل نامعتبر؛ از extract یا load استفاده کن.")
        sys.exit(1)