"""
features/merge_approved_gaps.py — ادغام واژه‌های تأییدشده به واژه‌نامه‌ی categorized

ورودی: CSV با ستون‌های اصلی gap_report + یک ستون "دسته_بازبینی" که مقدارش
یکی از این‌هاست:
    - یکی از کلیدهای VOCAB_CATEGORIES ("concept"/"action"/"role"/"object"/"principle")
      → یعنی به‌عنوان واژه‌ی جدید به همون دسته اضافه بشه
    - "alias:<canonical>" → یعنی alias برای یک واژه‌ی canonical موجود
    - "REMOVE" یا خالی → نادیده گرفته می‌شه

هیچ فایلی مستقیم overwrite نمی‌شه مگر با dry_run=False صریح.
"""

import csv
import json
from pathlib import Path

from features.configs import VOCAB_CATEGORIES

_HERE = Path(__file__).resolve().parent.parent
VOCAB_DIR = _HERE / "data" / "legal-vocabulary" / "categorized"
REVIEWED_CSV = _HERE / "data" / "legal-vocabulary" / "gap_audit" / "gap_report_reviewed.csv"  # مسیر فایل خودتون رو اینجا تنظیم کنید

VALID_CATEGORIES = set(VOCAB_CATEGORIES.keys())  # {"concept","action","role","object","principle"}


def run(csv_path: Path = REVIEWED_CSV, dry_run: bool = True):
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if "دسته_بازبینی" not in rows[0]:
        raise ValueError("ستون «دسته_بازبینی» در فایل پیدا نشد — نام ستون رو چک کن.")

    to_add_canonical: dict[str, list[str]] = {k: [] for k in VOCAB_CATEGORIES}
    to_add_alias: dict[str, list[tuple[str, str]]] = {k: [] for k in VOCAB_CATEGORIES}
    skipped, invalid = [], []

    # واژه‌نامه‌ی فعلی رو یک‌بار لود کن تا برای alias بتونیم دسته‌ی canonical رو پیدا کنیم
    existing_canon = {
        key: set(json.load(open(VOCAB_DIR / f"{key}s.json", encoding="utf-8")))
        if (VOCAB_DIR / f"{key}s.json").exists() else set()
        for key in VOCAB_CATEGORIES
    }

    for r in rows:
        term = r["عبارت"].strip()
        decision = r["دسته_بازبینی"].strip()

        if not decision or decision.upper() == "REMOVE":
            skipped.append(term)
            continue

        if decision in VALID_CATEGORIES:
            to_add_canonical[decision].append(term)
        elif decision.startswith("alias:"):
            canonical = decision.split(":", 1)[1].strip()
            found_key = next((k for k, words in existing_canon.items() if canonical in words), None)
            if found_key:
                to_add_alias[found_key].append((canonical, term))
            else:
                invalid.append((term, decision, "canonical پیدا نشد در هیچ دسته‌ای"))
        else:
            invalid.append((term, decision, "مقدار نامعتبر"))

    # --- گزارش قبل از نوشتن ---
    print(f"📄 {len(rows)} ردیف خونده شد — {len(skipped)} REMOVE/خالی رد شد\n")
    for key in VOCAB_CATEGORIES:
        if to_add_canonical[key]:
            print(f"➕ {key} ({len(to_add_canonical[key])} واژه‌ی جدید): {to_add_canonical[key]}")
        if to_add_alias[key]:
            print(f"🔗 {key} ({len(to_add_alias[key])} alias جدید): {to_add_alias[key]}")
    if invalid:
        print(f"\n⚠️ {len(invalid)} ردیفِ نامعتبر (بررسی دستی لازمه):")
        for term, decision, reason in invalid:
            print(f"  - «{term}» → «{decision}» ({reason})")

    if dry_run:
        print("\n🔍 dry_run=True — هیچ فایلی تغییر نکرد. اگه گزارش بالا درسته: run(dry_run=False)")
        return

    # --- نوشتن واقعی ---
    for key in VOCAB_CATEGORIES:
        canon_path = VOCAB_DIR / f"{key}s.json"
        alias_path = VOCAB_DIR / f"{key}s_aliases.json"

        canon_list = json.load(open(canon_path, encoding="utf-8")) if canon_path.exists() else []
        alias_map = json.load(open(alias_path, encoding="utf-8")) if alias_path.exists() else {}

        for term in to_add_canonical[key]:
            if term not in canon_list:
                canon_list.append(term)
                alias_map.setdefault(term, [term])

        for canonical, new_alias in to_add_alias[key]:
            alias_map.setdefault(canonical, [canonical])
            if new_alias not in alias_map[canonical]:
                alias_map[canonical].append(new_alias)

        json.dump(sorted(canon_list), open(canon_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        json.dump(alias_map, open(alias_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2, sort_keys=True)

    print("\n✅ فایل‌های categorized/*.json به‌روزرسانی شدن.")


if __name__ == "__main__":
    run(dry_run=True)  # اول همیشه dry_run — بعد از بررسی خروجی، run(dry_run=False) کن