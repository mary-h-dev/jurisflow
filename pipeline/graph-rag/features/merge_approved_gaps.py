"""
features/merge_approved_gaps.py — merges approved terms into the
categorized vocabulary

Input: a CSV with the main gap_report columns plus a "دسته_بازبینی"
(review category) column, whose value is one of:
    - one of the VOCAB_CATEGORIES keys ("concept"/"action"/"role"/"object"/"principle")
      → add as a new term to that category
    - "alias:<canonical>" → add as an alias of an existing canonical term
    - "REMOVE" or empty → ignored

No file is overwritten directly unless dry_run=False is passed explicitly.
"""

import csv
import json
from pathlib import Path

from features.configs import VOCAB_CATEGORIES

_HERE = Path(__file__).resolve().parent.parent
VOCAB_DIR = _HERE / "data" / "legal-vocabulary" / "categorized"
REVIEWED_CSV = _HERE / "data" / "legal-vocabulary" / "gap_audit" / "gap_report_reviewed.csv"  # set your file path here

VALID_CATEGORIES = set(VOCAB_CATEGORIES.keys())  # {"concept","action","role","object","principle"}


def run(csv_path: Path = REVIEWED_CSV, dry_run: bool = True):
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if "دسته_بازبینی" not in rows[0]:
        raise ValueError("Column «دسته_بازبینی» not found in the file — check the column name.")

    to_add_canonical: dict[str, list[str]] = {k: [] for k in VOCAB_CATEGORIES}
    to_add_alias: dict[str, list[tuple[str, str]]] = {k: [] for k in VOCAB_CATEGORIES}
    skipped, invalid = [], []

    # Load the current vocabulary once, so for aliases we can find the
    # canonical term's category
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
                invalid.append((term, decision, "canonical not found in any category"))
        else:
            invalid.append((term, decision, "invalid value"))

    # --- report before writing ---
    print(f"📄 Read {len(rows)} rows — {len(skipped)} REMOVE/empty skipped\n")
    for key in VOCAB_CATEGORIES:
        if to_add_canonical[key]:
            print(f"➕ {key} ({len(to_add_canonical[key])} new terms): {to_add_canonical[key]}")
        if to_add_alias[key]:
            print(f"🔗 {key} ({len(to_add_alias[key])} new aliases): {to_add_alias[key]}")
    if invalid:
        print(f"\n⚠️ {len(invalid)} invalid rows (need manual review):")
        for term, decision, reason in invalid:
            print(f"  - «{term}» → «{decision}» ({reason})")

    if dry_run:
        print("\n🔍 dry_run=True — no file was changed. If the report above looks right: run(dry_run=False)")
        return

    # --- actual write ---
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

    print("\n✅ categorized/*.json files updated.")


if __name__ == "__main__":
    run(dry_run=True)  # always dry_run first — after reviewing the output, run(dry_run=False)