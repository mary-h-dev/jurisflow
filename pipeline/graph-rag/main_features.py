"""
main_features.py — two-stage runner for the Feature Extraction pipeline
(third graph layer)

⚠️ Why a separate file from main_cases.py?
    Because this stage is LLM-heavy (not just cheap scraping/regex), and
    per the agreed plan it should start with a small pilot sample, not
    the whole dataset. That's why this script's default limit is
    deliberately set to 50 — so nobody accidentally runs all ~2400 cases
    at once and gets an unexpected cost/time bill.

Why are extract and load two separate stages (like main_cases.py)?
    extract only calls the LLM and writes the result to disk as JSON
    (resumable: if a case was already extracted, it's skipped). load
    independently reads those JSON files later and writes them into
    Neo4j. This separation means if the graph schema changes one day,
    there's no need to pay the LLM cost again — just re-run load.

Usage:
    uv run main_features.py extract civil --limit 50
    uv run main_features.py load civil --limit 50

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
FEATURES_DIR = "data/features"  # extract output goes here (load's input)

DEFAULT_PILOT_LIMIT = 50


def _ruling_full_text(ruling: dict) -> str:
    """Reconstructs the full ruling text from its stored sections (see cases/schemas.py)"""
    return "\n\n".join(s["text"] for s in ruling.get("sections", []))


def extract_domain(domain: str, limit: int | None):
    out_dir = f"{FEATURES_DIR}/{domain}"
    os.makedirs(out_dir, exist_ok=True)

    paths = sorted(glob.glob(f"{CASES_DIR}/{domain}/*.json"))
    if limit:
        paths = paths[:limit]

    print(f"🔎 Extracting Features for {len(paths)} cases from «{domain}» "
          f"(limit={limit or 'unlimited'})...")

    resolver = VocabResolver()
    print("📖 Vocabulary and embedding cache loaded for the resolver.")

    done, skipped, failed = 0, 0, 0
    for i, path in enumerate(paths, start=1):
        try:
            with open(path, encoding="utf-8") as f:
                ruling = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ⚠️ [{i}/{len(paths)}] Corrupted/empty file skipped ({path}): {e}")
            failed += 1
            continue

        out_path = f"{out_dir}/{ruling['ruling_id']}.json"
        if os.path.exists(out_path):
            skipped += 1
            continue  # resumable — already extracted

        text = _ruling_full_text(ruling)
        if not text.strip():
            print(f"  ⚠️ [{i}/{len(paths)}] {ruling['ruling_id']}: empty text, skipped")
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
        print(f"  ✅ [{i}/{len(paths)}] {ruling['ruling_id']}: {n_features} features extracted")

    print(f"\n📊 Result: {done} extracted, {skipped} already existed (skipped), {failed} failed")


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
        print(f"⚠️ No extracted files found for «{domain}». "
              f"Run extract first.")
        return

    connection = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASS)
    loader = FeatureGraphLoader(connection)
    loader.create_indexes()

    loaded_count, skipped_count, failed_count = 0, 0, 0
    for i, path in enumerate(paths, start=1):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        result = _dict_to_result(data)

        if loader.is_ruling_loaded(result.ruling_id):
            skipped_count += 1
            if skipped_count % 200 == 0:
                print(f"  ⏭️  {skipped_count} previous items skipped so far...")
            continue  # already loaded -- resumable

        try:
            loader.load_result(result)
            loaded_count += 1
            print(f"  ✅ [{i}/{len(paths)}] {result.ruling_id} loaded")
        except ValueError as e:
            skipped_count += 1
            print(f"  ⚠️ [{i}/{len(paths)}] {e}")
        except Exception as e:
            failed_count += 1
            print(f"  ❌ [{i}/{len(paths)}] Unexpected error on case {result.ruling_id}: {e}")

    connection.close()
    print(f"\n✅ {loaded_count} new results loaded, {skipped_count} skipped (already existed or orphan case), {failed_count} failed.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: uv run main_features.py extract|load <domain> [--limit N]")
        print(f"       (default limit={DEFAULT_PILOT_LIMIT}; for the full dataset: --limit 0)")
        sys.exit(1)

    action, domain = sys.argv[1], sys.argv[2]
    limit = DEFAULT_PILOT_LIMIT
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    limit = limit or None  # --limit 0 means no limit

    if action == "extract":
        extract_domain(domain, limit)
    elif action == "load":
        load_domain(domain, limit)
    else:
        print("❌ Invalid action; use extract or load.")
        sys.exit(1)