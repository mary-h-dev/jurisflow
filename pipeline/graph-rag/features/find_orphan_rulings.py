"""
features/find_orphan_rulings.py — precisely finds ruling_ids that exist
in data/features/<domain>/*.json but have no matching Ruling node in
Neo4j.

Why was this script needed?
    Two different diagnoses were proposed (a type mismatch from one AI, a
    leftover from an incomplete migration from another). Rather than
    picking between guesses, this script directly checks, for *every*
    ruling_id, whether the matching node exists (as both string and
    number type), and reports the exact list of orphans.

Usage:
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
    For each ruling_id, checks: does it exist as a string type? As an
    integer type? Output: {"missing_both": [...], "found_as_string": n, "found_as_int": n}
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
    print(f"📂 Read {len(ruling_ids)} ruling_ids from the extracted files for «{domain}»")

    connection = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASS)
    try:
        stats = find_orphans(ruling_ids, connection)
    finally:
        connection.close()

    print(f"\n📊 Result:")
    print(f"  ✅ Found as string type: {stats['found_as_string']}")
    print(f"  ✅ Found as int type   : {stats['found_as_int']}  "
          f"(if this is > 0, the type mismatch was genuinely part of the problem)")
    print(f"  🔴 Not found at all    : {len(stats['missing_both'])}  "
          f"(their Ruling node was never created in the Case graph)")

    if stats["missing_both"]:
        out_path = f"data/features/{domain}_orphan_ruling_ids.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(stats["missing_both"]))
        print(f"\n💾 Full list of orphan ruling_ids saved: {out_path}")
        print(f"   Sample (first 10): {stats['missing_both'][:10]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run -m features.find_orphan_rulings <domain>")
        sys.exit(1)
    run(sys.argv[1])