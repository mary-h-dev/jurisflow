"""
main_cases.py — two-stage pipeline for the cases/rulings data (Fact Graph)

⚠️ This file is independent of main_laws.py (which is only for the 5 core
statutes) — like main_laws.py, it has two separate stages (scrape / load) and
uses the same shared common/fetcher.py and database/connection.py.

Dynamic with respect to statutes:
    Instead of a manual list of the 5 statutes, this file loops directly
    over laws.configs.LAW_CONFIGS. So if a 6th statute is later added to
    laws/configs.py, this pipeline will automatically scrape that
    statute's cases too — as long as its ID on the rulings system is also
    registered in cases/configs.py (otherwise it stops with a clear error,
    see cases/configs.py).

About case_type:
    Regardless of which statute it was found through, each Ruling gets its
    own case_type field (civil/criminal) from "گروه رأی" — because a
    criminal ruling might also cite a civil article (e.g. a criminal bad-
    check ruling citing the Commercial Code). The final report shows this
    distribution separately.

The load stage is resumable and resilient to transient network outages:
before loading each ruling, it checks whether that Ruling already exists
in Neo4j *and* is "complete" (i.e. has at least one HAS_SECTION), and skips
it if so. Retries use exponential backoff (5, 10, 20, 40, 80s) and only
apply to transient errors — a logical/data error from Neo4j (Neo4jError)
is not retried, since retrying wouldn't help; that ruling is instead
recorded in the error report and the script moves on to the next one.

If the network drops mid-run, you don't need to do anything — the script
waits and continues on its own. If it's fully interrupted (e.g. the
Codespace restarts), just run the same command again; already-loaded
rulings will be skipped.

Usage:
    uv run main_cases.py scrape civil
    uv run main_cases.py scrape all

    uv run main_cases.py load civil
    uv run main_cases.py load all
"""

import os
import sys
import json
import time
import traceback
import dataclasses
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from neo4j.exceptions import Neo4jError, ServiceUnavailable, SessionExpired

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
ERRORS_DIR = "data/features"
REQUEST_DELAY = 1.5   # seconds — respect the site's request rate

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fa-IR,fa;q=0.9,en;q=0.8",
}

LARGE_PAGE_SIZE = 2500   # larger than our biggest statute (civil: 1934)

# Retry settings for transient network outages
MAX_RETRIES = 5
RETRY_BASE_DELAY = 5   # seconds — doubles each time (5, 10, 20, 40, 80)

# Errors considered "transient" / network-related, which should be retried
TRANSIENT_EXCEPTIONS = (
    ServiceUnavailable,
    SessionExpired,
    ConnectionError,
    TimeoutError,
    OSError,   # includes DNS outages, socket resets, and similar
)


def _law_dir(law_key: str) -> str:
    path = f"{DATA_DIR}/{law_key}"
    os.makedirs(path, exist_ok=True)
    return path


def _ruling_path(law_key: str, ruling_id: str) -> str:
    return f"{_law_dir(law_key)}/{ruling_id}.json"


def _errors_path(law_key: str) -> str:
    os.makedirs(ERRORS_DIR, exist_ok=True)
    return f"{ERRORS_DIR}/{law_key}_load_errors.json"


# ─────────────────────────────────────────────────────────────────────────
# Step 1: fetch the list of ruling IDs for each statute
# ─────────────────────────────────────────────────────────────────────────

def _extract_ruling_ids(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    links = soup.find_all("a", href=lambda h: h and "/Judge/Text/" in h)
    ids = [a.get("href").rstrip("/").split("/")[-1] for a in links]
    # keep the order but drop duplicates
    seen = set()
    unique_ids = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            unique_ids.append(i)
    return unique_ids


def fetch_all_ruling_ids(law_search_id: int) -> list[str]:
    """
    First tries a large PageSize to get everything in one request (faster).
    If that doesn't work for any reason (fewer results than expected), it
    automatically falls back to page-by-page mode with the default PageSize.
    """
    base_url = f"https://ara.jri.ac.ir/Judge/Index?Laws={law_search_id}"

    # First attempt: a single request with a large PageSize
    r = requests.get(f"{base_url}&PageSize={LARGE_PAGE_SIZE}", headers=HEADERS, timeout=20)
    r.encoding = "utf-8"
    ids = _extract_ruling_ids(r.text)
    time.sleep(REQUEST_DELAY)

    if ids:
        return ids

    # Fallback: page-by-page with the default size (25)
    print("   ⚠️ Large PageSize didn't work — falling back to page-by-page mode...")
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

        if page > 500:   # guard against an infinite loop
            print("   ⚠️ Reached the 500-page cap — stopping (this is abnormal).")
            break

    return all_ids


# ─────────────────────────────────────────────────────────────────────────
# Step 2: scrape — fetch + parse + save each ruling (resumable)
# ─────────────────────────────────────────────────────────────────────────

def scrape_one_law(law_key: str):
    if law_key not in LAW_CONFIGS:
        print(f"❌ Unknown statute key: {law_key}")
        return

    law_search_id = get_case_search_id(law_key)   # raises a clear error if not registered
    law_name = LAW_CONFIGS[law_key].law_name

    print(f"\n🔎 {law_name} (rulings-system id: {law_search_id})")
    ruling_ids = fetch_all_ruling_ids(law_search_id)
    print(f"   {len(ruling_ids)} rulings found.")

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
                print(f"   ... {i}/{len(ruling_ids)} processed ({fetched} new, {skipped} already existed)")

        except Exception as e:
            print(f"   ❌ Error on {ruling_id}: {e}")
            failed += 1

        time.sleep(REQUEST_DELAY)

    print(f"✅ {law_name}: {fetched} new, {skipped} already existed, {failed} errors")


def scrape_all():
    for law_key in LAW_CONFIGS:
        scrape_one_law(law_key)


# ─────────────────────────────────────────────────────────────────────────
# Step 3: Load (run only from a non-Iran IP / Codespace)
# Resumable + resilient to transient network outages.
# ─────────────────────────────────────────────────────────────────────────

def _already_loaded(loader: CaseGraphLoader, ruling_id: str) -> bool:
    """
    Returns True if this ruling_id already exists in the graph *and* has at
    least one RulingSection (meaning its previous load ran to completion,
    rather than being interrupted partway through with only an empty
    Ruling node created).
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
    """Retries the initial connection a few times if it hits an outage too."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASS)
        except TRANSIENT_EXCEPTIONS as e:
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"   ⚠️ Connection failed (attempt {attempt}/{MAX_RETRIES}): {e}")
            print(f"      Waiting {delay} seconds and retrying...")
            time.sleep(delay)
    # Final attempt without catching the exception — if it fails again, it propagates
    return Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASS)


def _load_with_retry(loader: CaseGraphLoader, ruling) -> tuple[bool, str | None]:
    """
    Attempts to load a ruling with retry against transient network errors.
    Returns: (whether it succeeded, error message on final failure)
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
                print(f"      ⏳ Transient network error (attempt {attempt}/{MAX_RETRIES}) — "
                      f"waiting {delay}s and retrying... ({last_error})")
                time.sleep(delay)
            else:
                return False, last_error

        except Neo4jError as e:
            # Logical/data error (not network) — retrying won't help, fail immediately
            return False, f"{type(e).__name__}: {e}"

    return False, last_error


def load_one_law(law_key: str, loader: CaseGraphLoader):
    law_dir = _law_dir(law_key)

    if not os.path.isdir(law_dir):
        print(f"❌ Folder not found: {law_dir} — run scrape first.")
        return

    files = [f for f in os.listdir(law_dir) if f.endswith(".json")]

    if not files:
        print(f"❌ No files found for {law_key} — run scrape first.")
        return

    total = len(files)
    print(f"\n📂 {law_key}: {total} files found. Checking already-loaded rulings...")

    case_type_counts: dict[str, int] = {}
    succeeded = 0
    already_done = 0
    errors: list[dict] = []

    start_time = time.time()

    for i, filename in enumerate(files, 1):
        ruling_id = filename.replace(".json", "")
        path = os.path.join(law_dir, filename)

        # ── resumability check: skip if already fully loaded ──
        try:
            if _already_loaded(loader, ruling_id):
                already_done += 1
                if i % 50 == 0:
                    elapsed = time.time() - start_time
                    print(f"   ... checked {i}/{total} "
                          f"({succeeded} newly loaded, {already_done} already existed, {len(errors)} errors) "
                          f"— {elapsed:.0f}s elapsed")
                continue
        except TRANSIENT_EXCEPTIONS:
            # even the check itself can hit an outage — wait and continue
            print(f"   ⚠️ Transient outage while checking {ruling_id} — waiting 10s...")
            time.sleep(10)

        # ── read the file ──
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()

            if not raw.strip():
                raise ValueError("File is empty (0 bytes or whitespace only)")

            data = json.loads(raw)
            ruling = dict_to_ruling(data)

        except json.JSONDecodeError as e:
            errors.append({
                "ruling_id": ruling_id, "file": path,
                "error_type": "JSONDecodeError", "error_message": str(e),
            })
            print(f"   🔴 [{i}/{total}] Corrupt JSON: {ruling_id} — {e}")
            continue

        except Exception as e:
            errors.append({
                "ruling_id": ruling_id, "file": path,
                "error_type": type(e).__name__, "error_message": str(e),
                "traceback": traceback.format_exc(),
            })
            print(f"   🔴 [{i}/{total}] Error reading file {ruling_id}: {type(e).__name__}: {e}")
            continue

        # ── load with retry against transient outages ──
        ok, err_msg = _load_with_retry(loader, ruling)

        if ok:
            case_type_counts[ruling.case_type] = case_type_counts.get(ruling.case_type, 0) + 1
            succeeded += 1
        else:
            errors.append({
                "ruling_id": ruling_id, "file": path,
                "error_type": "LoadFailedAfterRetries", "error_message": err_msg,
            })
            print(f"   🔴 [{i}/{total}] Loading {ruling_id} failed after {MAX_RETRIES} attempts: {err_msg}")

        if i % 50 == 0:
            elapsed = time.time() - start_time
            print(f"   ... processed {i}/{total} "
                  f"({succeeded} newly loaded, {already_done} already existed, {len(errors)} errors) "
                  f"— {elapsed:.0f}s elapsed")

    elapsed = time.time() - start_time

    if errors:
        err_path = _errors_path(law_key)
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"\n⚠️  {len(errors)} ruling(s) failed — details in: {err_path}")
    else:
        print(f"\n✅ No errors occurred.")

    print(f"✅ {law_key}: {succeeded} newly loaded, {already_done} already existed, "
          f"{len(errors)} errors, out of {total} files total.")
    print(f"   Case-type distribution (newly loaded): {case_type_counts}")
    print(f"   Total time: {elapsed:.0f}s")


def load_all():
    print("💾 Connecting to Neo4j...")
    connection = _connect_with_retry()
    loader = CaseGraphLoader(connection)

    try:
        loader.create_indexes()
        for law_key in LAW_CONFIGS:
            load_one_law(law_key, loader)
    finally:
        connection.close()

    print("\n✅ Fact Graph loading finished!")


def load_single(law_key: str):
    if law_key not in LAW_CONFIGS:
        print(f"❌ Unknown statute key: {law_key}")
        print(f"   Valid keys: {', '.join(LAW_CONFIGS.keys())}")
        return

    print("💾 Connecting to Neo4j...")
    connection = _connect_with_retry()
    loader = CaseGraphLoader(connection)

    try:
        loader.create_indexes()
        load_one_law(law_key, loader)
    finally:
        connection.close()

    print("\n✅ Loading finished!")


# ─────────────────────────────────────────────────────────────────────────

def _usage():
    keys = "|".join(LAW_CONFIGS.keys())
    print("Usage:")
    print(f"  uv run main_cases.py scrape <{keys}|all>")
    print(f"  uv run main_cases.py load   <{keys}|all>")


def _run_load(target: str):
    """
    Runs the load step with an outer retry loop: if a network outage
    happens at the top level of the program (not just for one ruling),
    wait and restart the whole operation. Already-loaded rulings are
    skipped thanks to the per-ruling resumability check in load_one_law.
    """
    while True:
        try:
            load_all() if target == "all" else load_single(target)
            break
        except TRANSIENT_EXCEPTIONS as e:
            print(f"\n⚠️ Network outage at the top level of the program: {e}")
            print("   Waiting 15 seconds and restarting the whole operation "
                  "(already-loaded rulings will be skipped)...")
            time.sleep(15)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        _usage()
        sys.exit(1)

    mode, target = sys.argv[1], sys.argv[2]

    if mode == "scrape":
        scrape_all() if target == "all" else scrape_one_law(target)

    elif mode == "load":
        _run_load(target)

    else:
        _usage()