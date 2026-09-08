"""
main_laws.py — Rule Graph pipeline for the 5 core statutes (civil, penal,
          commercial, criminal_procedure, civil_procedure)

Two independent steps:

  1) scrape  — parse a locally saved HTML file into a JSON file.
               Must be run with an Iran IP, because qavanin.ir sits behind
               a JS anti-bot challenge and the page has to be saved manually
               from the browser first (see the instructions printed by
               scrape_one when the HTML file is missing).

  2) load    — read the JSON file and write it into Neo4j.
               Must be run from a non-Iran IP / Codespace.

The load step is resumable and resilient to transient network outages:
before writing each Article, it checks whether that article (with the same
content and the same note count) was already fully loaded in a previous
run, and skips it if so. Retries use exponential backoff and only apply to
transient errors (connection/timeout issues) — a logical/data error from
Neo4j (Neo4jError) is not retried, since retrying wouldn't help.

If the network drops mid-run, you don't need to do anything — the script
waits and continues on its own. If it's fully interrupted (e.g. the
Codespace restarts), just run the same command again; articles already
loaded will be skipped.
"""

import os
import sys
import json
import time
import dataclasses

from dotenv import load_dotenv
from neo4j.exceptions import Neo4jError, ServiceUnavailable, SessionExpired

from common.fetcher import load_local_html
from laws.parser import parse_law
from laws.configs import LAW_CONFIGS
from laws.schemas import Law, Article
from database.connection import Neo4jConnection
from database.law_loader import LawGraphLoader, dict_to_law


load_dotenv()

NEO4J_URI  = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")

DATA_DIR = "data/laws"
ERRORS_DIR = "data/features"

MAX_RETRIES = 5
RETRY_BASE_DELAY = 5   # seconds — 5, 10, 20, 40, 80

TRANSIENT_EXCEPTIONS = (
    ServiceUnavailable,
    SessionExpired,
    ConnectionError,
    TimeoutError,
    OSError,
)

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


def _errors_path(key: str) -> str:
    os.makedirs(ERRORS_DIR, exist_ok=True)
    return f"{ERRORS_DIR}/{key}_law_load_errors.json"


# ─────────────────────────────────────────────────────────────────────────
# Step 1 — Scrape (Iran IP only; the HTML must already be saved manually)
# ─────────────────────────────────────────────────────────────────────────

def scrape_one(key: str):
    if key not in LAW_CONFIGS:
        print(f"❌ Unknown key: {key}")
        return

    config = LAW_CONFIGS[key]
    raw_html, raw_json = _paths(key)

    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(raw_html) or os.path.getsize(raw_html) == 0:
        print(f"⚠️ HTML file not found or empty: {raw_html}")
        print("   qavanin.ir sits behind a JS anti-bot challenge and cannot be fetched automatically.")
        print("   Please do the following:")
        print(f"   1) Open in browser: {LAW_URLS.get(key, '(URL not defined)')}")
        print("   2) Wait for the page to fully load.")
        print(f"   3) Ctrl+S -> \"Webpage, Complete\" -> save it exactly at this path: {raw_html}")
        print(f"   4) Run again: python main_laws.py scrape {key}")
        return

    soup = load_local_html(raw_html)
    if not soup:
        print(f"❌ Error reading HTML for {config.law_name}")
        return

    print(f"🔍 Parsing: {config.law_name} ...")
    law = parse_law(soup, config, LAW_URLS.get(key, ""))

    if len(law.articles) == 0:
        print(f"⚠️ No articles found. article_class in laws/configs.py")
        print(f"   probably doesn't match this statute's actual HTML — check with debug_check.py:")
        print(f"   python debug_check.py {raw_html}")
        return

    with open(raw_json, "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(law), f, ensure_ascii=False, indent=2)

    print(f"✅ {len(law.articles)} articles saved -> {raw_json}")


def scrape_all():
    for key in LAW_CONFIGS:
        scrape_one(key)


# ─────────────────────────────────────────────────────────────────────────
# Step 2 — Load (run only from a non-Iran IP / Codespace)
# Resumable + resilient to transient network outages.
# ─────────────────────────────────────────────────────────────────────────

def _article_already_loaded(loader: LawGraphLoader, article: Article, law_name: str) -> bool:
    """
    Returns True if an Article with this exact content already exists *and*
    its Note count matches the note count of this same article in the JSON
    file — meaning it was already fully loaded before, not interrupted
    partway through.
    """
    query = """
    MATCH (a:Article {article_number: $num, law: $law})
    WHERE a.content = $content
    OPTIONAL MATCH (a)-[:HAS_NOTE]->(n:Note)
    RETURN count(n) AS note_count
    """
    with loader.connection.session() as session:
        result = session.run(
            query, num=article.article_number, law=law_name, content=article.content
        )
        record = result.single()
        if record is None:
            return False
        return record["note_count"] >= len(article.notes)


def _connect_with_retry() -> Neo4jConnection:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASS)
        except TRANSIENT_EXCEPTIONS as e:
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"   ⚠️ Connection failed (attempt {attempt}/{MAX_RETRIES}): {e}")
            print(f"      Waiting {delay} seconds and retrying...")
            time.sleep(delay)
    return Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASS)


def _run_with_retry(fn, *args, **kwargs):
    """
    Runs an operation (e.g. _create_law or loading one article) with retry
    against transient network errors. If it still fails after MAX_RETRIES,
    returns (False, error_message); otherwise returns (True, None).
    """
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            fn(*args, **kwargs)
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
            # Logical/data error — retrying won't help
            return False, f"{type(e).__name__}: {e}"
    return False, last_error


def _load_article_with_retry(loader: LawGraphLoader, article: Article, law_name: str) -> tuple[bool, str | None]:
    """
    Equivalent to LawGraphLoader's internal _load_article, but instead of a
    single session.execute_write that puts everything (the article + its
    relationship to the Law + all its notes) in one transaction, it runs the
    same logic with retry.
    """
    def _do_load():
        with loader.connection.session() as session:
            session.execute_write(loader._load_article, article, law_name)

    return _run_with_retry(_do_load)


def _create_law_tx(loader: LawGraphLoader, law: Law):
    with loader.connection.session() as session:
        session.execute_write(loader._create_law, law)


def _create_references_tx(loader: LawGraphLoader, law_name: str):
    with loader.connection.session() as session:
        session.execute_write(loader._create_references, law_name)


def load_one(key: str, loader: LawGraphLoader):
    raw_json = _paths(key)[1]

    if not os.path.exists(raw_json):
        print(f"❌ JSON file not found: {raw_json} — run scrape first.")
        return

    with open(raw_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    law: Law = dict_to_law(data)
    total = len(law.articles)
    print(f"\n📂 {law.name} ({law.domain}): {total} articles found. "
          f"Checking already-loaded articles...")

    # ── Law node step — with retry ──
    ok, err = _run_with_retry(_create_law_tx, loader, law)
    if not ok:
        print(f"❌ Creating the Law node for {law.name} failed: {err}")
        print("   Without the Law node, there's no point continuing — this statute was skipped.")
        return

    loaded = 0
    already_done = 0
    errors: list[dict] = []

    start_time = time.time()

    for i, article in enumerate(law.articles, 1):
        try:
            if _article_already_loaded(loader, article, law.name):
                already_done += 1
                if i % 50 == 0:
                    elapsed = time.time() - start_time
                    print(f"   ... checked {i}/{total} "
                          f"({loaded} newly loaded, {already_done} already existed, {len(errors)} errors) "
                          f"— {elapsed:.0f}s elapsed")
                continue
        except TRANSIENT_EXCEPTIONS:
            print(f"   ⚠️ Transient outage while checking article {article.article_number} — waiting 10s...")
            time.sleep(10)

        ok, err = _load_article_with_retry(loader, article, law.name)

        if ok:
            loaded += 1
        else:
            errors.append({
                "article_number": article.article_number,
                "law": law.name,
                "error_message": err,
            })
            print(f"   🔴 [{i}/{total}] Loading article {article.article_number} "
                  f"failed after {MAX_RETRIES} attempts: {err}")

        if i % 50 == 0:
            elapsed = time.time() - start_time
            print(f"   ... processed {i}/{total} "
                  f"({loaded} newly loaded, {already_done} already existed, {len(errors)} errors) "
                  f"— {elapsed:.0f}s elapsed")

    # ── Final step: create REFERENCES relationships between articles of this statute ──
    ok, err = _run_with_retry(_create_references_tx, loader, law.name)
    if not ok:
        errors.append({
            "article_number": None,
            "law": law.name,
            "error_message": f"Creating REFERENCES relationships failed: {err}",
        })
        print(f"   🔴 Creating REFERENCES relationships for {law.name} failed: {err}")

    elapsed = time.time() - start_time

    if errors:
        err_path = _errors_path(key)
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"\n⚠️  {len(errors)} item(s) failed — details in: {err_path}")
    else:
        print(f"\n✅ No errors occurred.")

    print(f"✅ {law.name}: {loaded} newly loaded, {already_done} already existed, "
          f"{len(errors)} errors, out of {total} articles total.")
    print(f"   Total time: {elapsed:.0f}s")


def load_all():
    print("💾 Connecting to Neo4j...")
    connection = _connect_with_retry()
    loader = LawGraphLoader(connection)

    try:
        loader.create_indexes()
        for key in LAW_CONFIGS:
            load_one(key, loader)
    finally:
        connection.close()

    print("\n✅ Rule Graph loading finished!")


def load_single(key: str):
    if key not in LAW_CONFIGS:
        print(f"❌ Unknown key: {key}")
        print(f"   Valid keys: {', '.join(LAW_CONFIGS.keys())}")
        return

    print("💾 Connecting to Neo4j...")
    connection = _connect_with_retry()
    loader = LawGraphLoader(connection)

    try:
        loader.create_indexes()
        load_one(key, loader)
    finally:
        connection.close()

    print("\n✅ Loading finished!")


# ─────────────────────────────────────────────────────────────────────────

def _usage():
    print("Usage:")
    print("  uv run main_laws.py scrape <civil|penal|commercial|criminal_procedure|civil_procedure|all>")
    print("  uv run main.laws.py load   <civil|penal|commercial|criminal_procedure|civil_procedure|all>")


def _run_load(target: str):
    """
    Runs the load step with an outer retry loop: if a network outage
    happens at the top level of the program (e.g. between statutes), wait
    and restart the whole operation. Already-loaded articles are skipped
    thanks to the per-article resumability check in load_one.
    """
    while True:
        try:
            load_all() if target == "all" else load_single(target)
            break
        except TRANSIENT_EXCEPTIONS as e:
            print(f"\n⚠️ Network outage at the top level of the program: {e}")
            print("   Waiting 15 seconds and restarting the whole operation "
                  "(already-loaded articles will be skipped)...")
            time.sleep(15)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        _usage()
        sys.exit(1)

    mode, target = sys.argv[1], sys.argv[2]

    if mode == "scrape":
        scrape_all() if target == "all" else scrape_one(target)

    elif mode == "load":
        _run_load(target)

    else:
        _usage()