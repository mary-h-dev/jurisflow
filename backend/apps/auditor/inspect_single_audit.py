
"""
Debug script: single ruling — runs search + Auditor on one query from the
real annotation file and prints the per-article checklist, applicability
decision, and any API/parse errors in full detail, matched against the
real gold_checklist_by_article for that ruling.

Swap _TARGET_RULING_ID to debug a different case (must exist in
_ANNOTATION_PATH with a non-empty gold_checklist_by_article).


Run:
    DJANGO_SETTINGS_MODULE=config.settings.development python -c "import django; django.setup(); from apps.auditor.inspect_single_audit import run; run()"
"""


from __future__ import annotations

import json
import time
from pathlib import Path

from apps.auditor.pruning import decide
from apps.auditor.services import auditor_service
from apps.auditor.verifier import AuditorAPIError, AuditorParseError
from apps.search.calibration.data import load_case_annotations
from apps.search.services import search_service

_ANNOTATION_PATH = "apps/search/calibration/data/annotations/case_grounded.json"
_TARGET_RULING_ID = "10493"


def _load_gold_checklist_by_article(ruling_id: str) -> dict[str, list[dict]]:
    """
    Reads gold_checklist_by_article straight from the annotation JSON for
    the given ruling_id. Read separately from load_case_annotations()
    below (which returns query/gold_articles as a dataclass) since that
    loader may predate this field -- this avoids touching a shared
    loader other scripts depend on.
    """
    raw = json.loads(Path(_ANNOTATION_PATH).read_text(encoding="utf-8"))
    for sample in raw:
        if sample.get("ruling_id") == ruling_id:
            checklist = sample.get("gold_checklist_by_article")
            if not checklist:
                print(f"[WARNING] ruling_id={ruling_id} has no (or empty) "
                      f"gold_checklist_by_article in the annotation file.")
            return checklist or {}
    print(f"[WARNING] ruling_id={ruling_id} not found while reading gold_checklist_by_article.")
    return {}


def _print_checklist_item(item, gold_satisfied: bool | None) -> None:
    tag = "necessary" if item.necessary else "context  "
    got = "TRUE " if item.satisfied else "FALSE"
    if gold_satisfied is None:
        match = "(no gold for this condition)"
    else:
        match = "✓ MATCH" if item.satisfied == gold_satisfied else f"✗ MISMATCH (gold={gold_satisfied})"
    print(f"    [{tag}] satisfied={got}  {match}")
    print(f"      condition: {item.condition}")


def run() -> None:
    samples = load_case_annotations(_ANNOTATION_PATH)
    target = next((s for s in samples if s.ruling_id == _TARGET_RULING_ID), None)

    if target is None:
        print(f"[ERROR] ruling_id={_TARGET_RULING_ID} not found in {_ANNOTATION_PATH}")
        return

    print("=" * 70)
    print(f"SINGLE RULING AUDIT DEBUG  |  ruling_id={target.ruling_id}")
    print("=" * 70)
    print(f"\nQuery:\n  {target.query}\n")

    gold_checklist_by_article = _load_gold_checklist_by_article(target.ruling_id)

    # -- run the real pipeline: search -> audit ------------------------------
    search_result_raw = search_service.search(target.query)

    from apps.search.calibration.leakage_guard import exclude_self_ruling

    seen_ruling_ids = sorted({
        e.ruling_id for e in (search_result_raw.ruling_results + search_result_raw.feature_results)
        if e.ruling_id is not None
    })
    print(f"target.ruling_id = {target.ruling_id!r}  (type={type(target.ruling_id).__name__})")
    print(f"ruling_ids seen in retrieved evidence ({len(seen_ruling_ids)} distinct): {seen_ruling_ids}")
    print(f"exact match present: {target.ruling_id in seen_ruling_ids}")

    self_ruling_evidence_count = sum(
        1 for e in search_result_raw.ruling_results if e.ruling_id == target.ruling_id
    ) + sum(
        1 for e in search_result_raw.feature_results if e.ruling_id == target.ruling_id
    )

    _EXCLUDE_SELF_RULING = True  # flip to False to compare against the leaked/contaminated run
    if _EXCLUDE_SELF_RULING:
        search_result = exclude_self_ruling(search_result_raw, target.ruling_id)
    else:
        search_result = search_result_raw

    print(f"self-ruling (ruling_id={target.ruling_id}) evidence pieces found: "
          f"{self_ruling_evidence_count}  "
          f"({'EXCLUDED from evidence below' if _EXCLUDE_SELF_RULING else 'INCLUDED -- leakage risk!'})")
    print(f"article_refs BEFORE exclusion: {len(search_result_raw.article_refs)}")
    print(f"article_refs AFTER exclusion:  {len(search_result.article_refs)}")

    print(f"\narticle_refs from retrieval ({len(search_result.article_refs)}):")
    for ref in search_result.article_refs:
        print(f"  - {ref}")

    from django.conf import settings as django_settings
    from apps.auditor.checklist_builder import build_evidence_bundles, debug_rank_article_refs
    from apps.auditor.verifier import verify_article

    from .apps_auditor_config import AUDITOR_MODEL
    provider = AUDITOR_MODEL.model
    delay = getattr(django_settings, "AUDITOR_LLM_REQUEST_DELAY_SECONDS", 4.0)
    from apps.auditor.checklist_builder import _DEFAULT_MAX_ARTICLES_PER_QUERY
    max_articles = getattr(
        django_settings, "AUDITOR_MAX_ARTICLES_PER_QUERY", _DEFAULT_MAX_ARTICLES_PER_QUERY
    )

    ranked = debug_rank_article_refs(search_result)

    print("\n" + "=" * 70)
    print(f"RANKING  (top {max_articles} get sent to the Auditor, rest are cut)")
    print("=" * 70)
    for i, (ref, tier, weight) in enumerate(ranked, start=1):
        cutoff_mark = "  <-- CUTOFF" if i == max_articles else ""
        gold_mark = "  [GOLD]" if ref in gold_checklist_by_article else ""
        kept = "IN " if i <= max_articles else "OUT"
        print(f"  {i:>3}. [{kept}] [{tier:<8}] weight={weight:.4f}  {ref}{gold_mark}{cutoff_mark}")

    gold_refs_debug = set(gold_checklist_by_article)
    ranked_refs_only = [ref for ref, _, _ in ranked]
    print("\n  gold ref positions:")
    for gref in gold_refs_debug:
        if gref in ranked_refs_only:
            pos = ranked_refs_only.index(gref) + 1
            print(f"    {gref}: rank #{pos} of {len(ranked)}  "
                  f"({'MADE cutoff' if pos <= max_articles else 'CUT'})")
        else:
            print(f"    {gref}: NOT in article_refs at all (retrieval never surfaced it)")

    bundles = build_evidence_bundles(search_result)

    print("\n" + "=" * 70)
    print("EVIDENCE BUNDLES FOR GOLD ARTICLES  (exactly what the LLM sees)")
    print("=" * 70)
    for bundle in bundles:
        if bundle.article_ref not in gold_checklist_by_article:
            continue
        print(f"\n[{bundle.article_ref}]")
        print(f"  article_text ({len(bundle.article_text)} chars): {bundle.article_text[:200]}...")
        print(f"  supporting_texts ({len(bundle.supporting_texts)}):")
        if not bundle.supporting_texts:
            print("    (none -- no ruling/feature evidence was attached to this article)")
        for t in bundle.supporting_texts:
            print(f"    - {t}")

    print(f"\nRunning Auditor via model={provider!r}: "
          f"{len(bundles)} call(s), ~{delay}s delay between calls "
          f"(free tiers can be slow -- this can take a few minutes)...\n")

    audited_by_ref = {}
    errored_refs = set()
    api_errors: list[AuditorAPIError] = []
    parse_errors: list[AuditorParseError] = []

    for i, bundle in enumerate(bundles):
        print(f"  [{i + 1}/{len(bundles)}] verifying {bundle.article_ref} ...", end=" ", flush=True)
        if i > 0 and delay:
            time.sleep(delay)

        started = time.monotonic()
        try:
            result = verify_article(target.query, bundle)
        except AuditorAPIError as e:
            api_errors.append(e)
            errored_refs.add(bundle.article_ref)
            print(f"[API ERROR after {time.monotonic() - started:.1f}s] {e}")
            continue
        except AuditorParseError as e:
            parse_errors.append(e)
            errored_refs.add(bundle.article_ref)
            print(f"[PARSE ERROR after {time.monotonic() - started:.1f}s] {e}")
            continue
        audited_by_ref[bundle.article_ref] = decide(bundle.article_ref, result.checklist, result.topically_relevant)
        print(f"ok ({time.monotonic() - started:.1f}s)")

    selected_refs = {bundle.article_ref for bundle in bundles}

    # -- per-article results, matched against embedded gold ------------------
    print("\n" + "=" * 70)
    print("PER-ARTICLE RESULTS")
    print("=" * 70)

    for ref in search_result.article_refs:
        audited = audited_by_ref.get(ref)
        gold_items = gold_checklist_by_article.get(ref)

        print(f"\n[{ref}]")
        if ref not in gold_checklist_by_article:
            print("  (not in gold_articles for this ruling — expected to be pruned)")

        if audited is None:
            if ref in errored_refs:
                print("  -> no result (API or parse error above)")
            elif ref not in selected_refs:
                print(f"  -> SKIPPED — not in the top {len(bundles)} ranked articles, "
                      f"never sent to the Auditor (see AUDITOR_MAX_ARTICLES_PER_QUERY)")
                if ref in gold_checklist_by_article:
                    print("     ⚠ this IS a gold article and got cut by the ranking/cap -- "
                          "raise AUDITOR_MAX_ARTICLES_PER_QUERY or fix the ranking")
            continue

        print(f"  is_applicable={audited.is_applicable}  topically_relevant={audited.topically_relevant}  auditor_confidence={audited.auditor_confidence}")
        gold_by_condition = {g["condition"]: g["satisfied"] for g in (gold_items or [])}
        for item in audited.checklist:
            _print_checklist_item(item, gold_by_condition.get(item.condition))

    # -- summary ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  articles from retrieval : {len(search_result.article_refs)}")
    print(f"  verified successfully   : {len(audited_by_ref)}")
    print(f"  API errors              : {len(api_errors)}")
    print(f"  parse errors            : {len(parse_errors)}")
    kept = {ref for ref, a in audited_by_ref.items() if a.is_applicable}
    print(f"  kept after prune        : {len(kept)}  -> {sorted(kept)}")
    print(f"  gold_articles           : {sorted(target.gold_articles)}")

    gold_refs = set(gold_checklist_by_article)
    missed_by_cap = gold_refs - selected_refs
    if missed_by_cap:
        print(f"\n  ⚠ {len(missed_by_cap)} gold article(s) never reached the Auditor "
              f"(ranked below the top {len(bundles)} cutoff): {sorted(missed_by_cap)}")
        print("    -> raise AUDITOR_MAX_ARTICLES_PER_QUERY, or the ranking in "
              "checklist_builder._rank_article_refs needs another look.")

    if api_errors:
        print("\n  ⚠ API errors occurred — results above are INCOMPLETE. Fix the API")
        print("    issue (key/quota/network) before drawing conclusions from this run.")

    print(
        "\n  note: gold_checklist_by_article above was loaded live from "
        f"{_ANNOTATION_PATH} for ruling_id={target.ruling_id}. To debug a "
        "different case, change _TARGET_RULING_ID at the top of this file."
    )


if __name__ == "__main__":
    run()