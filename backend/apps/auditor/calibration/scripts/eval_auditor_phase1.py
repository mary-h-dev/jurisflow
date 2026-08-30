#  DJANGO_SETTINGS_MODULE=config.settings.development python -c "import django; django.setup(); from apps.search.calibration.eval_auditor_phase1 import run; run()"

from __future__ import annotations

import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

from apps.auditor.checklist_builder import build_evidence_bundles
from apps.auditor.pruning import decide
from apps.auditor.schemas import ChecklistItemOut
from apps.auditor.verifier import AuditorAPIError, AuditorParseError, verify_article
from apps.search.calibration.leakage_guard import exclude_self_ruling
from apps.search.services import search_service

_DATA_PATH = Path("apps/search/calibration/data/annotations/case_grounded.json")
_SAMPLE_SIZE = 20

# One cache file PER RULING (not per article) -- much smaller footprint on
# disk. A crash mid-ruling costs only that one ruling's API calls, not the
# whole run; every ruling that finished before the crash is still cached.
_CACHE_DIR = Path("apps/search/calibration/cache/auditor_eval_rulings")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_VERSION = "v2"  # bump manually if the Auditor prompt (verifier.py) or
                        # checklist_db.json changes -- forces a fresh re-run


def _ruling_cache_key(ruling_id: str, query: str) -> str:
    payload = json.dumps(
        {"version": _CACHE_VERSION, "ruling_id": ruling_id, "query": query},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return _CACHE_DIR / f"{key}.pkl"


def _load_cached_ruling(key: str):
    path = _cache_path(key)
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def _save_cached_ruling(key: str, verified: list[dict]) -> None:
    with _cache_path(key).open("wb") as f:
        pickle.dump(verified, f)


def _prf(predicted: set[str], gold: set[str]) -> tuple[float, float, float]:
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    tp = len(predicted & gold)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def run(sample_size: int = _SAMPLE_SIZE) -> None:
    samples = json.loads(_DATA_PATH.read_text(encoding="utf-8"))[:sample_size]

    print("=" * 70)
    print(f"AUDITOR EVAL  |  {len(samples)} rulings queued  |  cache_version={_CACHE_VERSION}")
    print("=" * 70)

    raw_scores, pruned_scores = [], []
    leaked_evidence_counts = []
    calib_confidence: list[float] = []
    calib_label: list[int] = []

    ruling_cache_hits, ruling_cache_misses = 0, 0
    api_errors_total, parse_errors_total = 0, 0

    for idx, sample in enumerate(samples, start=1):
        t_start = time.monotonic()
        query = sample["query"]
        gold = set(sample["gold_articles"])
        ruling_id = sample["ruling_id"]

        print(f"\n[{idx}/{len(samples)}] ruling_id={ruling_id}  "
              f"gold={sorted(gold)}")

        ruling_key = _ruling_cache_key(ruling_id, query)
        cached_verified = _load_cached_ruling(ruling_key)

        if cached_verified is not None:
            ruling_cache_hits += 1
            print(f"  [CACHE HIT] loaded {len(cached_verified)} verified articles, "
                  f"0 API calls")
            verified = cached_verified
            # still need raw_refs for the raw-P/R/F1 metric -- cheap, no LLM cost
            search_result_raw = search_service.search(query)
            leaked_count = sum(
                1 for e in search_result_raw.ruling_results if e.ruling_id == ruling_id
            ) + sum(
                1 for e in search_result_raw.feature_results if e.ruling_id == ruling_id
            )
            search_result = exclude_self_ruling(search_result_raw, ruling_id)
            raw_refs = set(search_result.article_refs)
        else:
            ruling_cache_misses += 1
            print(f"  [CACHE MISS] running retrieval + Auditor from scratch...")

            search_result_raw = search_service.search(query)
            leaked_count = sum(
                1 for e in search_result_raw.ruling_results if e.ruling_id == ruling_id
            ) + sum(
                1 for e in search_result_raw.feature_results if e.ruling_id == ruling_id
            )
            search_result = exclude_self_ruling(search_result_raw, ruling_id)
            raw_refs = set(search_result.article_refs)

            bundles = build_evidence_bundles(search_result)
            print(f"  retrieval: {len(raw_refs)} candidate articles, "
                  f"{len(bundles)} sent to Auditor (self-ruling excluded={leaked_count})")

            verified = []
            for i, bundle in enumerate(bundles, start=1):
                print(f"    [{i}/{len(bundles)}] verifying {bundle.article_ref} ...",
                      end=" ", flush=True)
                t0 = time.monotonic()
                try:
                    result = verify_article(query, bundle)
                except AuditorAPIError as e:
                    api_errors_total += 1
                    print(f"API ERROR ({time.monotonic()-t0:.1f}s): {e}")
                    continue
                except AuditorParseError as e:
                    parse_errors_total += 1
                    print(f"PARSE ERROR ({time.monotonic()-t0:.1f}s): {e}")
                    continue

                verified.append({
                    "article_ref": bundle.article_ref,
                    "checklist": [
                        {"condition": c.condition, "necessary": c.necessary, "satisfied": c.satisfied}
                        for c in result.checklist
                    ],
                    "topically_relevant": result.topically_relevant,
                })
                print(f"ok ({time.monotonic()-t0:.1f}s)")

            # save the whole ruling's results at once -- only after a
            # successful full pass, so a crash mid-ruling leaves no
            # partial/corrupt cache entry for it
            _save_cached_ruling(ruling_key, verified)
            print(f"  [SAVED] {len(verified)} verified articles cached for this ruling")

        leaked_evidence_counts.append(leaked_count)

        # -- apply pruning decision to every verified article --
        audited_by_ref = {}
        for v in verified:
            checklist_objs = [ChecklistItemOut(**c) for c in v["checklist"]]
            audited_by_ref[v["article_ref"]] = decide(
                v["article_ref"], checklist_objs, v["topically_relevant"]
            )

        kept_refs = {ref for ref, a in audited_by_ref.items() if a.is_applicable}

        for ref, a in audited_by_ref.items():
            calib_confidence.append(a.auditor_confidence)
            calib_label.append(1 if ref in gold else 0)

        raw_scores.append(_prf(raw_refs, gold))
        pruned_scores.append(_prf(kept_refs, gold))

        elapsed = time.monotonic() - t_start
        print(f"  raw    P/R/F1 = {raw_scores[-1][0]:.3f} / {raw_scores[-1][1]:.3f} / {raw_scores[-1][2]:.3f}")
        print(f"  pruned P/R/F1 = {pruned_scores[-1][0]:.3f} / {pruned_scores[-1][1]:.3f} / {pruned_scores[-1][2]:.3f}"
              f"   kept={sorted(kept_refs)}")
        print(f"  [done in {elapsed:.1f}s]")

    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    def _avg(scores, idx):
        return sum(s[idx] for s in scores) / len(scores)

    print(f"\nruling-level cache: {ruling_cache_hits} hits / {ruling_cache_misses} misses "
          f"(only misses cost API calls)")
    print(f"errors: {api_errors_total} API errors, {parse_errors_total} parse errors "
          f"(skipped, not counted as verified)")

    print(f"\naverages over {len(samples)} samples:")
    print(f"  raw    precision={_avg(raw_scores,0):.3f}  recall={_avg(raw_scores,1):.3f}  f1={_avg(raw_scores,2):.3f}")
    print(f"  pruned precision={_avg(pruned_scores,0):.3f}  recall={_avg(pruned_scores,1):.3f}  f1={_avg(pruned_scores,2):.3f}")
    print(f"\nsamples with >=1 self-ruling evidence piece excluded: "
          f"{sum(1 for c in leaked_evidence_counts if c > 0)}/{len(samples)}  "
          f"(avg {sum(leaked_evidence_counts)/len(samples):.1f} pieces/sample)")

    n_pos = sum(calib_label)
    n_neg = len(calib_label) - n_pos
    print(f"\nauditor_confidence calibration ({len(calib_label)} article rows, "
          f"{n_pos} gold / {n_neg} non-gold):")

    if n_pos == 0 or n_neg == 0:
        print("  cannot compute AUROC/Spearman: need both classes present")
        return

    auroc = roc_auc_score(calib_label, calib_confidence)
    ap = average_precision_score(calib_label, calib_confidence)
    rho, pval = spearmanr(calib_confidence, calib_label)

    print(f"  AUROC:    {auroc:.3f}")
    print(f"  PR-AUC:   {ap:.3f}   (baseline = {n_pos/len(calib_label):.3f})")
    print(f"  Spearman: {rho:.3f}  (p={pval:.3g})")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else _SAMPLE_SIZE
    run(n)