from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

from apps.auditor.checklist_builder import build_evidence_bundles
from apps.auditor.pruning import decide
from apps.auditor.verifier import AuditorAPIError, AuditorParseError, verify_article
from apps.search.calibration.leakage_guard import exclude_self_ruling
from apps.search.services import search_service
from apps.auditor.schemas import ChecklistItemOut

_DATA_PATH = Path("apps/search/calibration/data/annotations/case_grounded.json")
_SAMPLE_SIZE = 20

_CACHE_DIR = Path("apps/search/calibration/cache/auditor_eval_articles")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_VERSION = "v2"  # bump manually if the Auditor prompt (verifier.py) changes


def _article_cache_key(query: str, article_ref: str, evidence_text: str) -> str:
    """
    One cache entry PER (query, article, evidence-content) triple --
    not per whole query. This means a crash partway through verifying
    the N candidate articles for one query only costs the articles
    verified before the crash; re-running picks up exactly where it
    left off instead of re-paying for the whole query.

    evidence_text is hashed in (not just article_ref) so that if
    checklist_builder's evidence bundling logic changes for this
    article, the key changes and we don't serve a stale verification
    from before that change.
    """
    payload = json.dumps(
        {
            "version": _CACHE_VERSION,
            "query": query,
            "article_ref": article_ref,
            "evidence_text": evidence_text,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return _CACHE_DIR / f"{key}.pkl"


def _load_cached_verification(key: str):
    path = _cache_path(key)
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def _save_cached_verification(key: str, article_ref: str, checklist, topically_relevant: bool) -> None:
    serializable = {
        "article_ref": article_ref,
        "checklist": [
            {"condition": c.condition, "necessary": c.necessary, "satisfied": c.satisfied}
            for c in checklist
        ],
        "topically_relevant": topically_relevant,
    }
    with _cache_path(key).open("wb") as f:
        pickle.dump(serializable, f)


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

    raw_scores, pruned_scores = [], []
    leaked_evidence_counts = []
    calib_confidence: list[float] = []
    calib_label: list[int] = []
    calib_meta: list[tuple[str, str]] = []

    cache_hits, cache_misses, api_errors_total, parse_errors_total = 0, 0, 0, 0

    for sample in samples:
        query = sample["query"]
        gold = set(sample["gold_articles"])
        ruling_id = sample["ruling_id"]

        search_result_raw = search_service.search(query)
        leaked_count = sum(
            1 for e in search_result_raw.ruling_results if e.ruling_id == ruling_id
        ) + sum(
            1 for e in search_result_raw.feature_results if e.ruling_id == ruling_id
        )
        leaked_evidence_counts.append(leaked_count)

        search_result = exclude_self_ruling(search_result_raw, ruling_id)
        raw_refs = set(search_result.article_refs)

        # --- per-article verification with resumable caching ---
        bundles = build_evidence_bundles(search_result)
        audited_by_ref = {}

        for bundle in bundles:
            evidence_blob = bundle.article_text + "|".join(bundle.supporting_texts)
            key = _article_cache_key(query, bundle.article_ref, evidence_blob)
            cached = _load_cached_verification(key)

            if cached is not None:
                cache_hits += 1
                checklist_dicts = cached["checklist"]
                topically_relevant = cached["topically_relevant"]
            else:
                cache_misses += 1
                try:
                    result = verify_article(query, bundle)
                except AuditorAPIError as e:
                    api_errors_total += 1
                    print(f"  [API ERROR] {bundle.article_ref}: {e}")
                    continue
                except AuditorParseError as e:
                    parse_errors_total += 1
                    print(f"  [PARSE ERROR] {bundle.article_ref}: {e}")
                    continue
                checklist_dicts = [
                    {"condition": c.condition, "necessary": c.necessary, "satisfied": c.satisfied}
                    for c in result.checklist
                ]
                topically_relevant = result.topically_relevant
                _save_cached_verification(key, bundle.article_ref, result.checklist, topically_relevant)

            # decide() needs ChecklistItemOut-like objects; rebuild minimal
            # namespace objects from the cached/plain dicts so pruning.decide
            # doesn't need to change.
            from types import SimpleNamespace
            # checklist_objs = [SimpleNamespace(**c) for c in checklist_dicts]
            checklist_objs = [ChecklistItemOut(**c) for c in checklist_dicts]
            audited_by_ref[bundle.article_ref] = decide(bundle.article_ref, checklist_objs, topically_relevant)

        kept_refs = {ref for ref, a in audited_by_ref.items() if a.is_applicable}

        for ref, a in audited_by_ref.items():
            calib_confidence.append(a.auditor_confidence)
            calib_label.append(1 if ref in gold else 0)
            calib_meta.append((ruling_id, ref))

        raw_scores.append(_prf(raw_refs, gold))
        pruned_scores.append(_prf(kept_refs, gold))

        print(f"[{ruling_id}] self-ruling excluded={leaked_count}  "
              f"raw P/R/F1={raw_scores[-1]}  pruned P/R/F1={pruned_scores[-1]}  "
              f"pruned_count={len(kept_refs)}/{len(raw_refs)}")

    def _avg(scores, idx):
        return sum(s[idx] for s in scores) / len(scores)

    print(f"\n--- per-article cache: {cache_hits} hits / {cache_misses} misses "
          f"(only misses cost API calls) ---")
    print(f"--- errors: {api_errors_total} API errors, {parse_errors_total} parse errors "
          f"(these articles were skipped, not counted as verified) ---")

    print("\n--- averages over", len(samples), "samples ---")
    print(f"raw    precision={_avg(raw_scores,0):.3f}  recall={_avg(raw_scores,1):.3f}  f1={_avg(raw_scores,2):.3f}")
    print(f"pruned precision={_avg(pruned_scores,0):.3f}  recall={_avg(pruned_scores,1):.3f}  f1={_avg(pruned_scores,2):.3f}")
    print(f"\nsamples with >=1 self-ruling evidence piece excluded: "
          f"{sum(1 for c in leaked_evidence_counts if c > 0)}/{len(samples)}  "
          f"(avg {sum(leaked_evidence_counts)/len(samples):.1f} pieces/sample)")

    n_pos = sum(calib_label)
    n_neg = len(calib_label) - n_pos
    print(f"\n--- auditor_confidence calibration ({len(calib_label)} article rows, "
          f"{n_pos} gold / {n_neg} non-gold) ---")

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