"""
Calibrates apps/auditor/confidence.py's combine_confidence weights via
logistic regression, reusing the ALREADY-CACHED per-article Auditor
Platt-scaling data (no new Gemini/Auditor API calls needed) joined with
freshly-recomputed retrieval scores (also free: router_cache.py reads
from disk, embedding runs on local Ollama -- neither touches a paid API).

Steps:
  1. Load the cached per-article calibration data (train_val + test).
  2. Aggregate to per-ruling_id (query-level): auditor_score (mean
     raw_auditor_confidence over predicted-applicable articles, matching
     apps/auditor/confidence.compute_auditor_confidence's own logic),
     prune_ratio, and an F1-based binary success label.
  3. Recompute retrieval_score per ruling_id by re-running
     search_service.search() -- cheap, no Auditor/LLM calls involved.
  4. Fit logistic regression: [retrieval_score, auditor_score,
     prune_ratio] -> binary success (F1 vs gold >= 0.7, matching the
     0.7 threshold convention already used elsewhere in calibration/).
  5. Print fitted coefficients AND test-set accuracy/AUROC, so you can
     either (a) hardcode the coefficients as the new weights in
     combine_confidence, or (b) replace the whole hand-built formula
     with model.predict_proba() directly -- see note printed at the end.


calibration data was actually cached (the "per_sample": {"train_val":
[...], "test": [...]} JSON you already have).

Run:
    DJANGO_SETTINGS_MODULE=config.settings.development python -c "
    import django; django.setup()
    from apps.auditor.calibrate_confidence_fusion import run
    run()
    "
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from apps.search.services import search_service

_PER_ARTICLE_CACHE_PATH = "apps/auditor/calibration/data/auditor_calibration_results.json"  
_ANNOTATION_PATH = "apps/search/calibration/data/annotations/case_grounded.json"
_SUCCESS_F1_THRESHOLD = 0.7  


def _load_query_text_by_ruling_id() -> dict[str, str]:
    samples = json.loads(Path(_ANNOTATION_PATH).read_text(encoding="utf-8"))
    return {s["ruling_id"]: s["query"] for s in samples}


def _f1(predicted: set[str], gold: set[str]) -> float:
    if not predicted and not gold:
        return 1.0
    tp = len(predicted & gold)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def _aggregate_to_query_level(rows: list[dict]) -> dict[str, dict]:
    """
    Groups per-article rows by ruling_id and computes the same
    auditor_score/prune_ratio apps/auditor/confidence.py's
    compute_auditor_confidence produces, plus a gold/kept set for the
    F1-based success label.
    """
    by_ruling: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_ruling[row["ruling_id"]].append(row)

    aggregated = {}
    for ruling_id, article_rows in by_ruling.items():
        kept = {r["article_ref"] for r in article_rows if r["predicted_is_applicable"]}
        gold = {r["article_ref"] for r in article_rows if r["gold_is_applicable"]}

        kept_confidences = [r["raw_auditor_confidence"] for r in article_rows if r["predicted_is_applicable"]]
        auditor_score = sum(kept_confidences) / len(kept_confidences) if kept_confidences else 0.0

        total = len(article_rows)
        pruned = sum(1 for r in article_rows if not r["predicted_is_applicable"])
        prune_ratio = pruned / total if total else 0.0

        aggregated[ruling_id] = {
            "auditor_score": round(auditor_score, 3),
            "prune_ratio": round(prune_ratio, 3),
            "f1": round(_f1(kept, gold), 3),
        }
    return aggregated


def _build_feature_matrix(aggregated: dict[str, dict], query_by_ruling: dict[str, str]) -> tuple[list[list[float]], list[int], list[str]]:
    X, y, used_ruling_ids = [], [], []

    for ruling_id, agg in aggregated.items():
        query = query_by_ruling.get(ruling_id)
        if query is None:
            print(f"  [SKIP] ruling_id={ruling_id}: no query text found in annotation file")
            continue

        # Free: router_cache.py reads from disk if this query was seen
        # before, embedding runs on local Ollama -- no paid API call.
        search_result = search_service.search(query)
        retrieval_score = search_result.confidence.score

        X.append([retrieval_score, agg["auditor_score"], agg["prune_ratio"]])
        y.append(1 if agg["f1"] >= _SUCCESS_F1_THRESHOLD else 0)
        used_ruling_ids.append(ruling_id)

    return X, y, used_ruling_ids


def run() -> None:
    cache = json.loads(Path(_PER_ARTICLE_CACHE_PATH).read_text(encoding="utf-8"))
    train_val_rows = cache["per_sample"]["train_val"]
    test_rows = cache["per_sample"]["test"]

    query_by_ruling = _load_query_text_by_ruling_id()

    print("Aggregating cached per-article data to query-level...")
    train_val_agg = _aggregate_to_query_level(train_val_rows)
    test_agg = _aggregate_to_query_level(test_rows)

    print(f"  train_val: {len(train_val_agg)} queries")
    print(f"  test:      {len(test_agg)} queries")

    print("\nRecomputing retrieval_score per query (free -- cached router + local embedding)...")
    X_train, y_train, _ = _build_feature_matrix(train_val_agg, query_by_ruling)
    X_test, y_test, _ = _build_feature_matrix(test_agg, query_by_ruling)

    print(f"\nFitting logistic regression on {len(X_train)} train_val samples...")
    model = LogisticRegression()
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    test_auroc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if len(set(y_test)) > 1 else float("nan")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  coefficients (retrieval, auditor, prune_ratio): {model.coef_[0]}")
    print(f"  intercept: {model.intercept_[0]:.4f}")
    print(f"  train accuracy: {train_acc:.3f}")
    print(f"  test accuracy:  {test_acc:.3f}")
    print(f"  test AUROC:     {test_auroc:.3f}")

    print(
        "\nTwo ways to use this in apps/auditor/confidence.py:\n"
        "  (a) Hardcode the three coefficients (normalized to sum to 1, "
        "keeping their relative sign/magnitude) as _RETRIEVAL_WEIGHT / "
        "_AUDITOR_WEIGHT / _PRUNE_RATIO_PENALTY_WEIGHT in the existing "
        "linear formula -- minimal code change, but loses the sigmoid's "
        "shape.\n"
        "  (b) Replace combine_confidence's formula entirely with "
        "model.predict_proba() using these exact coefficients (still "
        "fully deterministic, just a calibrated logistic function "
        "instead of a hand-built weighted sum) -- matches the same idea "
        "as the Platt scaling you already did at the article level, "
        "just applied one level up. Recommended if test AUROC above is "
        "meaningfully better than the current formula's."
    )


if __name__ == "__main__":
    run()