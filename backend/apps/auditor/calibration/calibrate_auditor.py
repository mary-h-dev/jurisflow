"""
Auditor-only calibration: evaluates the Auditor's EVALUATION step in
isolation from retrieval/ranking. Uses gold_checklist_by_article as the
fixed input checklist (bypasses checklist generation entirely) and asks
the model only to judge satisfied=true/false for each condition, plus
topical relevance -- exactly the evaluation-only prompt path already
built for checklist_db.json.

Why this isolates the right thing: this removes two confounds we hit
earlier -- (1) ranking noise (gold articles cut before reaching the
Auditor), and (2) checklist-generation drift (Auditor inventing
competing-article conditions). What's left is a clean measurement of:
"given the correct conditions, does the Auditor judge them correctly?"

Run (dry-run first, ALWAYS do this before spending anything):
    DJANGO_SETTINGS_MODULE=config.settings.development python -c \
        "import django; django.setup(); from apps.auditor.calibration.calibrate_auditor import dry_run; dry_run()"

Run (actual calibration, after checking the dry-run cost estimate):
    DJANGO_SETTINGS_MODULE=config.settings.development python -c \
        "import django; django.setup(); from apps.auditor.calibration.calibrate_auditor import run; run()"
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from apps.auditor.checklist_builder import _fetch_article_texts
from apps.auditor.verifier import (
    AuditorAPIError, AuditorParseError, _EVALUATION_ONLY_PROMPT_TEMPLATE,
    _truncate, _MAX_ARTICLE_TEXT_CHARS,
)
from apps.search.calibration.data import load_case_annotations, stratified_test_split
from core.llm_client import get_client
from apps.auditor.apps_auditor_config import AUDITOR_MODEL

_ANNOTATION_PATH = "apps/search/calibration/data/annotations/case_grounded.json"
_CACHE_DIR = Path(__file__).resolve().parent / "data" / "calibration_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_VERSION = "v1"


# ---------------------------------------------------------------------- #
# Data collection: (ruling_id, article_ref) pairs with gold checklists
# ---------------------------------------------------------------------- #

@dataclass
class GoldChecklistPair:
    ruling_id: str
    query: str
    article_ref: str
    gold_conditions: list[str]           # condition text, in order
    gold_satisfied: list[bool]           # ground truth per condition
    gold_is_applicable: bool             # AND over gold_satisfied


def _collect_pairs(samples) -> list[GoldChecklistPair]:
    pairs = []
    for sample in samples:
        gold_by_article = getattr(sample, "gold_checklist_by_article", None)
        if not gold_by_article:
            continue
        for article_ref, items in gold_by_article.items():
            if not items:
                continue
            conditions = [it["condition"] for it in items]
            satisfied = [bool(it["satisfied"]) for it in items]
            pairs.append(GoldChecklistPair(
                ruling_id=sample.ruling_id,
                query=sample.query,
                article_ref=article_ref,
                gold_conditions=conditions,
                gold_satisfied=satisfied,
                gold_is_applicable=all(satisfied),
            ))
    return pairs


def dry_run(sample_size: int | None = None) -> None:
    """Prints the exact call count and a cost estimate -- makes ZERO API
    calls. Always run this before run()."""
    samples = load_case_annotations(_ANNOTATION_PATH)
    if sample_size:
        samples = samples[:sample_size]
    train_val, test = stratified_test_split(samples, test_fraction=0.2)

    train_pairs = _collect_pairs(train_val)
    test_pairs = _collect_pairs(test)
    total = len(train_pairs) + len(test_pairs)

    print("=" * 60)
    print("DRY RUN -- no API calls made")
    print("=" * 60)
    print(f"train_val samples: {len(train_val)}  ->  {len(train_pairs)} (ruling, article) pairs")
    print(f"test samples:      {len(test)}  ->  {len(test_pairs)} (ruling, article) pairs")
    print(f"TOTAL calls needed (if cache empty): {total}")
    print(f"\nEstimated cost: ${total * 0.001:.3f} - ${total * 0.003:.3f}")
    print("(based on short evaluation-only prompts, no evidence text --")
    print(" similar per-call cost to build_checklist_db.py extraction calls)")
    print("\nIf this looks reasonable, run calibrate_auditor.run() to proceed.")


# ---------------------------------------------------------------------- #
# Cached evaluation of a single (ruling, article) pair
# ---------------------------------------------------------------------- #

def _pair_cache_key(pair: GoldChecklistPair) -> str:
    payload = json.dumps({
        "version": _CACHE_VERSION,
        "ruling_id": pair.ruling_id,
        "article_ref": pair.article_ref,
        "conditions": pair.gold_conditions,
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _evaluate_pair(pair: GoldChecklistPair, article_text: str) -> dict | None:
    """
    Calls the evaluation-only prompt with gold conditions as the FIXED
    checklist (no evidence block -- this isolates pure "case facts +
    article text -> judge these conditions" reasoning, deliberately
    excluding retrieval evidence per the calibration design above).
    Returns None on API/parse failure (caller skips this pair).
    """
    conditions_block = "\n".join(f"- {c}" for c in pair.gold_conditions)
    prompt = _EVALUATION_ONLY_PROMPT_TEMPLATE.format(
        query=pair.query,
        article_ref=pair.article_ref,
        article_text=_truncate(article_text or "(متن ماده در دسترس نیست)", _MAX_ARTICLE_TEXT_CHARS),
        evidence_block="(این ارزیابی بدون شواهد رأی مرتبط انجام می‌شود -- فقط بر اساس واقعیت‌های پرونده و متن ماده قضاوت کن)",
        conditions_block=conditions_block,
    )
    try:
        raw_text = get_client().complete(
            model=AUDITOR_MODEL.model,
            temperature=AUDITOR_MODEL.temperature,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
        )
    except Exception as e:
        print(f"    [API ERROR] {pair.ruling_id}/{pair.article_ref}: {e}")
        return None

    raw_text = raw_text.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(raw_text)
        items = parsed.get("checklist")
        if not isinstance(items, list) or len(items) != len(pair.gold_conditions):
            raise ValueError(f"expected {len(pair.gold_conditions)} items, got {items!r}")
        predicted_satisfied = [bool(it["satisfied"]) for it in items]
        topically_relevant = bool(parsed.get("topically_relevant", True))
        return {
            "predicted_satisfied": predicted_satisfied,
            "topically_relevant": topically_relevant,
        }
    except Exception as e:
        print(f"    [PARSE ERROR] {pair.ruling_id}/{pair.article_ref}: {e} -- raw: {raw_text[:200]!r}")
        return None


def _evaluate_all_pairs(pairs: list[GoldChecklistPair], label: str) -> list[dict]:
    """Evaluates every pair with per-pair disk caching. Returns a list of
    result rows ready for metric computation."""
    article_texts = _fetch_article_texts(list({p.article_ref for p in pairs}))

    results = []
    cache_hits, cache_misses = 0, 0

    print(f"\n--- evaluating {len(pairs)} pairs ({label}) ---")
    for i, pair in enumerate(pairs, start=1):
        key = _pair_cache_key(pair)
        cache_path = _CACHE_DIR / f"{key}.pkl"

        if cache_path.exists():
            cache_hits += 1
            with cache_path.open("rb") as f:
                evaluation = pickle.load(f)
        else:
            cache_misses += 1
            print(f"  [{i}/{len(pairs)}] {pair.ruling_id} / {pair.article_ref} ...",
                  end=" ", flush=True)
            t0 = time.monotonic()
            evaluation = _evaluate_pair(pair, article_texts.get(pair.article_ref, ""))
            if evaluation is None:
                print("skipped")
                continue
            with cache_path.open("wb") as f:
                pickle.dump(evaluation, f)
            print(f"ok ({time.monotonic()-t0:.1f}s)")

        n_conditions = len(pair.gold_conditions)
        predicted_satisfied = evaluation["predicted_satisfied"]
        item_correct = [
            pred == gold for pred, gold in zip(predicted_satisfied, pair.gold_satisfied)
        ]
        predicted_is_applicable = all(predicted_satisfied)
        raw_confidence = sum(predicted_satisfied) / n_conditions if n_conditions else 0.0

        results.append({
            "ruling_id": pair.ruling_id,
            "article_ref": pair.article_ref,
            "gold_is_applicable": pair.gold_is_applicable,
            "predicted_is_applicable": predicted_is_applicable,
            "is_applicable_correct": predicted_is_applicable == pair.gold_is_applicable,
            "raw_auditor_confidence": raw_confidence,
            "item_level_accuracy": sum(item_correct) / len(item_correct) if item_correct else None,
            "topically_relevant_predicted": evaluation["topically_relevant"],
        })

    print(f"cache: {cache_hits} hits / {cache_misses} misses "
          f"(only misses cost API calls)")
    return results


# ---------------------------------------------------------------------- #
# Metrics
# ---------------------------------------------------------------------- #

def _prf_binary(y_true: list[bool], y_pred: list[bool]) -> tuple[float, float, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _expected_calibration_error(confidences: list[float], correct: list[bool], n_bins: int = 10) -> float:
    """
    Standard ECE: bins predictions by confidence, compares average
    confidence to actual accuracy in each bin, weighted by bin size.
    Lower is better (0 = perfectly calibrated).
    """
    confidences = np.array(confidences)
    correct = np.array(correct, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def _report_metrics(results: list[dict], label: str) -> dict:
    y_true = [r["gold_is_applicable"] for r in results]
    y_pred = [r["predicted_is_applicable"] for r in results]
    confidences = [r["raw_auditor_confidence"] for r in results]
    correct = [r["is_applicable_correct"] for r in results]

    precision, recall, f1 = _prf_binary(y_true, y_pred)
    ece = _expected_calibration_error(confidences, correct)

    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    auroc = roc_auc_score(y_true, confidences) if n_pos and n_neg else None

    print(f"\n=== {label} metrics ({len(results)} article rows) ===")
    print(f"  is_applicable  precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}")
    print(f"  overall accuracy: {sum(correct)/len(correct):.3f}")
    print(f"  ECE: {ece:.3f}  (lower is better, 0 = perfectly calibrated)")
    print(f"  AUROC: {auroc:.3f}" if auroc is not None else "  AUROC: N/A (single class)")
    print(f"  gold_is_applicable distribution: {n_pos} true / {n_neg} false")

    return {
        "label": label,
        "n": len(results),
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": sum(correct) / len(correct),
        "ece": ece,
        "auroc": auroc,
        "n_pos": n_pos, "n_neg": n_neg,
    }


# ---------------------------------------------------------------------- #
# Confidence calibration: Platt scaling (1D logistic regression)
# ---------------------------------------------------------------------- #

def _fit_confidence_calibration(train_results: list[dict]) -> LogisticRegression:
    """
    Platt scaling: fits P(is_applicable_correct=True | raw_confidence)
    with a single-feature logistic regression. Only 2 free parameters
    (slope + intercept) -- appropriate for ~150-200 training rows,
    avoids the overfitting risk of a richer feature set at this sample
    size. This produces a CALIBRATED confidence that downstream fusion
    with the agent layer should use instead of the raw ratio.
    """
    X = np.array([[r["raw_auditor_confidence"]] for r in train_results])
    y = np.array([r["is_applicable_correct"] for r in train_results])

    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = []
    for train_idx, val_idx in skf.split(X, y):
        if len(set(y[train_idx])) < 2 or len(set(y[val_idx])) < 2:
            continue
        model.fit(X[train_idx], y[train_idx])
        probs = model.predict_proba(X[val_idx])[:, 1]
        cv_aucs.append(roc_auc_score(y[val_idx], probs))

    if cv_aucs:
        print(f"\nPlatt scaling CV AUROC: {np.mean(cv_aucs):.3f} +/- {np.std(cv_aucs):.3f}")

    model.fit(X, y)
    print(f"Learned calibration: slope={model.coef_[0][0]:+.3f}  intercept={model.intercept_[0]:+.3f}")
    return model


# ---------------------------------------------------------------------- #
# Main entry point
# ---------------------------------------------------------------------- #

def run(sample_size: int | None = None) -> None:
    samples = load_case_annotations(_ANNOTATION_PATH)
    if sample_size:
        samples = samples[:sample_size]
    train_val, test = stratified_test_split(samples, test_fraction=0.2)

    train_pairs = _collect_pairs(train_val)
    test_pairs = _collect_pairs(test)

    train_results = _evaluate_all_pairs(train_pairs, "train_val")
    test_results = _evaluate_all_pairs(test_pairs, "test")

    train_metrics = _report_metrics(train_results, "TRAIN_VAL")
    test_metrics = _report_metrics(test_results, "TEST")

    calibration_model = _fit_confidence_calibration(train_results)

    # apply calibration to test set for a calibrated-confidence report
    test_conf_calibrated = calibration_model.predict_proba(
        np.array([[r["raw_auditor_confidence"]] for r in test_results])
    )[:, 1]
    test_ece_calibrated = _expected_calibration_error(
        list(test_conf_calibrated), [r["is_applicable_correct"] for r in test_results]
    )
    print(f"\nTest ECE with CALIBRATED confidence: {test_ece_calibrated:.3f} "
          f"(vs raw ECE: {test_metrics['ece']:.3f})")

    # ---- save everything for downstream fusion with the agent layer ----
    output = {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "calibration": {
            "slope": float(calibration_model.coef_[0][0]),
            "intercept": float(calibration_model.intercept_[0]),
            "test_ece_calibrated": test_ece_calibrated,
        },
        "per_sample": {
            "train_val": train_results,
            "test": test_results,
        },
    }
    out_path = Path(__file__).resolve().parent / "data" / "auditor_calibration_results.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    run()