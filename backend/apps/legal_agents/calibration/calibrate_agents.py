"""
Agent-only calibration: evaluates the three deliberation agents in isolation
from retrieval and Auditor. Uses gold_checklist_by_article as fixed input
(bypasses real Auditor output) -- mirrors calibrate_auditor.py design.

Why this isolates the right thing:
    Removes retrieval noise and Auditor errors. What remains is a clean
    measurement of: "given correct articles and conditions, do agents
    reach the right verdict with well-calibrated confidence?"

Run (dry-run first):
    DJANGO_SETTINGS_MODULE=config.settings.development python -c "
    import django; django.setup()
    from apps.legal_agents.calibration.calibrate_agents import dry_run; dry_run()"

Run (train_val — needed for fusion calibration):
DJANGO_SETTINGS_MODULE=config.settings.development python -c "import django; django.setup(); from apps.legal_agents.calibration.calibrate_agents import run; run('train_val')"

Run (test — final evaluation):
    DJANGO_SETTINGS_MODULE=config.settings.development python -c "
    import django; django.setup()
    from apps.legal_agents.calibration.calibrate_agents import run; run('test')"

Cost estimate:
    train_val (~112 samples) x 3 agents = ~336 calls  ≈ $0.50
    test      (~28  samples) x 3 agents = ~84  calls  ≈ $0.15
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from apps.search.calibration.data import load_case_annotations, stratified_test_split
from apps.legal_agents.services import deliberation_service
from apps.legal_agents.schemas import (
    AuditedArticleOut,
    AuditorOut,
    ChecklistItemOut,
    CombinedConfidenceOut,
)

_ANNOTATION_PATH = "apps/search/calibration/data/annotations/case_grounded.json"
_RESULTS_DIR     = Path(__file__).parent / "data"

# Neutral prior — isolates agent signal from Auditor signal
_NEUTRAL_PRIOR = CombinedConfidenceOut(
    score           = 0.70,
    level           = "medium",
    retrieval_score = 0.70,
    auditor_score   = 0.70,
    prune_ratio     = 0.0,
)


# ---------------------------------------------------------------------------
# Build mock AuditorOut from gold annotation
# ---------------------------------------------------------------------------

def _build_auditor_out(sample) -> AuditorOut:
    """
    gold_articles     → is_applicable=True,  topically_relevant=True
    excluded_articles → is_applicable=False, topically_relevant=True
    """
    gold_set      = set(sample.gold_articles)
    excluded_refs = set()

    for ex in (getattr(sample, "excluded_articles", None) or []):
        if isinstance(ex, dict):
            ref = f"{ex.get('law_name', '')} - ماده {ex.get('article_number', '')}"
        else:
            ref = str(ex)
        excluded_refs.add(ref)

    gold_checklist = getattr(sample, "gold_checklist_by_article", {}) or {}
    verified: list[AuditedArticleOut] = []

    for ref in gold_set:
        raw_items = gold_checklist.get(ref, [])
        checklist = [
            ChecklistItemOut(
                condition = it.get("condition", "") if isinstance(it, dict) else str(it),
                necessary = it.get("necessary", True) if isinstance(it, dict) else True,
                satisfied = it.get("satisfied", True) if isinstance(it, dict) else True,
            )
            for it in raw_items
        ]
        verified.append(AuditedArticleOut(
            article_ref        = ref,
            checklist          = checklist,
            is_applicable      = True,
            topically_relevant = True,
            auditor_confidence = 1.0,
        ))

    for ref in excluded_refs:
        raw_items = gold_checklist.get(ref, [])
        checklist = [
            ChecklistItemOut(
                condition = it.get("condition", "") if isinstance(it, dict) else str(it),
                necessary = it.get("necessary", True) if isinstance(it, dict) else True,
                satisfied = it.get("satisfied", False) if isinstance(it, dict) else False,
            )
            for it in raw_items
        ]
        verified.append(AuditedArticleOut(
            article_ref        = ref,
            checklist          = checklist,
            is_applicable      = False,
            topically_relevant = True,
            auditor_confidence = 0.5,
        ))

    return AuditorOut(
        query             = sample.query,
        verified_articles = verified,
        pruned_articles   = [],
        confidence        = _NEUTRAL_PRIOR,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _correctness(sample, fusion) -> int:
    """1 if any gold article is in agreed_articles (cited by ≥2 agents)."""
    if fusion is None:
        return 0
    return int(bool(set(sample.gold_articles) & set(fusion.agreed_articles)))


def _article_prf(sample, fusion) -> tuple[float, float, float]:
    """Precision/Recall/F1 at article level (predicted = all_cited_articles)."""
    if fusion is None:
        return 0.0, 0.0, 0.0
    gold      = set(sample.gold_articles)
    predicted = set(fusion.all_cited_articles)
    if not predicted:
        return 0.0, 0.0, 0.0
    tp   = len(gold & predicted)
    prec = tp / len(predicted)
    rec  = tp / len(gold) if gold else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return round(prec, 4), round(rec, 4), round(f1, 4)


def _ece(confidences: list[float], labels: list[int], n_bins: int = 5) -> float:
    confs  = np.array(confidences)
    labels = np.array(labels, dtype=float)
    bins   = np.linspace(0.0, 1.0, n_bins + 1)
    ece    = 0.0
    n      = len(confs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(labels[mask].mean() - confs[mask].mean())
    return round(float(ece), 4)


def _report(rows: list[dict], label: str) -> dict:
    confs      = [r["final_confidence"] for r in rows]
    corrects   = [r["correct"] for r in rows]
    precs      = [r["precision"] for r in rows]
    recs       = [r["recall"] for r in rows]
    f1s        = [r["f1"] for r in rows]
    consensuses = [r["consensus"] for r in rows]

    auroc = (
        round(roc_auc_score(corrects, confs), 4)
        if len(set(corrects)) > 1 else None
    )
    ece_val        = _ece(confs, corrects)
    accuracy       = round(sum(corrects) / len(corrects), 4)
    mean_prec      = round(float(np.mean(precs)), 4)
    mean_rec       = round(float(np.mean(recs)), 4)
    mean_f1        = round(float(np.mean(f1s)), 4)
    consensus_rate = round(
        sum(1 for c in consensuses if c in ("full", "majority")) / len(consensuses), 4
    )
    full_rate = round(sum(1 for c in consensuses if c == "full") / len(consensuses), 4)

    print(f"\n=== {label} metrics ({len(rows)} samples) ===")
    print(f"  AUROC          : {auroc}")
    print(f"  ECE            : {ece_val}  (lower is better)")
    print(f"  Accuracy       : {accuracy}")
    print(f"  Mean Precision : {mean_prec}")
    print(f"  Mean Recall    : {mean_rec}")
    print(f"  Mean F1        : {mean_f1}")
    print(f"  Consensus rate : {consensus_rate}  (full+majority / n)")
    print(f"  Full consensus : {full_rate}")

    return {
        "label": label, "n": len(rows),
        "auroc": auroc, "ece": ece_val, "accuracy": accuracy,
        "mean_precision": mean_prec, "mean_recall": mean_rec, "mean_f1": mean_f1,
        "consensus_rate": consensus_rate, "full_consensus_rate": full_rate,
    }


# ---------------------------------------------------------------------------
# Platt scaling on agent confidence (mirrors calibrate_auditor.py)
# ---------------------------------------------------------------------------

def _fit_platt(rows: list[dict]) -> LogisticRegression:
    X = np.array([[r["final_confidence"]] for r in rows])
    y = np.array([r["correct"] for r in rows])

    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = []
    for tr, va in skf.split(X, y):
        if len(set(y[tr])) < 2 or len(set(y[va])) < 2:
            continue
        model.fit(X[tr], y[tr])
        probs = model.predict_proba(X[va])[:, 1]
        cv_aucs.append(roc_auc_score(y[va], probs))

    if cv_aucs:
        print(f"\nPlatt scaling CV AUROC: {np.mean(cv_aucs):.3f} +/- {np.std(cv_aucs):.3f}")
    model.fit(X, y)
    print(f"Calibration: slope={model.coef_[0][0]:+.3f}  intercept={model.intercept_[0]:+.3f}")
    return model


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def dry_run() -> None:
    samples         = load_case_annotations(_ANNOTATION_PATH)
    train_val, test = stratified_test_split(samples, test_fraction=0.2)
    print("=" * 60)
    print("DRY RUN — no API calls made")
    print("=" * 60)
    print(f"train_val : {len(train_val)} samples  →  {len(train_val) * 3} LLM calls  ≈ ${len(train_val) * 3 * 0.002:.2f}")
    print(f"test      : {len(test)} samples  →  {len(test) * 3} LLM calls  ≈ ${len(test) * 3 * 0.002:.2f}")
    print(f"total     : {(len(train_val) + len(test)) * 3} LLM calls  ≈ ${(len(train_val) + len(test)) * 3 * 0.002:.2f}")
    print("\nIf this looks reasonable, run calibrate_agents.run('train_val') first.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(split: str = "test") -> None:
    """
    Args:
        split: "train_val" or "test"
               Always run train_val first — needed for fusion calibration.
    """
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = _RESULTS_DIR / f"agents_{split}.json"

    samples         = load_case_annotations(_ANNOTATION_PATH)
    train_val, test = stratified_test_split(samples, test_fraction=0.2)
    target          = train_val if split == "train_val" else test

    print(f"calibrate_agents [{split}]: {len(target)} samples")
    print(f"Estimated cost   : ~${len(target) * 3 * 0.002:.2f}\n")

    rows: list[dict] = []

    for i, sample in enumerate(target, 1):
        print(f"[{i:>3}/{len(target)}] ruling={sample.ruling_id} ...", end=" ", flush=True)
        t0 = time.monotonic()

        try:
            auditor_out = _build_auditor_out(sample)
            result      = deliberation_service.run(
                auditor_out, session_id=f"cal-agent-{sample.ruling_id}"
            )
            fusion   = result.fusion
            correct  = _correctness(sample, fusion)
            conf     = fusion.final_confidence if fusion else 0.5
            prec, rec, f1 = _article_prf(sample, fusion)
            consensus = fusion.consensus_level.value if fusion else "none"
            elapsed  = time.monotonic() - t0

            rows.append({
                "ruling_id":        sample.ruling_id,
                "case_type":        getattr(sample, "case_type", ""),
                "gold_articles":    sample.gold_articles,
                "final_confidence": round(conf, 4),
                "correct":          correct,
                "majority_verdict": fusion.majority_verdict.value if fusion else None,
                "consensus":        consensus,
                "agreed_articles":  fusion.agreed_articles if fusion else [],
                "all_cited":        fusion.all_cited_articles if fusion else [],
                "precision":        prec,
                "recall":           rec,
                "f1":               f1,
                "uncertainty_flag": fusion.uncertainty_flag if fusion else None,
                "verdict_spread":   fusion.verdict_spread if fusion else {},
                "elapsed_s":        round(elapsed, 1),
                "error":            result.error,
            })

            status = "✓" if correct else "✗"
            print(f"{status}  conf={conf:.2f}  consensus={consensus}  "
                  f"P={prec:.2f} R={rec:.2f}  ({elapsed:.1f}s)")

        except Exception as exc:
            elapsed = time.monotonic() - t0
            print(f"ERROR ({elapsed:.1f}s): {exc}")
            rows.append({"ruling_id": sample.ruling_id, "error": str(exc),
                         "correct": 0, "final_confidence": 0.5})

    metrics = _report(rows, split.upper())

    # Platt scaling only on train_val
    calibration = None
    if split == "train_val":
        model       = _fit_platt(rows)
        calibration = {
            "slope":     float(model.coef_[0][0]),
            "intercept": float(model.intercept_[0]),
        }

    output = {"metrics": metrics, "calibration": calibration, "per_sample": rows}
    out_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved → {out_json}")