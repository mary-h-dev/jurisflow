"""
Conformal prediction for JurisFlow's full pipeline (our method).

Uses final_confidence from the end-to-end pipeline as the non-conformity
signal, instead of the raw retrieval score used in the baseline.

Key differences from the baseline (calibrate_conformal.py):
  - Non-conformity score: 1 - final_confidence  (not 1 - retrieval_score)
  - Candidate pool: agreed_articles from fusion  (not top-20 raw retrieval)

Both "any" and "all" coverage modes are reported.

No API calls — uses cached e2e_calibration.json and e2e_test.json.

Run:
    DJANGO_SETTINGS_MODULE=config.settings.development python -c "
    import django; django.setup()
    from apps.legal_agents.calibration.calibrate_conformal_e2e import run; run()"
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

import numpy as np

_DATA_DIR      = Path(__file__).parent / "data"
_CALIB_PATH    = _DATA_DIR / "e2e_calibration.json"
_TEST_PATH     = _DATA_DIR / "e2e_test.json"
_RESULTS_PATH  = _DATA_DIR / "conformal_e2e_results.json"

CoverMode = Literal["any", "all"]

_ALPHAS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]


# ---------------------------------------------------------------------------
# Non-conformity score
# ---------------------------------------------------------------------------

def nonconformity(final_confidence: float) -> float:
    """1 - final_confidence. Lower = more confident."""
    return 1.0 - final_confidence


# ---------------------------------------------------------------------------
# Coverage check
# ---------------------------------------------------------------------------

def is_covered(gold: set[str], agreed: set[str], mode: CoverMode) -> bool:
    if not gold:
        return True
    return gold.issubset(agreed) if mode == "all" else bool(gold & agreed)


# ---------------------------------------------------------------------------
# Calibration score per sample
# ---------------------------------------------------------------------------

def calibration_score(gold: set[str], agreed: set[str],
                       conf: float, mode: CoverMode) -> float:
    """
    If gold is covered by agreed_articles → nonconformity(conf)
    If gold is NOT in agreed_articles    → inf  (same as baseline)
    """
    if is_covered(gold, agreed, mode):
        return nonconformity(conf)
    return math.inf


# ---------------------------------------------------------------------------
# Quantile (identical to baseline)
# ---------------------------------------------------------------------------

def compute_quantile(scores: list[float], alpha: float) -> float:
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return math.inf
    return sorted(scores)[k - 1]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(test_samples: list[dict], q_hat: float,
             mode: CoverMode, alpha: float) -> dict:
    covered  = []
    set_sizes = []

    for s in test_samples:
        gold   = set(s.get("gold_articles", []))
        agreed = set(s.get("agreed_articles", []))
        conf   = s.get("final_confidence", 0.5)

        # Prediction set: agreed_articles where nonconformity <= q_hat
        if q_hat == math.inf:
            # Entire agreed pool is the prediction set
            pred = agreed
        else:
            pred = {a for a in agreed if nonconformity(conf) <= q_hat}

        covered.append(is_covered(gold, pred, mode))
        set_sizes.append(len(pred))

    n    = len(test_samples)
    ecr  = sum(covered) / n * 100
    apss = sum(set_sizes) / n

    return {
        "mode":    mode,
        "alpha":   alpha,
        "q_hat":   round(q_hat, 4) if q_hat != math.inf else "inf",
        "ecr":     round(ecr, 2),
        "apss":    round(apss, 2),
        "n_covered": sum(covered),
        "n_test":  n,
    }


# ---------------------------------------------------------------------------
# ECE / AUROC on final_confidence
# ---------------------------------------------------------------------------

def scoring_metrics(test_samples: list[dict], mode: CoverMode) -> dict:
    confs   = [s.get("final_confidence", 0.5) for s in test_samples]
    labels  = [
        int(is_covered(
            set(s.get("gold_articles", [])),
            set(s.get("agreed_articles", [])),
            mode,
        ))
        for s in test_samples
    ]

    # ECE
    confs_arr  = np.array(confs)
    labels_arr = np.array(labels, dtype=float)
    bins = np.linspace(0.0, 1.0, 6)
    ece  = 0.0
    n    = len(confs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs_arr >= lo) & (confs_arr < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(labels_arr[mask].mean() - confs_arr[mask].mean())

    # AUROC
    from sklearn.metrics import roc_auc_score
    auroc = (round(roc_auc_score(labels, confs), 4)
             if len(set(labels)) > 1 else None)

    return {"auroc": auroc, "ece": round(float(ece), 4),
            "accuracy": round(float(np.mean(labels)), 4)}


# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------

def _print_table(results: list[dict], mode: str) -> None:
    print(f"\n  mode={mode}")
    print(f"  {'alpha':>6}  {'q_hat':>8}  {'ECR':>7}  {'APSS':>6}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*7}  {'-'*6}")
    for r in results:
        q = f"{r['q_hat']:.4f}" if r['q_hat'] != "inf" else "    inf"
        print(f"  {r['alpha']:>6.2f}  {q:>8}  {r['ecr']:>6.1f}%  {r['apss']:>6.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    print("=" * 60)
    print("CONFORMAL PREDICTION — OUR METHOD (final_confidence)")
    print("=" * 60)
    print("Non-conformity : 1 - final_confidence")
    print("Candidate pool : agreed_articles (cited by >=2 agents)")
    print("No API calls   — using cached e2e results\n")

    calib_data = json.loads(_CALIB_PATH.read_text(encoding="utf-8"))
    test_data  = json.loads(_TEST_PATH.read_text(encoding="utf-8"))

    calib_samples = calib_data["per_sample"]
    test_samples  = test_data["per_sample"]

    # Filter out errored samples
    calib_samples = [s for s in calib_samples if "error" not in s]
    test_samples  = [s for s in test_samples  if "error" not in s]

    print(f"Calibration samples : {len(calib_samples)}")
    print(f"Test samples        : {len(test_samples)}")

    all_results: dict = {"any": [], "all": []}

    for mode in ("any", "all"):
        # Calibration scores
        scores = [
            calibration_score(
                gold   = set(s.get("gold_articles", [])),
                agreed = set(s.get("agreed_articles", [])),
                conf   = s.get("final_confidence", 0.5),
                mode   = mode,
            )
            for s in calib_samples
        ]

        n_inf = sum(1 for s in scores if s == math.inf)
        print(f"\n[{mode}] calibration: {len(scores)} scores  "
              f"inf={n_inf} ({n_inf/len(scores):.0%} gold not in agreed)")

        for alpha in _ALPHAS:
            q_hat  = compute_quantile(scores, alpha)
            result = evaluate(test_samples, q_hat, mode, alpha)
            all_results[mode].append(result)

        _print_table(all_results[mode], mode)

        # Scoring function metrics
        sf = scoring_metrics(test_samples, mode)
        print(f"\n  Scoring function [{mode}]:")
        print(f"    AUROC    : {sf['auroc']}")
        print(f"    ECE      : {sf['ece']}")
        print(f"    Accuracy : {sf['accuracy']}")

    # Save base output (will be extended below)
    output = {
        "method":         "e2e_final_confidence",
        "nonconformity":  "1 - final_confidence",
        "candidate_pool": "agreed_articles (cited by >=2 agents)",
        "n_calibration":  len(calib_samples),
        "n_test":         len(test_samples),
        "results":        all_results,
    }

    # ---------------------------------------------------------------------------
    # ECR-matched comparison
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("VALID COMPARISON: ECR-matched (not alpha-matched)")
    print("=" * 60)

    # Baseline results at all alphas (from calibrate_conformal.py output)
    baseline = {
        0.10: {"ecr": 71.4, "apss": 20.0},
        0.15: {"ecr": 71.4, "apss": 20.0},
        0.20: {"ecr": 71.4, "apss": 20.0},
        0.25: {"ecr": 71.4, "apss": 20.0},
        0.30: {"ecr": 71.4, "apss": 20.0},
        0.40: {"ecr": 64.3, "apss": 7.71},
        0.50: {"ecr": 57.1, "apss": 2.50},
        0.60: {"ecr": 53.6, "apss": 1.39},
    }

    # Our method results
    ours_any = {r["alpha"]: r for r in all_results["any"]}

    print(f"\n  {'ECR target':>12}  {'Baseline APSS':>14}  {'Ours APSS':>10}  {'Coverage OK':>12}")
    print(f"  {'-'*12}  {'-'*14}  {'-'*10}  {'-'*12}")

    matched_rows = []
    for alpha, base in sorted(baseline.items()):
        ours = ours_any.get(alpha, {})
        base_ecr  = base["ecr"]
        ours_ecr  = ours.get("ecr", 0)
        base_apss = base["apss"]
        ours_apss = ours.get("apss", float("inf"))

        # Coverage OK = ours ECR >= baseline ECR (guarantee maintained)
        coverage_ok = ours_ecr >= base_ecr
        ok_str = "✓" if coverage_ok else f"✗ (ours={ours_ecr:.1f}%)"

        print(f"  α={alpha:.2f}  base_ECR={base_ecr:.1f}%  "
              f"base_APSS={base_apss:>6.2f}  "
              f"ours_APSS={ours_apss:>5.2f}  {ok_str}")

        matched_rows.append({
            "alpha": alpha,
            "baseline_ecr": base_ecr, "baseline_apss": base_apss,
            "ours_ecr": ours_ecr,     "ours_apss": ours_apss,
            "coverage_maintained": coverage_ok,
        })

    # Find the best valid comparison point
    valid = [r for r in matched_rows if r["coverage_maintained"]
             and r["ours_apss"] != float("inf")]
    if valid:
        best = min(valid, key=lambda r: r["ours_apss"] / max(r["baseline_apss"], 0.01))
        print(f"\n  Best valid comparison point: α={best['alpha']}")
        print(f"    Baseline: ECR={best['baseline_ecr']:.1f}%  APSS={best['baseline_apss']:.2f}")
        print(f"    Ours    : ECR={best['ours_ecr']:.1f}%   APSS={best['ours_apss']:.2f}")
        if best["baseline_apss"] > 0:
            reduction = (1 - best["ours_apss"] / best["baseline_apss"]) * 100
            print(f"    APSS reduction: {reduction:.1f}%  (coverage maintained ✓)")
    else:
        print("\n  ⚠ No alpha where our ECR >= baseline ECR.")
        print("  => The 32% inf rate in calibration limits our maximum ECR to ~68%.")
        print("  => Report upper-bound ECR gap as a Limitation.")

    # Full trade-off curve data (for the ECR-vs-APSS plot)
    print("\n  Full trade-off curve (for ECR-vs-APSS figure):")
    print(f"  {'alpha':>6}  {'Baseline ECR':>13}  {'Baseline APSS':>14}  "
          f"{'Ours ECR':>9}  {'Ours APSS':>10}")
    for alpha in sorted(baseline.keys()):
        b    = baseline[alpha]
        o    = ours_any.get(alpha, {})
        o_ecr  = o.get("ecr", 0)
        o_apss = o.get("apss", float("inf"))
        print(f"  {alpha:>6.2f}  {b['ecr']:>12.1f}%  {b['apss']:>14.2f}  "
              f"{o_ecr:>8.1f}%  {o_apss:>10.2f}")

    output["ecr_matched_comparison"] = matched_rows
    output["best_valid_comparison"]  = best if valid else None
    _RESULTS_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved → {_RESULTS_PATH}")