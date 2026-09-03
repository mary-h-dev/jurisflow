"""
Fusion weight calibration using cached e2e_calibration.json — no API calls.

Fits optimal alpha_p (prior weight) and alpha_a (agent weight) using
logistic regression on the 25 calibration samples.

Note on sample size: n=25 is small for fitting 2 parameters. Results
should be treated as indicative rather than definitive; this limitation
is reported in the paper's Limitations section.

Run:
    DJANGO_SETTINGS_MODULE=config.settings.development python -c "
    import django; django.setup()
    from apps.legal_agents.calibration.calibrate_fusion import run; run()"
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_DATA_DIR     = Path(__file__).parent / "data"
_CALIB_PATH   = _DATA_DIR / "e2e_calibration.json"
_RESULTS_PATH = _DATA_DIR / "fusion_calibration.json"

_CURRENT_PRIOR_WEIGHT = 0.40
_CURRENT_AGENT_WEIGHT = 0.60


def _blended_confidence(row: dict, alpha_p: float, alpha_a: float) -> float:
    """Recompute final_confidence with given weights."""
    prior      = row.get("auditor_confidence", 0.5)
    agent_mean = row.get("agent_conf_mean", 0.5)
    conf_std   = row.get("agent_conf_std", 0.0) or 0.0
    consensus  = row.get("consensus", "majority")

    blended = alpha_p * prior + alpha_a * agent_mean

    # Apply split penalty (fixed — not a free parameter)
    if consensus == "split":
        blended = max(0.0, blended - 0.10)

    return round(min(1.0, blended), 4)


def run() -> None:
    print("=" * 60)
    print("FUSION WEIGHT CALIBRATION (n=25 calibration samples)")
    print("=" * 60)
    print("No API calls — using cached e2e_calibration.json\n")

    data    = json.loads(_CALIB_PATH.read_text(encoding="utf-8"))
    samples = [s for s in data["per_sample"] if "error" not in s
               and s.get("agent_conf_mean") is not None]

    print(f"Usable samples: {len(samples)}")

    # Labels: correct_any (our primary metric)
    y = np.array([s.get("correct_any", 0) for s in samples])

    print(f"Label distribution: {sum(y)} correct / {len(y)-sum(y)} incorrect\n")

    # Grid search over alpha_p (alpha_a = 1 - alpha_p)
    best_auroc = -1.0
    best_alpha_p = _CURRENT_PRIOR_WEIGHT
    results_grid = []

    for alpha_p_int in range(10, 91, 5):  # 0.10 to 0.90 step 0.05
        alpha_p = alpha_p_int / 100
        alpha_a = 1.0 - alpha_p

        confs = np.array([
            _blended_confidence(s, alpha_p, alpha_a) for s in samples
        ])

        if len(set(y)) < 2:
            auroc = None
        else:
            try:
                auroc = round(roc_auc_score(y, confs), 4)
            except Exception:
                auroc = None

        results_grid.append({
            "alpha_p": alpha_p,
            "alpha_a": alpha_a,
            "auroc":   auroc,
        })

        if auroc is not None and auroc > best_auroc:
            best_auroc   = auroc
            best_alpha_p = alpha_p

    best_alpha_a = round(1.0 - best_alpha_p, 2)

    print("Grid search results (alpha_p, alpha_a, AUROC):")
    for r in results_grid:
        marker = " ← best" if r["alpha_p"] == best_alpha_p else ""
        print(f"  alpha_p={r['alpha_p']:.2f}  alpha_a={r['alpha_a']:.2f}  "
              f"AUROC={r['auroc']}{marker}")

    print(f"\nCurrent weights  : alpha_p={_CURRENT_PRIOR_WEIGHT}  "
          f"alpha_a={_CURRENT_AGENT_WEIGHT}")
    print(f"Optimal weights  : alpha_p={best_alpha_p}  alpha_a={best_alpha_a}")
    print(f"Best AUROC       : {best_auroc}")

    # Current vs optimal confidence comparison
    current_confs = np.array([
        _blended_confidence(s, _CURRENT_PRIOR_WEIGHT, _CURRENT_AGENT_WEIGHT)
        for s in samples
    ])
    optimal_confs = np.array([
        _blended_confidence(s, best_alpha_p, best_alpha_a)
        for s in samples
    ])

    if len(set(y)) > 1:
        current_auroc = round(roc_auc_score(y, current_confs), 4)
        optimal_auroc = round(roc_auc_score(y, optimal_confs), 4)
        print(f"\nAUROC with current weights : {current_auroc}")
        print(f"AUROC with optimal weights : {optimal_auroc}")
        delta = optimal_auroc - current_auroc
        print(f"Delta                      : {delta:+.4f}")
        if abs(delta) < 0.02:
            print("=> Difference is negligible — current weights are acceptable.")
        else:
            print("=> Update fusion weights in fusion/deterministic.py")

    output = {
        "n_calibration":    len(samples),
        "optimal_alpha_p":  best_alpha_p,
        "optimal_alpha_a":  best_alpha_a,
        "optimal_auroc":    best_auroc,
        "current_alpha_p":  _CURRENT_PRIOR_WEIGHT,
        "current_alpha_a":  _CURRENT_AGENT_WEIGHT,
        "grid":             results_grid,
        "note": (
            "n=25 is small for 2-parameter fitting; treat as indicative. "
            "Full calibration on train_val is left to future work."
        ),
    }
    _RESULTS_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved → {_RESULTS_PATH}")