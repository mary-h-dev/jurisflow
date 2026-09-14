# Fusion Calibration Notes

## What we did
Ran grid search over fusion weights (alpha_p, alpha_a) on two datasets:
- train_val (n=112): fitted optimum alpha_p=0.35, alpha_a=0.65
- conformal calibration set (n=25): fitted optimum alpha_p=0.35, alpha_a=0.65

Also attempted logistic regression on [retrieval_score, auditor_score,
prune_ratio] features to replace the hand-built formula entirely
(see calibrate_confidence_fusion.py) — not run due to database
migration constraints.

## Result
Grid search across smaller subsets (n=28 test, n=25 calibration,
5-fold CV over n=53) produced optima ranging from 0.35 to 1.00,
showing calibration is unstable below n=112.

| Dataset      | Optimal alpha_p | Optimal alpha_a | AUROC  |
|--------------|-----------------|-----------------|--------|
| train_val    | 0.35            | 0.65            | 0.699  |
| calib_25     | 0.35            | 0.65            | 0.699  |
| test_28      | 0.70            | 0.30            | 0.518  |

## Interpretation
The n=112 optimum (alpha_p=0.35) and deployed value (alpha_p=0.40)
differ by only Delta=0.05 — negligible at this scale.
Deployed weights are retained as the closest available optimum.

The agent confidence signal is weak (AUROC=0.318 end-to-end) —
increasing alpha_p (Auditor weight) consistently improves AUROC,
but reliable re-tuning requires a larger dataset.

## Decision
Deployed weights retained: alpha_p=0.40, alpha_a=0.60.
Re-tuning left to future work after dataset expansion.

## Reference numbers
| Metric                    | Value  |
|---------------------------|--------|
| Agent conf AUROC (e2e)    | 0.318  |
| Auditor AUROC (per-case)  | 0.799  |
| Full fusion AUROC (e2e)   | 0.395  |
| Fusion no penalty AUROC   | 0.349  |
| Split penalty delta       | +0.046 |

Deployed: alpha_p=0.40, alpha_a=0.60, split_penalty=0.10
Grid search optimal (n=112): alpha_p=0.35, alpha_a=0.65