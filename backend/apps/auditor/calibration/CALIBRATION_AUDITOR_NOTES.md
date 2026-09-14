# Auditor Calibration Notes

## What we did
Fitted Platt scaling on top of the raw auditor confidence
(fraction of satisfied checklist items) to assess whether
post-hoc calibration improves ECE.

## Result
Platt scaling made calibration worse, not better:
- Raw ECE (test):        0.119  ← better
- Calibrated ECE (test): 0.334  ← worse

## Interpretation
The raw confidence score (fraction of satisfied necessary
conditions) is already reasonably well-calibrated on the
test set. Platt scaling was fitted on train_val but did not
generalize — likely due to the small dataset size
(n=112 train, n=28 test) and class imbalance
(25 true / 3 false on test).

## Decision
Platt scaling removed from the deployed system.
Raw fraction of satisfied conditions is used as
auditor_confidence throughout the pipeline.

## Reference numbers
| Metric        | Train_val | Test  |
|---------------|-----------|-------|
| AUROC         | 0.807     | 0.920 |
| ECE (raw)     | 0.246     | 0.119 |
| Precision     | 0.883     | 1.000 |
| Recall        | 0.872     | 0.840 |
| F1            | 0.877     | 0.913 |
| Accuracy      | 0.830     | 0.857 |

Platt: slope=+0.895, intercept=-0.614
Platt CV AUROC: 0.624 +/- 0.096