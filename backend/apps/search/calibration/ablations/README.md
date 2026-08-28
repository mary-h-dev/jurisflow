# Ablations — rejected/superseded configurations

## pre_calibration_services.py
Original SearchService with LLM channel gating (router decided which of
feature/ruling/article to query per request). Ablation showed always
querying all three channels performs better: AUROC 0.756 vs 0.682,
Spearman 0.401 vs 0.151 (140 samples, leakage-free, test set).

## pre_calibration_confidence.py
Original confidence formula with manually-set weights (0.4/0.4/0.2) plus
routing_confidence, ambiguity_penalty, and graph_support_weight terms.
Calibration (logistic regression, 140 samples) found routing_confidence
and n_missing_channels near-zero coefficients; graph_support reduced
AUROC when added (0.756 -> 0.675). Superseded by calibrated weights in
apps/search/confidence.py.

## graph_support.py (kept in place, not moved)
PMI/NPMI-based cross-channel signal. Tested as a 5th regression feature:
reduced AUROC and destabilized CV variance. No longer called from
services.py. Kept intact for future work with more annotation data.