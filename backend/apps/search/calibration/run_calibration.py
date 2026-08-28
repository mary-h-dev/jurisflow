from __future__ import annotations
import os
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.development")
django.setup()




import logging
import sys

from apps.search import confidence as confidence_module
from .data import load_case_annotations, stratified_test_split
from .calibrate import calibrate, _apply_config, _collect_raw_outputs
from .metrics import spearman_correlation, auroc_high_quality

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_ANNOTATION_PATH ="apps/search/calibration/data/annotations/case_grounded.json"


def main():
    all_samples = load_case_annotations(_ANNOTATION_PATH)
    logger.info(f"Loaded {len(all_samples)} annotated samples "
                f"({sum(1 for s in all_samples if s.case_type == 'حقوقی')} civil, "
                f"{sum(1 for s in all_samples if s.case_type == 'کیفری')} criminal)")
    train_val, test = stratified_test_split(all_samples, test_fraction=0.2)
    # train_val = train_val[:3]  
    logger.info(f"train_val={len(train_val)}, test={len(test)} "
                f"(test held out, not touched until final evaluation)")

    best_config, _ = calibrate(train_val, k=5)

    # ---- Final, one-time evaluation on held-out test set ----
    _apply_config(best_config)
    test_raw = _collect_raw_outputs(test)

    confidences, recalls, f1s = [], [], []
    for sample in test:
        raw = test_raw.get(sample.ruling_id)
        if raw is None:
            continue
        conf_result = confidence_module.compute_confidence(
            raw.vector,
            queried_channels=raw.queried_channels,
            routing_confidence=raw.routing_confidence,
            ambiguity_flag=raw.ambiguity_flag,
        )
        confidences.append(conf_result.score)
        recalls.append(raw.actual_recall)
        f1s.append(raw.actual_f1)

    test_spearman_recall = spearman_correlation(confidences, recalls)
    test_auroc_recall = auroc_high_quality(confidences, recalls)
    test_spearman_f1 = spearman_correlation(confidences, f1s)
    test_auroc_f1 = auroc_high_quality(confidences, f1s)

    print("\n=== FINAL TEST RESULTS (retrieval-confidence calibration only) ===")
    print(f"Best config: {best_config}")
    print(f"Test Spearman (vs recall): {test_spearman_recall:.3f}")
    print(f"Test AUROC    (vs recall): {test_auroc_recall:.3f}" if test_auroc_recall is not None else "Test AUROC (vs recall): N/A")
    print(f"Test Spearman (vs f1):     {test_spearman_f1:.3f}")
    print(f"Test AUROC    (vs f1):     {test_auroc_f1:.3f}" if test_auroc_f1 is not None else "Test AUROC (vs f1): N/A")

if __name__ == "__main__":
    main()