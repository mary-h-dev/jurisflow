"""
Replaces grid search (972 combos over 80 samples — severe overfitting)
with a simple, regularized Logistic Regression over channel quality
features. Predicts P(high quality) directly from features, using far
fewer free parameters (5-6 coefficients vs 972 grid points).

Run:
    DJANGO_SETTINGS_MODULE=config.settings.development python -c "
    import django; django.setup()
    from apps.search.calibration.calibrate_regression import run
    run()
    "
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

from apps.search.calibration.data import load_case_annotations, stratified_test_split
from apps.search.calibration.calibrate import _collect_raw_outputs
from apps.search.calibration.metrics import recall_at_k_refs

_ANNOTATION_PATH = "apps/search/calibration/data/annotations/case_grounded_100.json"
_RECALL_THRESHOLD = 0.7  # "high quality" label cutoff for AUROC


def _features_from_raw(raw) -> list[float]:
    """
    5 features, matching the dimensions confidence.py already tracks:
    feature_quality, ruling_quality, article_quality, n_missing_channels,
    routing_confidence. graph_support omitted here for simplicity (often
    None) — add back once more data is collected.
    """
    vec = raw.vector
    return [
        vec.feature_quality,
        vec.ruling_quality,
        vec.article_quality,
        # float(len(vec.missing_channels)),
        raw.routing_confidence,
    ]


def run():
    samples = load_case_annotations(_ANNOTATION_PATH)
    train_val, test = stratified_test_split(samples, test_fraction=0.2)

    print(f"Collecting raw outputs: {len(train_val)} train_val, {len(test)} test...")
    train_val_raw = _collect_raw_outputs(train_val)
    test_raw = _collect_raw_outputs(test)

    # Build X, y for train_val
    X_train, y_train = [], []
    for sample in train_val:
        raw = train_val_raw.get(sample.ruling_id)
        if raw is None:
            continue
        X_train.append(_features_from_raw(raw))
        y_train.append(1 if raw.actual_recall >= _RECALL_THRESHOLD else 0)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print(f"Train label distribution: {sum(y_train)} high / {len(y_train) - sum(y_train)} low")

    # Regularized logistic regression — C small = stronger regularization,
    # appropriate given the small sample size (prevents overfitting the
    # 5 coefficients too, though far less risk than 972 grid combos).
    model = LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000)

    # 5-fold CV on train_val to sanity-check stability before final fit
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        if len(set(y_train[train_idx])) < 2:
            continue  # skip fold if only one class present
        model.fit(X_train[train_idx], y_train[train_idx])
        probs = model.predict_proba(X_train[val_idx])[:, 1]
        if len(set(y_train[val_idx])) < 2:
            continue
        cv_aucs.append(roc_auc_score(y_train[val_idx], probs))

    print(f"CV AUROC (train_val folds): {np.mean(cv_aucs):.3f} +/- {np.std(cv_aucs):.3f}")

    # Final fit on all train_val
    model.fit(X_train, y_train)
    print(f"\nLearned coefficients:")
    # feature_names = ["feature_quality", "ruling_quality", "article_quality",
    #                   "n_missing_channels", "routing_confidence"]

    feature_names = ["feature_quality", "ruling_quality", "article_quality",
                      "routing_confidence"]
    for name, coef in zip(feature_names, model.coef_[0]):
        print(f"  {name}: {coef:+.3f}")
    print(f"  intercept: {model.intercept_[0]:+.3f}")

    # ---- Final test evaluation ----
    X_test, y_test, recalls_test = [], [], []
    for sample in test:
        raw = test_raw.get(sample.ruling_id)
        if raw is None:
            continue
        X_test.append(_features_from_raw(raw))
        y_test.append(1 if raw.actual_recall >= _RECALL_THRESHOLD else 0)
        recalls_test.append(raw.actual_recall)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    test_probs = model.predict_proba(X_test)[:, 1]

    test_auroc = roc_auc_score(y_test, test_probs) if len(set(y_test)) > 1 else None
    test_spearman, _ = spearmanr(test_probs, recalls_test)

    print(f"\n=== FINAL TEST RESULTS (logistic regression, not grid search) ===")
    print(f"Test AUROC:    {test_auroc:.3f}" if test_auroc is not None else "Test AUROC: N/A (single class)")
    print(f"Test Spearman: {test_spearman:.3f}")


if __name__ == "__main__":
    run()