"""
Continuous regression (not classification) calibration — predicts the
actual recall value directly, instead of a binary high/low label.
Avoids the class-imbalance failure seen with logistic regression when
nearly all samples score recall >= 0.7 (87/88 "high" in one run),
which made AUROC/CV undefined regardless of threshold choice.

Run:
DJANGO_SETTINGS_MODULE=config.settings.development python -c "import django; django.setup(); from apps.search.calibration.calibrate_linear_regression import run; run()"
"
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from apps.search.calibration.cache_utils import load_or_collect
from apps.search.calibration.data import load_case_annotations, stratified_test_split
from apps.search.calibration.calibrate import _collect_raw_outputs

_ANNOTATION_PATH = "apps/search/calibration/data/annotations/case_grounded.json"


def _features_from_raw(raw) -> list[float]:
    vec = raw.vector
    return [
        vec.feature_quality,
        vec.ruling_quality,
        vec.article_quality,
    ]


def run():
    samples = load_case_annotations(_ANNOTATION_PATH)
    train_val, test = stratified_test_split(samples, test_fraction=0.2)

    print(f"Collecting raw outputs: {len(train_val)} train_val, {len(test)} test...")

    # train_val_raw = _collect_raw_outputs(train_val)
    # test_raw = _collect_raw_outputs(test)
    

    train_val_raw = load_or_collect("train_val_120_v2", train_val, _collect_raw_outputs)
    test_raw = load_or_collect("test_120_v2", test, _collect_raw_outputs)

    X_train, y_train = [], []
    for sample in train_val:
        raw = train_val_raw.get(sample.ruling_id)
        if raw is None:
            continue
        X_train.append(_features_from_raw(raw))
        y_train.append(raw.actual_recall)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print(f"Train recall distribution: mean={y_train.mean():.3f} std={y_train.std():.3f} "
          f"min={y_train.min():.3f} max={y_train.max():.3f}")

    # Ridge regression (L2-regularized linear regression) — appropriate
    # given the small sample size, avoids overfitting the 3 coefficients.
    model = Ridge(alpha=1.0)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2s, cv_spearmans = [], []
    for train_idx, val_idx in kf.split(X_train):
        model.fit(X_train[train_idx], y_train[train_idx])
        preds = model.predict(X_train[val_idx])
        cv_r2s.append(r2_score(y_train[val_idx], preds))
        corr, _ = spearmanr(preds, y_train[val_idx])
        cv_spearmans.append(0.0 if corr != corr else corr)

    print(f"CV R²:       {np.mean(cv_r2s):.3f} +/- {np.std(cv_r2s):.3f}")
    print(f"CV Spearman: {np.mean(cv_spearmans):.3f} +/- {np.std(cv_spearmans):.3f}")

    model.fit(X_train, y_train)
    print(f"\nLearned coefficients:")
    feature_names = ["feature_quality", "ruling_quality", "article_quality"]
    for name, coef in zip(feature_names, model.coef_):
        print(f"  {name}: {coef:+.3f}")
    print(f"  intercept: {model.intercept_:+.3f}")

    # ---- Final test evaluation ----
    X_test, y_test = [], []
    for sample in test:
        raw = test_raw.get(sample.ruling_id)
        if raw is None:
            continue
        X_test.append(_features_from_raw(raw))
        y_test.append(raw.actual_recall)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    test_preds = model.predict(X_test)
    test_r2 = r2_score(y_test, test_preds)
    test_spearman, _ = spearmanr(test_preds, y_test)
    test_spearman = 0.0 if test_spearman != test_spearman else test_spearman

    print(f"\n=== FINAL TEST RESULTS (continuous regression) ===")
    print(f"Test R²:       {test_r2:.3f}")
    print(f"Test Spearman: {test_spearman:.3f}")
    print(f"\nTest recall distribution: mean={y_test.mean():.3f} std={y_test.std():.3f}")


if __name__ == "__main__":
    run()