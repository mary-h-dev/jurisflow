"""
Final calibration report: which samples pass/fail, confusion matrix vs
the 0.7 recall threshold, and per-sample detail for failures.
Uses the cached raw outputs from the final "no routing" calibration run
-- no new API calls.

Run:
DJANGO_SETTINGS_MODULE=config.settings.development python -c "import django; django.setup(); from apps.search.calibration.final_calibration_report import run; run()"
"""

from __future__ import annotations

from apps.search.calibration.data import load_case_annotations, stratified_test_split
from apps.search.calibration.cache_utils import load_or_collect
from apps.search.calibration.calibrate import _collect_raw_outputs

_ANNOTATION_PATH = "apps/search/calibration/data/annotations/case_grounded.json"
_RECALL_THRESHOLD = 0.7


def run():
    samples = load_case_annotations(_ANNOTATION_PATH)
    train_val, test = stratified_test_split(samples, test_fraction=0.2)
    all_samples = train_val + test

    train_val_raw = load_or_collect("train_val_140_no_routing", train_val, _collect_raw_outputs)
    test_raw = load_or_collect("test_140_no_routing", test, _collect_raw_outputs)
    all_raw = {**train_val_raw, **test_raw}

    passed, failed = [], []

    for sample in all_samples:
        raw = all_raw.get(sample.ruling_id)
        if raw is None:
            continue
        entry = {
            "ruling_id": sample.ruling_id,
            "case_type": sample.case_type,
            "query": sample.query[:80],
            "gold_articles": sample.gold_articles,
            "recall": raw.actual_recall,
            "f1": raw.actual_f1,
            "confidence_vector": {
                "feature_quality": raw.vector.feature_quality,
                "ruling_quality": raw.vector.ruling_quality,
                "article_quality": raw.vector.article_quality,
                "missing_channels": raw.vector.missing_channels,
            },
        }
        if raw.actual_recall >= _RECALL_THRESHOLD:
            passed.append(entry)
        else:
            failed.append(entry)

    print("=" * 70)
    print("FINAL CALIBRATION REPORT")
    print("=" * 70)
    print(f"Total samples evaluated: {len(passed) + len(failed)}")
    print(f"Passed (recall >= {_RECALL_THRESHOLD}): {len(passed)} ({100*len(passed)/(len(passed)+len(failed)):.1f}%)")
    print(f"Failed (recall <  {_RECALL_THRESHOLD}): {len(failed)} ({100*len(failed)/(len(passed)+len(failed)):.1f}%)")

    # breakdown by case_type
    for label, group in [("PASSED", passed), ("FAILED", failed)]:
        civil = sum(1 for e in group if e["case_type"] == "حقوقی")
        criminal = sum(1 for e in group if e["case_type"] == "کیفری")
        print(f"\n{label} by case_type: حقوقی={civil}, کیفری={criminal}")

    print("\n" + "=" * 70)
    print(f"FAILED SAMPLES ({len(failed)}) -- detail")
    print("=" * 70)
    for e in sorted(failed, key=lambda x: x["recall"]):
        print(f"\n[{e['ruling_id']}] recall={e['recall']:.2f} case_type={e['case_type']}")
        print(f"  query: {e['query']}...")
        print(f"  gold: {e['gold_articles']}")
        print(f"  channel quality: feature={e['confidence_vector']['feature_quality']:.2f} "
              f"ruling={e['confidence_vector']['ruling_quality']:.2f} "
              f"article={e['confidence_vector']['article_quality']:.2f}")

    # save full detail to JSON for later use
    import json
    with open("apps/search/calibration/final_report.json", "w", encoding="utf-8") as f:
        json.dump({"passed": passed, "failed": failed}, f, ensure_ascii=False, indent=2)
    print(f"\nFull report saved to apps/search/calibration/final_report.json")


if __name__ == "__main__":
    run()