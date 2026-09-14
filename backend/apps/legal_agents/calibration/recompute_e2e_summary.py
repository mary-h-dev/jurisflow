"""
Recomputes the aggregate E2E [TEST] summary metrics (AUROC, ECE,
Accuracy, etc.) from the already-updated e2e_test.json, after the 5
previously-failed agent samples were retried and patched in.

This does NOT call any API -- it just re-runs the same aggregation
logic from calibrate_e2e.py on the current per_sample data, so
Table 1 numbers match Table 2 (both computed from the same,
post-retry snapshot of e2e_test.json).

Run:
DJANGO_SETTINGS_MODULE=config.settings.development python -c "
import django; django.setup()
from apps.legal_agents.calibration.recompute_e2e_summary import run
run()
"
"""

from __future__ import annotations

import json
from pathlib import Path

from apps.legal_agents.calibration.calibrate_e2e import _aggregate, _print_metrics

_TEST_PATH = Path("apps/legal_agents/calibration/data/e2e_test.json")


def run():
    data = json.loads(_TEST_PATH.read_text(encoding="utf-8"))
    rows = data["per_sample"]

    metrics = _aggregate(rows, "test")
    _print_metrics(metrics)

    # overwrite the stored "metrics" block with the recomputed, consistent version
    data["metrics"] = metrics
    _TEST_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nUpdated metrics block saved back to {_TEST_PATH}")


if __name__ == "__main__":
    run()