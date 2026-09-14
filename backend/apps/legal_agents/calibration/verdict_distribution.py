"""
Recomputes the majority-verdict accuracy breakdown (Table 3) from the
current e2e_test.json -- after the max-token fix, all 28 samples
should now have a real majority_verdict instead of the 5 that
previously had None.

No API calls -- pure aggregation over cached data.

Run:
DJANGO_SETTINGS_MODULE=config.settings.development python -c "
import django; django.setup()
from apps.legal_agents.calibration.verdict_distribution import run
run()
"
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

_TEST_PATH = Path("apps/legal_agents/calibration/data/e2e_test.json")


def run():
    data = json.loads(_TEST_PATH.read_text(encoding="utf-8"))
    samples = data["per_sample"]

    print(f"Total samples: {len(samples)}")
    no_verdict = [s for s in samples if not s.get("majority_verdict")]
    if no_verdict:
        print(f"WARNING: {len(no_verdict)} samples still have no majority_verdict:")
        for s in no_verdict:
            print(f"  ruling={s['ruling_id']}  agent_error={s.get('agent_error')}")
    print()

    by_verdict: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        v = s.get("majority_verdict") or "MISSING"
        by_verdict[v].append(s)

    print("=" * 70)
    print("Majority Verdict Distribution & Accuracy Analysis (n=%d)" % len(samples))
    print("=" * 70)

    total_any = total_all = total_n = 0
    for verdict, group in sorted(by_verdict.items(), key=lambda kv: -len(kv[1])):
        n = len(group)
        correct_any = sum(s.get("correct_any", 0) for s in group)
        correct_all = sum(s.get("correct_all", 0) for s in group)
        pct_any = 100 * correct_any / n if n else 0
        pct_all = 100 * correct_all / n if n else 0
        print(f"Verdict: {verdict:<20} | n={n:<3} | "
              f"correct_any={correct_any}/{n} ({pct_any:.0f}%) | "
              f"correct_all={correct_all}/{n} ({pct_all:.0f}%)")
        total_any += correct_any
        total_all += correct_all
        total_n += n

    print("-" * 70)
    print(f"TOTAL: n={total_n} | "
          f"correct_any={total_any}/{total_n} ({100*total_any/total_n:.1f}%) | "
          f"correct_all={total_all}/{total_n} ({100*total_all/total_n:.1f}%)")


if __name__ == "__main__":
    run()