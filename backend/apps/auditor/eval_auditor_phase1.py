from __future__ import annotations

import json
import sys_CACHE_VERSION
from pathlib import Path

from apps.auditor.services import auditor_service
from apps.search.calibration.leakage_guard import exclude_self_ruling
from apps.search.services import search_service

_DATA_PATH = Path("apps/search/calibration/data/case_grounded.json")
_SAMPLE_SIZE = 20  # start small; Gemini calls are per-article, per-sample cost adds up


def _prf(predicted: set[str], gold: set[str]) -> tuple[float, float, float]:
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    tp = len(predicted & gold)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def run(sample_size: int = _SAMPLE_SIZE) -> None:
    samples = json.loads(_DATA_PATH.read_text(encoding="utf-8"))[:sample_size]

    raw_scores, pruned_scores = [], []
    leaked_evidence_counts = []

    for sample in samples:
        query = sample["query"]
        gold = set(sample["gold_articles"])
        ruling_id = sample["ruling_id"]

        search_result_raw = search_service.search(query)

        leaked_count = sum(
            1 for e in search_result_raw.ruling_results if e.ruling_id == ruling_id
        ) + sum(
            1 for e in search_result_raw.feature_results if e.ruling_id == ruling_id
        )
        leaked_evidence_counts.append(leaked_count)

        # Exclude the ruling this query was paraphrased FROM -- otherwise
        # the ruling_channel can trivially return the source ruling's own
        # verdict/reasoning as "supporting evidence", which is leakage,
        # not independent precedent. See apps/search/calibration/leakage_guard.py.
        search_result = exclude_self_ruling(search_result_raw, ruling_id)
        raw_refs = set(search_result.article_refs)

        audit_result = auditor_service.audit(search_result)
        kept_refs = {
            a.article_ref for a in audit_result.verified_articles if a.is_applicable
        }

        raw_scores.append(_prf(raw_refs, gold))
        pruned_scores.append(_prf(kept_refs, gold))

        print(f"[{ruling_id}] self-ruling evidence excluded={leaked_count}  "
              f"raw P/R/F1={raw_scores[-1]}  "
              f"pruned P/R/F1={pruned_scores[-1]}  "
              f"pruned_count={len(kept_refs)}/{len(raw_refs)}")

    def _avg(scores, idx):
        return sum(s[idx] for s in scores) / len(scores)

    print("\n--- averages over", len(samples), "samples ---")
    print(f"raw    precision={_avg(raw_scores,0):.3f}  recall={_avg(raw_scores,1):.3f}  f1={_avg(raw_scores,2):.3f}")
    print(f"pruned precision={_avg(pruned_scores,0):.3f}  recall={_avg(pruned_scores,1):.3f}  f1={_avg(pruned_scores,2):.3f}")
    print(f"\nsamples with >=1 self-ruling evidence piece excluded: "
          f"{sum(1 for c in leaked_evidence_counts if c > 0)}/{len(samples)}  "
          f"(avg {sum(leaked_evidence_counts)/len(samples):.1f} pieces/sample)")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else _SAMPLE_SIZE
    run(n)