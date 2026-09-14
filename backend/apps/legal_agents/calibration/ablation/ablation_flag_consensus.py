"""
Two analyses using only cached e2e_test.json — no API calls.

1. uncertainty_flag utility:
   Does the flag reliably predict errors?
   Reports precision, recall, F1 of flag as an error predictor.

2. Consensus quality breakdown:
   Full / majority / split accuracy on both any and all coverage.

Run:
    python ablation_flag_consensus.py
    (from backend/ directory)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_TEST_PATH = Path("apps/legal_agents/calibration/data/e2e_test.json")
_OUT_PATH  = Path("apps/legal_agents/calibration/data/flag_consensus_analysis.json")


def _prf(tp, fp, fn) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def run() -> None:
    data    = json.loads(_TEST_PATH.read_text(encoding="utf-8"))
    samples = [s for s in data["per_sample"]
               if "error" not in s and s.get("consensus") is not None]

    print("=" * 60)
    print(f"ANALYSIS: n={len(samples)} test samples")
    print("=" * 60)

    # ── 1. uncertainty_flag utility ──────────────────────────────────────────
    print("\n── 1. uncertainty_flag as error predictor ──────────────────")

    for mode, label_key in [("any", "correct_any"), ("all", "correct_all")]:
        flags  = [bool(s.get("uncertainty_flag", False)) for s in samples]
        errors = [s.get(label_key, 1) == 0 for s in samples]   # 1 = error

        tp = sum(f and e for f, e in zip(flags, errors))
        fp = sum(f and not e for f, e in zip(flags, errors))
        fn = sum(not f and e for f, e in zip(flags, errors))
        tn = sum(not f and not e for f, e in zip(flags, errors))

        p, r, f1 = _prf(tp, fp, fn)
        n_flagged  = sum(flags)
        n_errors   = sum(errors)
        flag_rate  = n_flagged / len(flags)

        print(f"\n  Coverage mode: {mode}")
        print(f"  Flagged samples : {n_flagged}/{len(samples)} ({flag_rate:.0%})")
        print(f"  Actual errors   : {n_errors}/{len(samples)}")
        print(f"  Confusion: TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        print(f"  Precision : {p:.4f}  (of flagged, how many were wrong?)")
        print(f"  Recall    : {r:.4f}  (of errors, how many were flagged?)")
        print(f"  F1        : {f1:.4f}")
        if p > 0.6:
            print(f"  => Flag is a reliable warning signal ✓")
        else:
            print(f"  => Flag has limited precision as error predictor")

    # ── 2. Consensus quality breakdown ───────────────────────────────────────
    print("\n── 2. Consensus quality breakdown ──────────────────────────")

    consensus_groups: dict[str, list] = {"full": [], "majority": [], "split": []}
    for s in samples:
        c = s.get("consensus", "majority")
        if c in consensus_groups:
            consensus_groups[c].append(s)

    print(f"\n  {'Consensus':<10} {'N':>4}  {'%':>5}  "
          f"{'Acc(any)':>9}  {'Acc(all)':>9}  {'Mean conf':>10}")
    print(f"  {'-'*10}  {'-'*4}  {'-'*5}  {'-'*9}  {'-'*9}  {'-'*10}")

    results_consensus = {}
    for level in ["full", "majority", "split"]:
        group = consensus_groups[level]
        if not group:
            print(f"  {level:<10}    0   0.0%          —          —           —")
            continue

        acc_any  = round(np.mean([s.get("correct_any", 0) for s in group]), 4)
        acc_all  = round(np.mean([s.get("correct_all", 0) for s in group]), 4)
        mean_conf = round(np.mean([s.get("final_confidence", 0.5) for s in group]), 4)
        pct      = len(group) / len(samples) * 100

        print(f"  {level:<10} {len(group):>4}  {pct:>4.1f}%  "
              f"{acc_any:>9.4f}  {acc_all:>9.4f}  {mean_conf:>10.4f}")

        results_consensus[level] = {
            "n": len(group), "pct": round(pct, 1),
            "accuracy_any": acc_any, "accuracy_all": acc_all,
            "mean_confidence": mean_conf,
        }

    # Check if full > majority > split (expected ordering)
    if all(k in results_consensus for k in ["full", "majority", "split"]):
        full_acc = results_consensus["full"]["accuracy_any"]
        maj_acc  = results_consensus["majority"]["accuracy_any"]
        spl_acc  = results_consensus["split"]["accuracy_any"]
        print(f"\n  Ordering check (full > majority > split):")
        print(f"    full={full_acc:.4f}  majority={maj_acc:.4f}  split={spl_acc:.4f}")
        if full_acc >= maj_acc >= spl_acc:
            print("    => ✓ Monotonic — consensus is a valid uncertainty signal")
        else:
            print("    => ✗ Not monotonic — consensus signal is noisy at n=28")

    # ── 3. Flag breakdown by source ──────────────────────────────────────────
    print("\n── 3. Flag source breakdown ────────────────────────────────")
    split_flagged = sum(1 for s in samples
                        if s.get("consensus") == "split")
    std_flagged   = sum(1 for s in samples
                        if (s.get("agent_conf_std") or 0) > 0.20
                        and s.get("consensus") != "split")
    both_flagged  = sum(1 for s in samples
                        if s.get("consensus") == "split"
                        and (s.get("agent_conf_std") or 0) > 0.20)
    total_flagged = sum(1 for s in samples
                        if s.get("uncertainty_flag", False))

    print(f"  Total flagged        : {total_flagged} ({total_flagged/len(samples):.0%})")
    print(f"  Source: split only   : {split_flagged - both_flagged}")
    print(f"  Source: high std only: {std_flagged}")
    print(f"  Source: both         : {both_flagged}")

    # Save
    output = {
        "n_test": len(samples),
        "flag_analysis": {},
        "consensus_breakdown": results_consensus,
        "flag_sources": {
            "total_flagged":   total_flagged,
            "split_only":      split_flagged - both_flagged,
            "high_std_only":   std_flagged,
            "both":            both_flagged,
        },
    }
    for mode, label_key in [("any", "correct_any"), ("all", "correct_all")]:
        flags  = [bool(s.get("uncertainty_flag", False)) for s in samples]
        errors = [s.get(label_key, 1) == 0 for s in samples]
        tp = sum(f and e for f, e in zip(flags, errors))
        fp = sum(f and not e for f, e in zip(flags, errors))
        fn = sum(not f and e for f, e in zip(flags, errors))
        tn = sum(not f and not e for f, e in zip(flags, errors))
        p, r, f1 = _prf(tp, fp, fn)
        output["flag_analysis"][mode] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": p, "recall": r, "f1": f1,
        }

    _OUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved → {_OUT_PATH}")


if __name__ == "__main__":
    run()