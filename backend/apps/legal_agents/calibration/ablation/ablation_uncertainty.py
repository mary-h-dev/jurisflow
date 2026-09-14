# """
# Ablation study: compares uncertainty signals across pipeline layers
# using cached e2e data — no API calls required.

# Variants compared:
#   1. retrieval_confidence      (baseline signal)
#   2. auditor_confidence        (after auditor layer)
#   3. final_confidence          (full fusion: prior + agents)
#   4. final_no_consensus        (fusion without split penalty)
#   5. agent_mean_only           (agent confidence, ignoring prior)
#   6. prior_only                (auditor prior, ignoring agents)

# All computed from e2e_calibration.json + e2e_test.json — zero API cost.

# Run:
#     python ablation_uncertainty.py
#     (from backend/ directory, with Django not required)
# """

# from __future__ import annotations

# import json
# import math
# from pathlib import Path

# import numpy as np
# from sklearn.metrics import roc_auc_score

# _CALIB_PATH = Path("apps/legal_agents/calibration/data/e2e_calibration.json")
# _TEST_PATH  = Path("apps/legal_agents/calibration/data/e2e_test.json")
# _OUT_PATH   = Path("apps/legal_agents/calibration/data/ablation_results.json")

# _PRIOR_WEIGHT = 0.40
# _AGENT_WEIGHT = 0.60
# _SPLIT_PENALTY = 0.10


# # ---------------------------------------------------------------------------
# # Compute confidence variants
# # ---------------------------------------------------------------------------

# def _variants(sample: dict) -> dict[str, float]:
#     retrieval  = sample.get("retrieval_confidence", 0.5)
#     auditor    = sample.get("auditor_confidence", 0.5)
#     agent_mean = sample.get("agent_conf_mean") or 0.5
#     consensus  = sample.get("consensus", "majority")
#     final      = sample.get("final_confidence", 0.5)

#     blended = _PRIOR_WEIGHT * auditor + _AGENT_WEIGHT * agent_mean
#     no_penalty = round(min(1.0, blended), 4)
#     with_penalty = round(max(0.0, blended - (_SPLIT_PENALTY if consensus == "split" else 0.0)), 4)

#     return {
#         "retrieval_only":     round(retrieval, 4),
#         "auditor_only":       round(auditor, 4),
#         "agent_mean_only":    round(agent_mean, 4),
#         "prior_only":         round(auditor, 4),   # same as auditor_only, explicit alias
#         "fusion_no_consensus": no_penalty,
#         "full_fusion":        round(final, 4),      # final_confidence from cache
#     }


# # ---------------------------------------------------------------------------
# # Metrics
# # ---------------------------------------------------------------------------

# def _ece(confs: list[float], labels: list[int], n_bins: int = 5) -> float:
#     if len(set(labels)) < 2:
#         return float("nan")
#     c = np.array(confs)
#     l = np.array(labels, dtype=float)
#     edges = np.linspace(0.0, 1.0, n_bins + 1)
#     ece = 0.0
#     n = len(c)
#     for lo, hi in zip(edges[:-1], edges[1:]):
#         mask = (c >= lo) & (c < hi)
#         if mask.sum() == 0:
#             continue
#         ece += mask.sum() / n * abs(l[mask].mean() - c[mask].mean())
#     return round(float(ece), 4)


# def _safe_auroc(labels: list[int], scores: list[float]) -> float | None:
#     if len(set(labels)) < 2:
#         return None
#     return round(roc_auc_score(labels, scores), 4)


# def _brier(confs: list[float], labels: list[int]) -> float:
#     return round(float(np.mean([(c - l) ** 2
#                                 for c, l in zip(confs, labels)])), 4)


# # ---------------------------------------------------------------------------
# # Main
# # ---------------------------------------------------------------------------

# def run() -> None:
#     calib_data = json.loads(_CALIB_PATH.read_text(encoding="utf-8"))
#     test_data  = json.loads(_TEST_PATH.read_text(encoding="utf-8"))

#     test_samples = [s for s in test_data["per_sample"]
#                     if "error" not in s and s.get("agent_conf_mean") is not None]

#     labels_any = [s["correct_any"] for s in test_samples]
#     labels_all = [s["correct_all"] for s in test_samples]

#     print("=" * 65)
#     print("ABLATION: Uncertainty Signal Comparison (n=28, no API calls)")
#     print("=" * 65)
#     print(f"\n{'Variant':<22} {'AUROC(any)':>11} {'ECE(any)':>9} "
#           f"{'AUROC(all)':>11} {'ECE(all)':>9} {'Brier':>7}")
#     print("-" * 65)

#     results = {}

#     variant_labels = {
#         "retrieval_only":      "Retrieval only",
#         "auditor_only":        "Auditor only",
#         "agent_mean_only":     "Agent mean only",
#         "fusion_no_consensus": "Fusion (no consensus)",
#         "full_fusion":         "Full fusion (ours)",
#     }

#     for key, label in variant_labels.items():
#         confs = [_variants(s)[key] for s in test_samples]

#         auroc_any = _safe_auroc(labels_any, confs)
#         auroc_all = _safe_auroc(labels_all, confs)
#         ece_any   = _ece(confs, labels_any)
#         ece_all   = _ece(confs, labels_all)
#         brier     = _brier(confs, labels_any)

#         auroc_any_str = f"{auroc_any:.4f}" if auroc_any else "  N/A  "
#         auroc_all_str = f"{auroc_all:.4f}" if auroc_all else "  N/A  "

#         print(f"{label:<22} {auroc_any_str:>11} {ece_any:>9.4f} "
#               f"{auroc_all_str:>11} {ece_all:>9.4f} {brier:>7.4f}")

#         results[key] = {
#             "label":     label,
#             "auroc_any": auroc_any,
#             "ece_any":   ece_any,
#             "auroc_all": auroc_all,
#             "ece_all":   ece_all,
#             "brier":     brier,
#             "mean_conf": round(float(np.mean(confs)), 4),
#             "std_conf":  round(float(np.std(confs)), 4),
#         }

#     print("\n" + "=" * 65)
#     print("CONSENSUS EFFECT (full_fusion vs fusion_no_consensus)")
#     print("=" * 65)
#     full    = results["full_fusion"]
#     no_pen  = results["fusion_no_consensus"]
#     delta_auroc = (full["auroc_any"] or 0) - (no_pen["auroc_any"] or 0)
#     delta_ece   = full["ece_any"] - no_pen["ece_any"]
#     print(f"  AUROC delta (full - no_consensus) : {delta_auroc:+.4f}")
#     print(f"  ECE   delta (full - no_consensus) : {delta_ece:+.4f}")
#     if delta_auroc > 0:
#         print("  => Consensus penalty IMPROVES discrimination")
#     else:
#         print("  => Consensus penalty HURTS discrimination (but may improve calibration)")

#     print("\n" + "=" * 65)
#     print("LAYER CONTRIBUTION (which layer adds the most signal?)")
#     print("=" * 65)
#     ret_auroc = results["retrieval_only"]["auroc_any"] or 0
#     aud_auroc = results["auditor_only"]["auroc_any"] or 0
#     full_auroc = results["full_fusion"]["auroc_any"] or 0
#     print(f"  Retrieval → Auditor  : {ret_auroc:.4f} → {aud_auroc:.4f}  "
#           f"(Δ={aud_auroc - ret_auroc:+.4f})")
#     print(f"  Auditor   → Full     : {aud_auroc:.4f} → {full_auroc:.4f}  "
#           f"(Δ={full_auroc - aud_auroc:+.4f})")

#     # Save
#     output = {
#         "n_test":   len(test_samples),
#         "variants": results,
#         "note":     "All computed from cached e2e data — zero API calls",
#     }
#     _OUT_PATH.write_text(
#         json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
#     )
#     print(f"\nSaved → {_OUT_PATH}")


# if __name__ == "__main__":
#     run()







"""
Ablation study: compares uncertainty signals across pipeline layers
using cached e2e data — no API calls required.

Variants compared:
  1. retrieval_confidence      (baseline signal)
  2. auditor_confidence        (after auditor layer)
  3. final_confidence          (full fusion: prior + agents)
  4. final_no_consensus        (fusion without split penalty)
  5. agent_mean_only           (agent confidence, ignoring prior)
  6. prior_only                (auditor prior, ignoring agents)

All computed from e2e_calibration.json + e2e_test.json — zero API cost.

Run:
    python ablation_uncertainty.py
    (from backend/ directory, with Django not required)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

_CALIB_PATH = Path("apps/legal_agents/calibration/data/e2e_calibration.json")
_TEST_PATH  = Path("apps/legal_agents/calibration/data/e2e_test.json")
_OUT_PATH   = Path("apps/legal_agents/calibration/data/ablation_results.json")

_PRIOR_WEIGHT = 0.40
_AGENT_WEIGHT = 0.60
_SPLIT_PENALTY = 0.10


# ---------------------------------------------------------------------------
# Compute confidence variants
# ---------------------------------------------------------------------------

def _variants(sample: dict) -> dict[str, float]:
    retrieval  = sample.get("retrieval_confidence", 0.5)
    auditor    = sample.get("auditor_confidence", 0.5)
    agent_mean = sample.get("agent_conf_mean") or 0.5
    consensus  = sample.get("consensus", "majority")
    final      = sample.get("final_confidence", 0.5)

    blended = _PRIOR_WEIGHT * auditor + _AGENT_WEIGHT * agent_mean
    no_penalty = round(min(1.0, blended), 4)
    with_penalty = round(max(0.0, blended - (_SPLIT_PENALTY if consensus == "split" else 0.0)), 4)

    return {
        "retrieval_only":     round(retrieval, 4),
        "auditor_only":       round(auditor, 4),
        "agent_mean_only":    round(agent_mean, 4),
        "prior_only":         round(auditor, 4),   # same as auditor_only, explicit alias
        "fusion_no_consensus": no_penalty,
        "full_fusion":        round(final, 4),      # final_confidence from cache
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _ece(confs: list[float], labels: list[int], n_bins: int = 5) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    c = np.array(confs)
    l = np.array(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(c)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (c >= lo) & (c < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(l[mask].mean() - c[mask].mean())
    return round(float(ece), 4)


def _safe_auroc(labels: list[int], scores: list[float]) -> float | None:
    if len(set(labels)) < 2:
        return None
    return round(roc_auc_score(labels, scores), 4)


def _brier(confs: list[float], labels: list[int]) -> float:
    return round(float(np.mean([(c - l) ** 2
                                for c, l in zip(confs, labels)])), 4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    calib_data = json.loads(_CALIB_PATH.read_text(encoding="utf-8"))
    test_data  = json.loads(_TEST_PATH.read_text(encoding="utf-8"))

    test_samples = [s for s in test_data["per_sample"]
                    if "error" not in s and s.get("agent_conf_mean") is not None]

    # Layer-specific labels — each layer is judged on what IT controls
    # not on the end-to-end outcome of downstream layers.
    #
    # retrieval_only : did retrieval surface at least one gold article?
    labels_retrieval = [
        int(bool(set(s.get("gold_articles", [])) &
                 set(s.get("retrieval_article_refs", []))))
        for s in test_samples
    ]

    # auditor_only : did gold survive auditor pruning?
    labels_auditor = [s.get("gold_in_applicable", 0) for s in test_samples]

    # agent / fusion : end-to-end correctness (agents own this outcome)
    labels_any = [s["correct_any"] for s in test_samples]
    labels_all = [s["correct_all"] for s in test_samples]

    print("=" * 70)
    print("ABLATION: Uncertainty Signal Comparison (n=28, layer-specific labels)")
    print("=" * 70)
    print("Labels used per variant:")
    print("  retrieval_only      → gold in retrieval_article_refs")
    print("  auditor_only        → gold_in_applicable (survived pruning)")
    print("  agent/fusion        → correct_any (end-to-end)")
    print()
    print(f"{'Variant':<22} {'AUROC(any)':>11} {'ECE(any)':>9} "
          f"{'AUROC(all)':>11} {'ECE(all)':>9} {'Brier':>7} {'Label':>12}")
    print("-" * 75)

    results = {}

    variant_configs = [
        ("retrieval_only",      "Retrieval only",       labels_retrieval, labels_retrieval),
        ("auditor_only",        "Auditor only",         labels_auditor,   labels_auditor),
        ("agent_mean_only",     "Agent mean only",      labels_any,       labels_all),
        ("fusion_no_consensus", "Fusion (no consensus)", labels_any,      labels_all),
        ("full_fusion",         "Full fusion (ours)",   labels_any,       labels_all),
    ]

    for key, label, lab_any, lab_all in variant_configs:
        confs = [_variants(s)[key] for s in test_samples]

        auroc_any = _safe_auroc(lab_any, confs)
        auroc_all = _safe_auroc(lab_all, confs)
        ece_any   = _ece(confs, lab_any)
        ece_all   = _ece(confs, lab_all)
        brier     = _brier(confs, lab_any)

        auroc_any_str = f"{auroc_any:.4f}" if auroc_any else "  N/A  "
        auroc_all_str = f"{auroc_all:.4f}" if auroc_all else "  N/A  "

        lbl_name = ("retrieval" if key == "retrieval_only"
                    else "auditor" if key == "auditor_only"
                    else "end-to-end")

        print(f"{label:<22} {auroc_any_str:>11} {ece_any:>9.4f} "
              f"{auroc_all_str:>11} {ece_all:>9.4f} {brier:>7.4f} {lbl_name:>12}")

        results[key] = {
            "label":      label,
            "auroc_any":  auroc_any,
            "ece_any":    ece_any,
            "auroc_all":  auroc_all,
            "ece_all":    ece_all,
            "brier":      brier,
            "mean_conf":  round(float(np.mean(confs)), 4),
            "std_conf":   round(float(np.std(confs)), 4),
            "label_type": lbl_name,
        }

    print("\n" + "=" * 70)
    print("CONSENSUS EFFECT (full_fusion vs fusion_no_consensus)")
    print("=" * 70)
    full   = results["full_fusion"]
    no_pen = results["fusion_no_consensus"]
    delta_auroc = (full["auroc_any"] or 0) - (no_pen["auroc_any"] or 0)
    delta_ece   = full["ece_any"] - no_pen["ece_any"]
    print(f"  AUROC delta (full - no_consensus) : {delta_auroc:+.4f}")
    print(f"  ECE   delta (full - no_consensus) : {delta_ece:+.4f}")
    if delta_auroc > 0:
        print("  => Consensus penalty IMPROVES discrimination")
    else:
        print("  => Consensus penalty HURTS discrimination")

    print("\n" + "=" * 70)
    print("LAYER CONTRIBUTION (layer-specific labels)")
    print("=" * 70)
    ret_auroc  = results["retrieval_only"]["auroc_any"] or 0
    aud_auroc  = results["auditor_only"]["auroc_any"] or 0
    full_auroc = results["full_fusion"]["auroc_any"] or 0
    print(f"  Retrieval signal  (vs gold-in-retrieval) : {ret_auroc:.4f}")
    print(f"  Auditor signal    (vs gold-in-applicable): {aud_auroc:.4f}")
    print(f"  Full fusion       (vs end-to-end)        : {full_auroc:.4f}")
    print(f"\n  Note: labels differ per row — direct numeric comparison")
    print(f"  is only valid within same label type.")

    # Save
    output = {
        "n_test":   len(test_samples),
        "variants": results,
        "note":     "Layer-specific labels: retrieval→gold_in_retrieval, "
                    "auditor→gold_in_applicable, agent/fusion→correct_any",
    }
    _OUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved → {_OUT_PATH}")


if __name__ == "__main__":
    run()