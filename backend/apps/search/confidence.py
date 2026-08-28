from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

_TOP_N_FOR_QUALITY = 3

# Calibrated via logistic regression on 140 leakage-free annotated samples,
# WITHOUT LLM channel gating (all three channels always queried) --
# AUROC=0.756, Spearman=0.401 on held-out test, vs AUROC=0.682/Spearman=0.151
# WITH gating. See apps/search/calibration/ for full methodology.
#
# Raw calibrated coefficients (calibrate_regression.py, "no_routing" run):
#   feature_quality: +0.000   ruling_quality: +0.200   article_quality: -0.002
# article_quality's coefficient was small/negative -- a Ridge (regularized)
# sensitivity check pushed it to +0.005, consistent with multicollinearity
# with ruling_quality, not a real negative effect (see calibration docs).
#
# PRODUCTION ADJUSTMENT: raw calibration gives ruling_quality ~100% of the
# signal (feature/article ~0), which is too fragile for production -- a
# single ruling_quality score would fully determine confidence, ignoring
# article/feature evidence entirely. We apply a floor (0.05) to
# feature/article before normalizing, so they still contribute a small,
# disclosed amount. This is a deliberate, documented deviation from the
# raw calibrated numbers for robustness, not a re-guess.
_RAW_CALIBRATED = {"feature_quality": 0.000, "ruling_quality": 0.200, "article_quality": 0.000}
_FLOOR = 0.05
_floored = {
    k: (v if v > _FLOOR else _FLOOR) for k, v in _RAW_CALIBRATED.items()
}
_weight_sum = sum(_floored.values())
_CHANNEL_WEIGHTS = {k: v / _weight_sum for k, v in _floored.items()}
# Result: feature_quality≈0.167, ruling_quality≈0.667, article_quality≈0.167

# Missing-channel penalty: kept from the earlier, manually-set design.
# This is NOT part of the calibrated model (calibration already implicitly
# reflects a missing channel via quality=0.0 feeding the weighted sum) --
# it is an additional, structural penalty on top, reflecting that having
# strictly less evidence is inherently worse regardless of what the
# available channels scored. Value unchanged from pre-calibration design.
_MISSING_CHANNEL_PENALTY = 0.1

_ALL_CHANNELS = ("feature", "ruling", "article")


@dataclass
class UncertaintyVector:
    """
    Per-channel retrieval quality. All three channels are now always
    queried (no LLM channel gating -- see services.py and the ablation
    results in calibration/). missing_channels lists channels that were
    queried but returned zero results; there is no longer a "skipped"
    case to distinguish from, since nothing is skipped anymore.
    """
    feature_quality: float
    ruling_quality:  float
    article_quality: float
    missing_channels: list[str] = field(default_factory=list)
    # graph_support is computed by graph_support.py but NOT called from
    # services.py anymore (ablation showed it hurt AUROC: 0.756 -> 0.675).
    # Field kept for schema stability; will always be None in production.
    graph_support_quality: Optional[float] = None

    @property
    def channel_qualities(self) -> dict[str, float]:
        return {
            "feature": self.feature_quality,
            "ruling":  self.ruling_quality,
            "article": self.article_quality,
        }


@dataclass
class ConfidenceResult:
    score:  float
    level:  str          # high | medium | low
    note:   str | None
    vector: UncertaintyVector


def _channel_quality(scores: list[float]) -> float:
    if not scores:
        return 0.0
    top = scores[:_TOP_N_FOR_QUALITY]
    return sum(top) / len(top)


def build_uncertainty_vector(
    feature_scores: list[float],
    ruling_scores:  list[float],
    article_scores: list[float],
) -> UncertaintyVector:
    missing = []
    if not feature_scores:
        missing.append("feature")
    if not ruling_scores:
        missing.append("ruling")
    if not article_scores:
        missing.append("article")

    return UncertaintyVector(
        feature_quality=_channel_quality(feature_scores),
        ruling_quality=_channel_quality(ruling_scores),
        article_quality=_channel_quality(article_scores),
        missing_channels=missing,
    )


def compute_confidence(vec: UncertaintyVector) -> ConfidenceResult:
    """
    score = weighted average of channel qualities (weights derived from
    calibration, see module docstring), minus a penalty per missing
    channel. This is a simplified, production-safe version of the
    calibrated model -- NOT a raw sigmoid(logistic regression), to keep
    scores interpretable and in the same [0,1] range as before.

    routing_confidence, ambiguity_flag, and graph_support are
    intentionally NOT part of this formula: all three had near-zero or
    unproven effect across every calibration run (see calibration/ for
    the 5-parameter and graph_support ablation runs).
    """
    weighted_sum = (
        _CHANNEL_WEIGHTS["feature_quality"] * vec.feature_quality
        + _CHANNEL_WEIGHTS["ruling_quality"]  * vec.ruling_quality
        + _CHANNEL_WEIGHTS["article_quality"] * vec.article_quality
    )
    missing_penalty = _MISSING_CHANNEL_PENALTY * len(vec.missing_channels)

    score = round(max(0.0, min(1.0, weighted_sum - missing_penalty)), 3)

    if score >= 0.7:
        level, note = "high", None
    elif score >= 0.5:
        level, note = "medium", "This response may need further review."
    else:
        level, note = "low", "Confidence is low. Consulting a lawyer is recommended."

    return ConfidenceResult(score=score, level=level, note=note, vector=vec)