from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

_CHANNEL_WEIGHTS = {
    "feature": 0.4,
    "ruling":  0.4,
    "article": 0.2,
}

_GRAPH_SUPPORT_WEIGHT       = 0.15  # blended in only when applicable
_MISSING_CHANNEL_PENALTY    = 0.15
_ROUTING_UNCERTAINTY_WEIGHT = 0.15
_AMBIGUITY_PENALTY          = 0.10
_TOP_N_FOR_QUALITY          = 3

_ALL_CHANNELS = ("feature", "ruling", "article")


@dataclass
class UncertaintyVector:
    feature_quality:      float
    ruling_quality:       float
    article_quality:      float
    missing_channels:     list[str]       = field(default_factory=list)
    graph_support_quality: Optional[float] = None  # None = not applicable

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
    level:  str
    note:   Optional[str]
    vector: UncertaintyVector


def _channel_quality(scores: list[float]) -> float:
    if not scores:
        return 0.0
    top = scores[:_TOP_N_FOR_QUALITY]
    return sum(top) / len(top)


def build_uncertainty_vector(
    feature_scores:        list[float],
    ruling_scores:         list[float],
    article_scores:        list[float],
    skipped_channels:      list[str] | None = None,
    graph_support_quality: Optional[float]   = None,
) -> UncertaintyVector:
    """
    A channel intentionally skipped by the router is NOT the same signal as
    a channel that was queried but returned nothing. Only the latter counts
    as "missing" — the former is routing uncertainty, handled separately
    via routing_confidence/ambiguity_flag in compute_confidence.
    """
    skipped = set(skipped_channels or [])
    scores_by_channel = {
        "feature": feature_scores,
        "ruling":  ruling_scores,
        "article": article_scores,
    }

    missing = [
        name for name in _ALL_CHANNELS
        if name not in skipped and not scores_by_channel[name]
    ]

    return UncertaintyVector(
        feature_quality=_channel_quality(feature_scores),
        ruling_quality=_channel_quality(ruling_scores),
        article_quality=_channel_quality(article_scores),
        missing_channels=missing,
        graph_support_quality=graph_support_quality,
    )


def compute_confidence(
    vec:                UncertaintyVector,
    queried_channels:   list[str],
    routing_confidence: float = 1.0,
    ambiguity_flag:     bool = False,
) -> ConfidenceResult:
    """
    IMPORTANT: weighted_sum is computed only over channels actually queried,
    renormalized to sum to 1.0 among them. A skipped channel must NOT drag
    the score down the same way a missing (queried-but-empty) channel does
    — that was a real bug in the previous version.
    """
    quality_by_channel = vec.channel_qualities
    active = [c for c in _ALL_CHANNELS if c in queried_channels]

    if active:
        weight_total = sum(_CHANNEL_WEIGHTS[c] for c in active)
        weighted_sum = sum(
            _CHANNEL_WEIGHTS[c] * quality_by_channel[c] for c in active
        ) / weight_total
    else:
        weighted_sum = 0.0

    if vec.graph_support_quality is not None:
        weighted_sum = (
            (1 - _GRAPH_SUPPORT_WEIGHT) * weighted_sum
            + _GRAPH_SUPPORT_WEIGHT * vec.graph_support_quality
        )

    missing_penalty   = _MISSING_CHANNEL_PENALTY * len(vec.missing_channels)
    routing_penalty   = _ROUTING_UNCERTAINTY_WEIGHT * (1.0 - routing_confidence)
    ambiguity_penalty = _AMBIGUITY_PENALTY if ambiguity_flag else 0.0

    score = round(
        max(0.0, min(1.0, weighted_sum - missing_penalty - routing_penalty - ambiguity_penalty)),
        3,
    )

    if score >= 0.8:
        level, note = "high", None
    elif score >= 0.6:
        level, note = "medium", "This response may need further review."
    else:
        level, note = "low", "Confidence is low. Consulting a lawyer is recommended."

    return ConfidenceResult(score=score, level=level, note=note, vector=vec)