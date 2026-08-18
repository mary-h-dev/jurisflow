from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from apps.search.confidence import ConfidenceResult

from .schemas import AuditorOut

_RETRIEVAL_WEIGHT = 0.5
_AUDITOR_WEIGHT = 0.5
_PRUNE_RATIO_PENALTY_WEIGHT = 0.2  # high prune ratio -> initial retrieval was noisy


@dataclass
class AuditorConfidenceResult:
    score: float           # mean auditor_confidence over KEPT (is_applicable=True) articles
    kept_count: int
    pruned_count: int
    prune_ratio: float     # pruned_count / (kept_count + pruned_count), 0.0 if none verified


@dataclass
class CombinedConfidenceResult:
    score: float
    level: str              # "high" | "medium" | "low"
    note: Optional[str]
    retrieval_score: float
    auditor_score: float
    prune_ratio: float


def compute_auditor_confidence(auditor_out: AuditorOut) -> AuditorConfidenceResult:
    """
    Aggregates AuditorOut.verified_articles into a single score. Only
    articles that survived pruning (is_applicable=True) are averaged --
    a pruned article's auditor_confidence says something about how
    confidently it was REJECTED, not about how confident the surviving
    answer is, so including it would conflate two different questions.

    If every verified article was pruned (kept_count == 0), score is
    0.0: no supporting article survived, which is itself a strong
    negative signal rather than an "undefined" one.
    """
    kept = [a for a in auditor_out.verified_articles if a.is_applicable]
    pruned = [a for a in auditor_out.verified_articles if not a.is_applicable]

    total = len(kept) + len(pruned)
    prune_ratio = (len(pruned) / total) if total else 0.0
    score = (sum(a.auditor_confidence for a in kept) / len(kept)) if kept else 0.0

    return AuditorConfidenceResult(
        score=round(score, 3),
        kept_count=len(kept),
        pruned_count=len(pruned),
        prune_ratio=round(prune_ratio, 3),
    )


def combine_confidence(
    retrieval: ConfidenceResult,
    auditor: AuditorConfidenceResult,
) -> CombinedConfidenceResult:
    """
    Deterministic fusion of retrieval-layer and auditor-layer confidence
    -- NO LLM call here by design, so uncertainty stays traceable to a
    fixed formula rather than another model's opinion of its own
    confidence. A high prune_ratio (initial retrieval brought in a lot
    of articles the Auditor rejected) further penalizes the score, since
    it's itself evidence the retrieval stage was noisy for this query.
    """
    weighted = (
        _RETRIEVAL_WEIGHT * retrieval.score
        + _AUDITOR_WEIGHT * auditor.score
    )
    penalty = _PRUNE_RATIO_PENALTY_WEIGHT * auditor.prune_ratio
    score = round(max(0.0, min(1.0, weighted - penalty)), 3)

    if score >= 0.8:
        level, note = "high", None
    elif score >= 0.6:
        level, note = "medium", "This response may need further review."
    else:
        level, note = "low", "Confidence is low. Consulting a lawyer is recommended."

    return CombinedConfidenceResult(
        score=score,
        level=level,
        note=note,
        retrieval_score=retrieval.score,
        auditor_score=auditor.score,
        prune_ratio=auditor.prune_ratio,
    )