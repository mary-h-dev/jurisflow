from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

_UNCERTAIN_VALUES = {"unknown", "unclear", "uncertain", "insufficient", "n/a", "none", "null"}


class ChecklistItemOut(BaseModel):
    condition: str
    necessary: bool
    satisfied: bool

    @field_validator("satisfied", mode="before")
    @classmethod
    def _coerce_satisfied(cls, v):
        """
        Some free-tier models occasionally return a string like "unknown"
        instead of a strict boolean when the model itself isn't sure.
        Treated as False (conservative -- burden of proof against
        applicability) rather than raising, so one ambiguous item doesn't
        discard the whole article's checklist (see verifier.py, which
        used to fail the entire ChecklistItemOut list on a single bad
        item before this was added).
        """
        if isinstance(v, bool):
            return v
        if isinstance(v, str) and v.strip().lower() in _UNCERTAIN_VALUES:
            logger.warning(f"ChecklistItemOut.satisfied={v!r} coerced to False")
            return False
        return v


class AuditedArticleOut(BaseModel):
    """
    is_applicable and topically_relevant answer two different questions
    and must NOT be conflated:
      - is_applicable: does the article's legal test come out True or
        False given the case facts (the checklist's necessary conditions).
      - topically_relevant: is the article about the same legal issue
        the query raises, regardless of which way is_applicable came out.

    A "no" answer is often the correct answer -- e.g. a query asking
    whether 2 participants meet a 3-participant threshold, where the
    governing article's own headcount condition is the reason the
    answer is no. That article is topically_relevant=True,
    is_applicable=False, and MUST still reach the deliberation layer:
    downstream code must not filter on is_applicable alone, or it will
    silently drop the article that the final answer needs to cite.
    """
    article_ref: str
    checklist: list[ChecklistItemOut]
    is_applicable: bool
    topically_relevant: bool
    auditor_confidence: float


class CombinedConfidenceOut(BaseModel):
    score: float
    level: str
    note: Optional[str] = None
    retrieval_score: float
    auditor_score: float
    prune_ratio: float


class AuditorOut(BaseModel):
    query: str
    verified_articles: list[AuditedArticleOut]
    pruned_articles: list[str]
    confidence: CombinedConfidenceOut