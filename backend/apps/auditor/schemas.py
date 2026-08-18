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
    article_ref: str
    checklist: list[ChecklistItemOut]
    is_applicable: bool
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