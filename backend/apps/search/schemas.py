from pydantic import BaseModel
from typing import Optional


# ── Input ─────────────────────────────────────────────────────────────────────

class SearchIn(BaseModel):
    query: str
    law:   str = "قانون مدنی"


# ── Output ────────────────────────────────────────────────────────────────────

class SourceOut(BaseModel):
    article_number: Optional[int] = None
    title:          Optional[str] = None
    source_type:    str = "law"
    law:            str


class ConfidenceBreakdownOut(BaseModel):
    embedding: float
    llm:       float
    graph:     float


class ConfidenceOut(BaseModel):
    final:     float
    level:     str        # high | medium | low
    note:      Optional[str] = None
    breakdown: ConfidenceBreakdownOut


class SearchOut(BaseModel):
    answer:     str
    confidence: ConfidenceOut
    sources:    list[SourceOut]