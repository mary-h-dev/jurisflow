from pydantic import BaseModel
from typing import Optional


# ── Input ─────────────────────────────────────────────────────────────────────

class CaseAnalysisIn(BaseModel):
    """ورودی تحلیل پرونده"""
    extracted_text: str
    case_context:   str = ""


# ── Sub-schemas ───────────────────────────────────────────────────────────────

class PartyOut(BaseModel):
    role:    str
    name:    str
    details: str


class RelatedLawOut(BaseModel):
    article_ref: str
    law_name:    str
    summary:     str
    relevance:   float


class RiskOut(BaseModel):
    title:       str
    description: str
    severity:    str


class TeammateOpinionOut(BaseModel):
    """نظر هر teammate — برای نمایش debate در frontend"""
    role:           str
    position:       str
    arguments:      list[str]
    cited_articles: list[str]
    confidence:     str


# ── Main Output ───────────────────────────────────────────────────────────────

class CaseAnalysisOut(BaseModel):
    """خروجی کامل تحلیل پرونده"""

    session_id: str

    # خروجی Analyzer
    case_subject: str
    case_type:    str
    parties:      list[PartyOut]
    key_facts:    list[str]

    # خروجی Searcher
    related_laws:   list[RelatedLawOut]
    cited_articles: list[str]

    # نظرات Teammates
    defender_opinion:   Optional[TeammateOpinionOut] = None
    prosecutor_opinion: Optional[TeammateOpinionOut] = None
    judge_opinion:      Optional[TeammateOpinionOut] = None

    # خروجی Lead Agent
    strengths:  list[str]
    weaknesses: list[str]
    risks:      list[RiskOut]
    win_chance: str

    # توصیه نهایی
    strategy:       str
    recommendation: str
    next_steps:     list[str]

    # Metadata
    completed_nodes: list[str]
    error:           Optional[str] = None


# ── Progress ──────────────────────────────────────────────────────────────────

class ProgressOut(BaseModel):
    """وضعیت realtime برای frontend"""
    session_id:      str
    current_node:    str
    completed_nodes: list[str]
    status:          str