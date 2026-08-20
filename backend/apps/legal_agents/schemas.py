"""
Data schemas for the legal-agents deliberation layer.

Inputs  : AuditorOut (produced by apps.auditor)
Outputs : AgentOpinion (per agent) → FusionOut (final)
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Inputs from the Auditor layer
# ---------------------------------------------------------------------------

class ChecklistItemOut(BaseModel):
    condition: str
    necessary: bool
    satisfied: bool


class AuditedArticleOut(BaseModel):
    article_ref:        str
    checklist:          list[ChecklistItemOut]
    is_applicable:      bool
    auditor_confidence: float = Field(ge=0.0, le=1.0)


class CombinedConfidenceOut(BaseModel):
    score:           float = Field(ge=0.0, le=1.0)
    level:           str                            # "high" | "medium" | "low"
    note:            Optional[str] = None
    retrieval_score: float = Field(ge=0.0, le=1.0)
    auditor_score:   float = Field(ge=0.0, le=1.0)
    prune_ratio:     float = Field(ge=0.0, le=1.0)


class AuditorOut(BaseModel):
    query:             str
    verified_articles: list[AuditedArticleOut]
    pruned_articles:   list[str]
    confidence:        CombinedConfidenceOut


# ---------------------------------------------------------------------------
# Agent output  (one per agent, comparable for deterministic fusion)
# ---------------------------------------------------------------------------

class Verdict(str, Enum):
    STRONG_FOR       = "strong_for"
    MODERATE_FOR     = "moderate_for"
    NEUTRAL          = "neutral"
    MODERATE_AGAINST = "moderate_against"
    STRONG_AGAINST   = "strong_against"


class AgentOpinion(BaseModel):
    role:            str
    verdict:         Verdict
    confidence:      float = Field(ge=0.0, le=1.0)
    position:        str
    arguments:       list[str]
    cited_articles:  list[str]
    risks:           list[str] = []


# ---------------------------------------------------------------------------
# Deterministic fusion output
# ---------------------------------------------------------------------------

class ConsensusLevel(str, Enum):
    FULL     = "full"       # all three agree
    MAJORITY = "majority"   # two of three agree
    SPLIT    = "split"      # no majority


class FusionOut(BaseModel):
    consensus_level:       ConsensusLevel
    majority_verdict:      Verdict
    verdict_spread:        dict[str, str]

    agent_confidence_mean: float = Field(ge=0.0, le=1.0)
    agent_confidence_std:  float = Field(ge=0.0)
    prior_confidence:      float = Field(ge=0.0, le=1.0)
    final_confidence:      float = Field(ge=0.0, le=1.0)
    final_level:           str

    agreed_articles:     list[str]
    all_cited_articles:  list[str]
    all_arguments:       dict[str, list[str]]

    uncertainty_flag: bool
    notes:            list[str] = []


# ---------------------------------------------------------------------------
# Final API response
# ---------------------------------------------------------------------------

class DeliberationOut(BaseModel):
    """Top-level response returned by DeliberationService."""
    session_id: str

    defender_opinion:   Optional[AgentOpinion] = None
    prosecutor_opinion: Optional[AgentOpinion] = None
    judge_opinion:      Optional[AgentOpinion] = None

    fusion:              Optional[FusionOut]          = None
    applicable_articles: list[AuditedArticleOut]      = []
    auditor_confidence:  Optional[CombinedConfidenceOut] = None

    completed_nodes: list[str] = []
    error:           Optional[str] = None