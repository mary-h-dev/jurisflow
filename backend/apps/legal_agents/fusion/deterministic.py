"""
Deterministic fusion of three AgentOpinion objects — no LLM involved.

Rules:
  1. Majority verdict  — whichever verdict appears ≥ 2 times wins.
     All three different → SPLIT → NEUTRAL + uncertainty_flag.
  2. Confidence blend  — 40% Auditor prior + 60% agent mean.
     SPLIT → additional 0.10 penalty.
  3. Uncertainty flag  — True when agent std > 0.20 OR consensus is SPLIT.
  4. Agreed articles   — cited by at least 2 of the 3 agents.
"""

from __future__ import annotations

import statistics
from collections import Counter

from apps.legal_agents.schemas import (
    AgentOpinion,
    CombinedConfidenceOut,
    ConsensusLevel,
    FusionOut,
    Verdict,
)

_PRIOR_WEIGHT    = 0.40
_AGENT_WEIGHT    = 0.60
_HIGH_STD        = 0.20
_SPLIT_PENALTY   = 0.10
_HIGH_THRESHOLD  = 0.70
_MED_THRESHOLD   = 0.45


def _level(score: float) -> str:
    if score >= _HIGH_THRESHOLD:
        return "high"
    if score >= _MED_THRESHOLD:
        return "medium"
    return "low"


def _majority(opinions: list[AgentOpinion]) -> tuple[Verdict, ConsensusLevel]:
    counts = Counter(op.verdict for op in opinions)
    top_verdict, top_count = counts.most_common(1)[0]
    if top_count == 3:
        return top_verdict, ConsensusLevel.FULL
    if top_count == 2:
        return top_verdict, ConsensusLevel.MAJORITY
    return Verdict.NEUTRAL, ConsensusLevel.SPLIT


def _agreed_articles(opinions: list[AgentOpinion]) -> list[str]:
    counts: Counter[str] = Counter()
    for op in opinions:
        counts.update(set(op.cited_articles))
    return [a for a, n in counts.items() if n >= 2]


def fuse(opinions: list[AgentOpinion], prior: CombinedConfidenceOut) -> FusionOut:
    if len(opinions) != 3:
        raise ValueError(f"fuse() expects exactly 3 opinions, got {len(opinions)}")

    majority_verdict, consensus_level = _majority(opinions)

    conf_scores = [op.confidence for op in opinions]
    agent_mean  = statistics.mean(conf_scores)
    agent_std   = statistics.stdev(conf_scores)

    blended = _PRIOR_WEIGHT * prior.score + _AGENT_WEIGHT * agent_mean
    if consensus_level == ConsensusLevel.SPLIT:
        blended = max(0.0, blended - _SPLIT_PENALTY)
    final_confidence = round(min(1.0, blended), 4)

    uncertainty_flag = agent_std > _HIGH_STD or consensus_level == ConsensusLevel.SPLIT

    notes: list[str] = []
    if uncertainty_flag:
        notes.append(
            f"High uncertainty: consensus={consensus_level.value}, std={agent_std:.2f}"
        )
    if prior.prune_ratio > 0.5:
        notes.append(f"Noisy retrieval: {prior.prune_ratio:.0%} of articles pruned")

    return FusionOut(
        consensus_level       = consensus_level,
        majority_verdict      = majority_verdict,
        verdict_spread        = {op.role: op.verdict.value for op in opinions},
        agent_confidence_mean = round(agent_mean, 4),
        agent_confidence_std  = round(agent_std, 4),
        prior_confidence      = prior.score,
        final_confidence      = final_confidence,
        final_level           = _level(final_confidence),
        agreed_articles       = _agreed_articles(opinions),
        all_cited_articles    = list({a for op in opinions for a in op.cited_articles}),
        all_arguments         = {op.role: op.arguments for op in opinions},
        uncertainty_flag      = uncertainty_flag,
        notes                 = notes,
    )