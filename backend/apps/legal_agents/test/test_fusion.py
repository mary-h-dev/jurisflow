"""
Unit tests for apps.legal_agents.fusion.deterministic.

No LLM calls, no Django setup required.
Run with:  pytest apps/legal_agents/tests/test_fusion.py -v
"""

from __future__ import annotations

import pytest

from apps.legal_agents.fusion.deterministic import (
    _HIGH_STD,
    _PRIOR_WEIGHT,
    _AGENT_WEIGHT,
    _SPLIT_PENALTY,
    fuse,
)
from apps.legal_agents.schemas import (
    AgentOpinion,
    CombinedConfidenceOut,
    ConsensusLevel,
    Verdict,
)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _opinion(
    role:       str,
    verdict:    Verdict,
    confidence: float,
    cited:      list[str] | None = None,
) -> AgentOpinion:
    return AgentOpinion(
        role           = role,
        verdict        = verdict,
        confidence     = confidence,
        position       = f"{role} position",
        arguments      = [f"{role} argument"],
        cited_articles = cited or [],
        risks          = [],
    )


def _prior(score: float = 0.75, prune_ratio: float = 0.20) -> CombinedConfidenceOut:
    return CombinedConfidenceOut(
        score           = score,
        level           = "high" if score >= 0.70 else "medium",
        retrieval_score = score,
        auditor_score   = score,
        prune_ratio     = prune_ratio,
    )


# ---------------------------------------------------------------------------
# Consensus / majority vote
# ---------------------------------------------------------------------------

class TestMajorityVote:

    def test_full_consensus(self):
        opinions = [
            _opinion("defender",   Verdict.STRONG_FOR, 0.90),
            _opinion("prosecutor", Verdict.STRONG_FOR, 0.85),
            _opinion("judge",      Verdict.STRONG_FOR, 0.88),
        ]
        result = fuse(opinions, _prior())
        assert result.consensus_level  == ConsensusLevel.FULL
        assert result.majority_verdict == Verdict.STRONG_FOR

    def test_majority_two_of_three(self):
        opinions = [
            _opinion("defender",   Verdict.MODERATE_FOR,   0.70),
            _opinion("prosecutor", Verdict.STRONG_AGAINST, 0.60),
            _opinion("judge",      Verdict.MODERATE_FOR,   0.65),
        ]
        result = fuse(opinions, _prior())
        assert result.consensus_level  == ConsensusLevel.MAJORITY
        assert result.majority_verdict == Verdict.MODERATE_FOR

    def test_split_returns_neutral(self):
        opinions = [
            _opinion("defender",   Verdict.STRONG_FOR,     0.80),
            _opinion("prosecutor", Verdict.STRONG_AGAINST, 0.80),
            _opinion("judge",      Verdict.NEUTRAL,        0.50),
        ]
        result = fuse(opinions, _prior())
        assert result.consensus_level  == ConsensusLevel.SPLIT
        assert result.majority_verdict == Verdict.NEUTRAL

    def test_verdict_spread_contains_all_roles(self):
        opinions = [
            _opinion("defender",   Verdict.MODERATE_FOR, 0.70),
            _opinion("prosecutor", Verdict.MODERATE_FOR, 0.65),
            _opinion("judge",      Verdict.MODERATE_FOR, 0.68),
        ]
        result = fuse(opinions, _prior())
        assert set(result.verdict_spread.keys()) == {"defender", "prosecutor", "judge"}


# ---------------------------------------------------------------------------
# Confidence blending
# ---------------------------------------------------------------------------

class TestConfidenceBlending:

    def test_blended_within_bounds(self):
        opinions = [
            _opinion("defender",   Verdict.MODERATE_FOR, 0.70),
            _opinion("prosecutor", Verdict.MODERATE_FOR, 0.60),
            _opinion("judge",      Verdict.MODERATE_FOR, 0.65),
        ]
        result = fuse(opinions, _prior(score=0.80))
        assert 0.0 <= result.final_confidence <= 1.0

    def test_blend_formula(self):
        # All agents agree, identical confidence → easy to verify formula
        agent_conf = 0.70
        prior_score = 0.80
        opinions = [
            _opinion("defender",   Verdict.MODERATE_FOR, agent_conf),
            _opinion("prosecutor", Verdict.MODERATE_FOR, agent_conf),
            _opinion("judge",      Verdict.MODERATE_FOR, agent_conf),
        ]
        result   = fuse(opinions, _prior(score=prior_score))
        expected = round(_PRIOR_WEIGHT * prior_score + _AGENT_WEIGHT * agent_conf, 4)
        assert result.final_confidence == expected

    def test_split_penalty_applied(self):
        # All same confidence so the only difference is the SPLIT penalty
        agent_conf  = 0.70
        prior_score = 0.75
        opinions = [
            _opinion("defender",   Verdict.STRONG_FOR,     agent_conf),
            _opinion("prosecutor", Verdict.STRONG_AGAINST, agent_conf),
            _opinion("judge",      Verdict.NEUTRAL,        agent_conf),
        ]
        result         = fuse(opinions, _prior(score=prior_score))
        without_penalty = _PRIOR_WEIGHT * prior_score + _AGENT_WEIGHT * agent_conf
        assert result.final_confidence == round(without_penalty - _SPLIT_PENALTY, 4)

    def test_confidence_never_below_zero(self):
        opinions = [
            _opinion("defender",   Verdict.STRONG_FOR,     0.01),
            _opinion("prosecutor", Verdict.STRONG_AGAINST, 0.01),
            _opinion("judge",      Verdict.NEUTRAL,        0.01),
        ]
        result = fuse(opinions, _prior(score=0.01))
        assert result.final_confidence >= 0.0


# ---------------------------------------------------------------------------
# Uncertainty flag
# ---------------------------------------------------------------------------

class TestUncertaintyFlag:

    def test_flag_off_when_full_consensus_low_std(self):
        opinions = [
            _opinion("defender",   Verdict.MODERATE_FOR, 0.75),
            _opinion("prosecutor", Verdict.MODERATE_FOR, 0.73),
            _opinion("judge",      Verdict.MODERATE_FOR, 0.74),
        ]
        result = fuse(opinions, _prior())
        assert result.uncertainty_flag is False

    def test_flag_on_when_split(self):
        opinions = [
            _opinion("defender",   Verdict.STRONG_FOR,     0.70),
            _opinion("prosecutor", Verdict.STRONG_AGAINST, 0.70),
            _opinion("judge",      Verdict.NEUTRAL,        0.70),
        ]
        result = fuse(opinions, _prior())
        assert result.uncertainty_flag is True

    def test_flag_on_when_high_std(self):
        # std of [0.95, 0.95, 0.10] is well above _HIGH_STD threshold
        opinions = [
            _opinion("defender",   Verdict.STRONG_FOR, 0.95),
            _opinion("prosecutor", Verdict.STRONG_FOR, 0.95),
            _opinion("judge",      Verdict.STRONG_FOR, 0.10),
        ]
        result = fuse(opinions, _prior())
        assert result.uncertainty_flag is True

    def test_uncertainty_note_present_when_flagged(self):
        opinions = [
            _opinion("defender",   Verdict.STRONG_FOR,     0.70),
            _opinion("prosecutor", Verdict.STRONG_AGAINST, 0.70),
            _opinion("judge",      Verdict.NEUTRAL,        0.70),
        ]
        result = fuse(opinions, _prior())
        assert any("uncertainty" in n.lower() for n in result.notes)


# ---------------------------------------------------------------------------
# Article agreement
# ---------------------------------------------------------------------------

class TestArticleAgreement:

    def test_agreed_requires_two_citations(self):
        opinions = [
            _opinion("defender",   Verdict.MODERATE_FOR, 0.70, cited=["Art-1", "Art-2"]),
            _opinion("prosecutor", Verdict.MODERATE_FOR, 0.65, cited=["Art-2", "Art-3"]),
            _opinion("judge",      Verdict.MODERATE_FOR, 0.68, cited=["Art-3"]),
        ]
        result = fuse(opinions, _prior())
        assert "Art-2" in result.agreed_articles   # defender + prosecutor
        assert "Art-3" in result.agreed_articles   # prosecutor + judge
        assert "Art-1" not in result.agreed_articles  # defender only

    def test_all_cited_is_union(self):
        opinions = [
            _opinion("defender",   Verdict.MODERATE_FOR, 0.70, cited=["Art-1"]),
            _opinion("prosecutor", Verdict.MODERATE_FOR, 0.65, cited=["Art-2"]),
            _opinion("judge",      Verdict.MODERATE_FOR, 0.68, cited=["Art-3"]),
        ]
        result = fuse(opinions, _prior())
        assert set(result.all_cited_articles) == {"Art-1", "Art-2", "Art-3"}

    def test_no_agreed_when_all_cite_different(self):
        opinions = [
            _opinion("defender",   Verdict.MODERATE_FOR, 0.70, cited=["Art-1"]),
            _opinion("prosecutor", Verdict.MODERATE_FOR, 0.65, cited=["Art-2"]),
            _opinion("judge",      Verdict.MODERATE_FOR, 0.68, cited=["Art-3"]),
        ]
        result = fuse(opinions, _prior())
        assert result.agreed_articles == []


# ---------------------------------------------------------------------------
# Noisy retrieval note
# ---------------------------------------------------------------------------

class TestNoisyRetrievalNote:

    def test_note_added_when_prune_ratio_high(self):
        opinions = [
            _opinion("defender",   Verdict.NEUTRAL, 0.50),
            _opinion("prosecutor", Verdict.NEUTRAL, 0.50),
            _opinion("judge",      Verdict.NEUTRAL, 0.50),
        ]
        result = fuse(opinions, _prior(prune_ratio=0.70))
        assert any("noisy" in n.lower() for n in result.notes)

    def test_no_noise_note_when_prune_ratio_low(self):
        opinions = [
            _opinion("defender",   Verdict.MODERATE_FOR, 0.70),
            _opinion("prosecutor", Verdict.MODERATE_FOR, 0.68),
            _opinion("judge",      Verdict.MODERATE_FOR, 0.69),
        ]
        result = fuse(opinions, _prior(prune_ratio=0.20))
        assert not any("noisy" in n.lower() for n in result.notes)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_requires_exactly_three_opinions(self):
        opinions = [
            _opinion("defender",   Verdict.MODERATE_FOR, 0.70),
            _opinion("prosecutor", Verdict.MODERATE_FOR, 0.65),
        ]
        with pytest.raises(ValueError, match="exactly 3"):
            fuse(opinions, _prior())

    def test_four_opinions_raises(self):
        opinions = [_opinion(f"agent{i}", Verdict.NEUTRAL, 0.5) for i in range(4)]
        with pytest.raises(ValueError, match="exactly 3"):
            fuse(opinions, _prior())

    def test_confidence_level_high(self):
        opinions = [
            _opinion("defender",   Verdict.STRONG_FOR, 0.90),
            _opinion("prosecutor", Verdict.STRONG_FOR, 0.88),
            _opinion("judge",      Verdict.STRONG_FOR, 0.89),
        ]
        result = fuse(opinions, _prior(score=0.90))
        assert result.final_level == "high"

    def test_confidence_level_low(self):
        opinions = [
            _opinion("defender",   Verdict.NEUTRAL, 0.20),
            _opinion("prosecutor", Verdict.NEUTRAL, 0.22),
            _opinion("judge",      Verdict.NEUTRAL, 0.21),
        ]
        result = fuse(opinions, _prior(score=0.20))
        assert result.final_level == "low"