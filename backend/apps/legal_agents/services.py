"""
DeliberationService — public interface for the legal-agents layer.

Accepts AuditorOut from apps.auditor.schemas directly.

min_articles fallback (added after calibration showed gold_kept_by_auditor=0.68):
    When the Auditor prunes aggressively and fewer than MIN_ARTICLES applicable
    articles remain, topically_relevant articles (is_applicable=False but
    topically_relevant=True) are added back in descending auditor_confidence
    order until the threshold is met. This prevents gold articles from being
    silently dropped before agents see them.

    Threshold set to 3 based on calibration data: 32% of samples had at least
    one gold article pruned, exceeding the 15% threshold set as the decision
    criterion for adding this fallback.
"""

from __future__ import annotations

import copy
import logging

from apps.legal_agents.checklist_builder import build_checklist
from apps.legal_agents.fusion.deterministic import fuse
from apps.legal_agents.nodes.defender import DefenderAgent
from apps.legal_agents.nodes.judge import JudgeAgent
from apps.legal_agents.nodes.prosecutor import ProsecutorAgent
from apps.legal_agents.schemas import (
    AgentOpinion,
    DeliberationOut,
    Verdict,
)

logger = logging.getLogger(__name__)

# Minimum number of articles passed to agents.
# When fewer applicable articles survive Auditor pruning, topically_relevant
# articles are added as fallback to avoid silently dropping gold evidence.
MIN_ARTICLES = 3


def _build_agent_input(auditor_out, min_articles: int = MIN_ARTICLES):
    """
    Returns a copy of auditor_out with a potentially expanded verified_articles
    list. If fewer than min_articles are applicable, adds topically_relevant
    articles (sorted by auditor_confidence desc) until the threshold is met.

    The original auditor_out is never mutated.
    """
    applicable = [a for a in auditor_out.verified_articles if a.is_applicable]

    if len(applicable) >= min_articles:
        return auditor_out  # no change needed

    # How many fallback articles do we need?
    deficit = min_articles - len(applicable)

    topical_only = sorted(
        [a for a in auditor_out.verified_articles
         if not a.is_applicable and getattr(a, "topically_relevant", False)],
        key=lambda x: x.auditor_confidence,
        reverse=True,
    )

    fallback = topical_only[:deficit]
    if fallback:
        logger.info(
            "[deliberation] min_articles fallback: applicable=%d < %d, "
            "adding %d topically_relevant articles",
            len(applicable), min_articles, len(fallback),
        )

    # Build a new AuditorOut with expanded articles — don't mutate the original
    import copy as _copy
    expanded = _copy.copy(auditor_out)
    # Pydantic models are immutable — rebuild with updated field
    expanded = auditor_out.model_copy(
        update={"verified_articles": auditor_out.verified_articles + fallback}
    )
    return expanded


class DeliberationService:

    def run(self, auditor_out, session_id: str = "") -> DeliberationOut:
        # Apply min_articles fallback before passing to agents
        agent_input = _build_agent_input(auditor_out)

        completed:          list[str]           = []
        defender_opinion:   AgentOpinion | None = None
        prosecutor_opinion: AgentOpinion | None = None
        judge_opinion:      AgentOpinion | None = None

        # ── defender ─────────────────────────────────────────────────────
        try:
            defender_opinion = DefenderAgent().run(agent_input)
            completed.append("defender")
        except Exception as exc:
            logger.error("[deliberation] defender failed: %s", exc, exc_info=True)

        # ── prosecutor ───────────────────────────────────────────────────
        try:
            prosecutor_opinion = ProsecutorAgent().run(agent_input)
            completed.append("prosecutor")
        except Exception as exc:
            logger.error("[deliberation] prosecutor failed: %s", exc, exc_info=True)

        # ── judge ────────────────────────────────────────────────────────
        try:
            if defender_opinion and prosecutor_opinion:
                judge_opinion = JudgeAgent().run_with_debate(
                    agent_input, defender_opinion, prosecutor_opinion
                )
            else:
                logger.warning("[deliberation] judge running without full debate")
                judge_opinion = JudgeAgent().run(agent_input)
            completed.append("judge")
        except Exception as exc:
            logger.error("[deliberation] judge failed: %s", exc, exc_info=True)

        # ── fusion ───────────────────────────────────────────────────────
        opinions = [op for op in [defender_opinion, prosecutor_opinion, judge_opinion]
                    if op is not None]
        fusion = None
        error  = None

        if len(opinions) >= 2:
            if len(opinions) == 2:
                filler            = copy.deepcopy(opinions[-1])
                filler.role       = "filler"
                filler.verdict    = Verdict.NEUTRAL
                filler.confidence = 0.5
                opinions          = opinions + [filler]
                logger.warning("[deliberation] padding fusion to 3 — one agent missing")
            try:
                fusion = fuse(opinions, auditor_out.confidence)
                completed.append("fusion")
            except Exception as exc:
                error = f"fusion failed: {exc}"
                logger.error("[deliberation] %s", error, exc_info=True)
        else:
            error = "insufficient agent opinions for fusion (need >= 2)"
            logger.error("[deliberation] %s", error)

        applicable = [a for a in auditor_out.verified_articles if a.is_applicable]

        result = DeliberationOut(
            session_id          = session_id,
            defender_opinion    = defender_opinion,
            prosecutor_opinion  = prosecutor_opinion,
            judge_opinion       = judge_opinion,
            fusion              = fusion,
            applicable_articles = [a.model_dump() for a in applicable],
            auditor_confidence  = auditor_out.confidence.model_dump(),
            completed_nodes     = completed,
            error               = error,
        )
        result.checklist = build_checklist(result)
        return result


deliberation_service = DeliberationService()