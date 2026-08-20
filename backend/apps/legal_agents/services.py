"""
DeliberationService — public interface for the legal-agents layer.

Usage:
    from apps.legal_agents.services import deliberation_service
    result = deliberation_service.run(auditor_out)
"""

from __future__ import annotations

import copy
import logging

from apps.legal_agents.fusion.deterministic import fuse
from apps.legal_agents.nodes.defender import DefenderAgent
from apps.legal_agents.nodes.judge import JudgeAgent
from apps.legal_agents.nodes.prosecutor import ProsecutorAgent
from apps.legal_agents.schemas import (
    AgentOpinion,
    AuditorOut,
    DeliberationOut,
    Verdict,
)

logger = logging.getLogger(__name__)


class DeliberationService:
    """
    Runs the three-agent deliberation pipeline on a completed AuditorOut.

    Pipeline:
        defender ──┐
                   ├──► judge ──► deterministic fusion ──► DeliberationOut
        prosecutor ┘

    defender and prosecutor are independent; judge sees both before ruling.
    fusion is fully deterministic — no LLM call.
    """

    def run(self, auditor_out: AuditorOut, session_id: str = "") -> DeliberationOut:
        completed: list[str] = []
        defender_opinion:   AgentOpinion | None = None
        prosecutor_opinion: AgentOpinion | None = None
        judge_opinion:      AgentOpinion | None = None

        # ── defender ─────────────────────────────────────────────────────────
        try:
            defender_opinion = DefenderAgent().run(auditor_out)
            completed.append("defender")
        except Exception as exc:
            logger.error("[deliberation] defender failed: %s", exc, exc_info=True)

        # ── prosecutor ───────────────────────────────────────────────────────
        try:
            prosecutor_opinion = ProsecutorAgent().run(auditor_out)
            completed.append("prosecutor")
        except Exception as exc:
            logger.error("[deliberation] prosecutor failed: %s", exc, exc_info=True)

        # ── judge ────────────────────────────────────────────────────────────
        try:
            if defender_opinion and prosecutor_opinion:
                judge_opinion = JudgeAgent().run_with_debate(
                    auditor_out, defender_opinion, prosecutor_opinion
                )
            else:
                logger.warning("[deliberation] judge running without full debate")
                judge_opinion = JudgeAgent().run(auditor_out)
            completed.append("judge")
        except Exception as exc:
            logger.error("[deliberation] judge failed: %s", exc, exc_info=True)

        # ── fusion ───────────────────────────────────────────────────────────
        opinions = [op for op in [defender_opinion, prosecutor_opinion, judge_opinion]
                    if op is not None]
        fusion = None
        error  = None

        if len(opinions) >= 2:
            if len(opinions) == 2:
                # One agent failed — pad with a neutral filler so fuse() gets 3
                filler           = copy.deepcopy(opinions[-1])
                filler.role      = "filler"
                filler.verdict   = Verdict.NEUTRAL
                filler.confidence = 0.5
                opinions         = opinions + [filler]
                logger.warning("[deliberation] padding fusion to 3 — one agent missing")
            try:
                fusion = fuse(opinions, auditor_out.confidence)
                completed.append("fusion")
            except Exception as exc:
                error = f"fusion failed: {exc}"
                logger.error("[deliberation] %s", error, exc_info=True)
        else:
            error = "insufficient agent opinions for fusion (need ≥ 2)"
            logger.error("[deliberation] %s", error)

        applicable = [a for a in auditor_out.verified_articles if a.is_applicable]

        return DeliberationOut(
            session_id          = session_id,
            defender_opinion    = defender_opinion,
            prosecutor_opinion  = prosecutor_opinion,
            judge_opinion       = judge_opinion,
            fusion              = fusion,
            applicable_articles = applicable,
            auditor_confidence  = auditor_out.confidence,
            completed_nodes     = completed,
            error               = error,
        )


deliberation_service = DeliberationService()