from __future__ import annotations

import json
import logging

from apps.legal_agents.nodes._base import (
    BaseDeliberationAgent,
    _format_applicable_articles,
    _RESPONSE_SCHEMA,
    _SHARED_CONTEXT,
    _get_client,
)
from apps.legal_agents.schemas import AgentOpinion, AuditorOut, Verdict

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an impartial Iranian court judge with 20+ years of experience.
You do not favour either party. Evaluate objectively based on Iranian law
and the verified articles only. Your verdict must reflect the most likely court outcome."""


class JudgeAgent(BaseDeliberationAgent):
    role          = "judge"
    system_prompt = _SYSTEM_PROMPT
    temperature   = 0.1

    def _role_instruction(self) -> str:
        return """\
=== YOUR TASK ===
1. Review both parties' positions shown above.
2. Evaluate objectively — which arguments hold under Iranian law?
3. Assign a verdict reflecting the most likely court outcome.
4. Note the 1-2 key legal pivots that determined your ruling."""

    def run_with_debate(
        self,
        auditor_out:        AuditorOut,
        defender_opinion:   AgentOpinion,
        prosecutor_opinion: AgentOpinion,
    ) -> AgentOpinion:
        logger.info("[judge] starting — query=%r", auditor_out.query[:80])

        debate = (
            "=== DEFENCE POSITION ===\n"
            f"Verdict   : {defender_opinion.verdict.value}\n"
            f"Confidence: {defender_opinion.confidence:.2f}\n"
            "Arguments :\n"
            + "\n".join(f"  • {a}" for a in defender_opinion.arguments)
            + "\n\n=== PROSECUTION POSITION ===\n"
            f"Verdict   : {prosecutor_opinion.verdict.value}\n"
            f"Confidence: {prosecutor_opinion.confidence:.2f}\n"
            "Arguments :\n"
            + "\n".join(f"  • {a}" for a in prosecutor_opinion.arguments)
            + "\n\n"
        )

        user_message = (
            _SHARED_CONTEXT.format(
                query       = auditor_out.query,
                articles    = _format_applicable_articles(auditor_out),
                prior_score = auditor_out.confidence.score,
                prior_level = auditor_out.confidence.level,
                prune_ratio = auditor_out.confidence.prune_ratio,
            )
            + debate
            + self._role_instruction()
            + "\n\n"
            + _RESPONSE_SCHEMA
        )

        raw = _get_client().complete(
            model       = self.model,
            temperature = self.temperature,
            messages    = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": user_message},
            ],
        )
        raw  = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)

        opinion = AgentOpinion(
            role           = self.role,
            verdict        = Verdict(data["verdict"]),
            confidence     = float(data["confidence"]),
            position       = data.get("position", ""),
            arguments      = data.get("arguments", []),
            cited_articles = data.get("cited_articles", []),
            risks          = data.get("risks", []),
        )
        logger.info(
            "[judge] verdict=%s confidence=%.2f",
            opinion.verdict.value, opinion.confidence,
        )
        return opinion


def judge(
    auditor_out:        AuditorOut,
    defender_opinion:   AgentOpinion,
    prosecutor_opinion: AgentOpinion,
) -> AgentOpinion:
    return JudgeAgent().run_with_debate(auditor_out, defender_opinion, prosecutor_opinion)