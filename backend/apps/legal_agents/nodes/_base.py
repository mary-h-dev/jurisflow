"""
Shared scaffolding for the three deliberation agents.

The LLM client is sourced from core.llm_client (shared with apps.auditor
and apps.search) -- no per-app client construction here.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod

from core.llm_client import ModelConfig, get_client

from apps.legal_agents.llm_config import DELIBERATION_MODEL
from apps.legal_agents.schemas import AgentOpinion, AuditorOut, Verdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _format_articles(auditor_out: AuditorOut) -> str:
    lines: list[str] = []
    for art in auditor_out.verified_articles:
        if not art.is_applicable and not getattr(art, "topically_relevant", False):
            continue
        if art.is_applicable:
            status    = "APPLICABLE"
            satisfied = [c.condition for c in art.checklist if c.satisfied]
            detail    = f"Satisfied conditions: {'; '.join(satisfied) or 'none'}"
        else:
            status = "NOT APPLICABLE"
            failed = [c.condition for c in art.checklist if c.necessary and not c.satisfied]
            detail = f"Failed condition(s): {'; '.join(failed) or 'see checklist'}"
        lines.append(
            f"  [{status}] {art.article_ref}  (auditor_confidence: {art.auditor_confidence:.2f})\n"
            f"    {detail}"
        )
    return "\n".join(lines) if lines else "  (no relevant articles found)"


_SHARED_CONTEXT = """\
=== LEGAL QUERY ===
{query}

=== AUDITOR-VERIFIED ARTICLES ===
Each article is labelled APPLICABLE or NOT APPLICABLE.
- Cite APPLICABLE articles as supporting evidence.
- Use NOT APPLICABLE articles to show why a charge or claim does NOT hold.
{articles}

=== RETRIEVAL QUALITY ===
Prior confidence : {prior_score:.2f}  ({prior_level})
Prune ratio      : {prune_ratio:.0%} of retrieved articles were rejected

"""

_RESPONSE_SCHEMA = """\
Respond with a single JSON object — no markdown fences, no extra keys:
{
  "verdict": "<strong_for | moderate_for | neutral | moderate_against | strong_against>",
  "confidence": <float 0.0-1.0>,
  "position": "<one sentence>",
  "arguments": ["<arg1>", "<arg2>", "<arg3>"],
  "cited_articles": ["<article_ref>"],
  "risks": ["<risk>"]
}
"""


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class BaseDeliberationAgent(ABC):
    role:         str
    system_prompt: str
    model_config: ModelConfig = DELIBERATION_MODEL

    def run(self, auditor_out: AuditorOut) -> AgentOpinion:
        logger.info("[%s] starting — query=%r", self.role, auditor_out.query[:80])

        user_message = (
            _SHARED_CONTEXT.format(
                query       = auditor_out.query,
                articles    = _format_articles(auditor_out),
                prior_score = auditor_out.confidence.score,
                prior_level = auditor_out.confidence.level,
                prune_ratio = auditor_out.confidence.prune_ratio,
            )
            + self._role_instruction()
            + "\n\n"
            + _RESPONSE_SCHEMA
        )

        raw = get_client().complete(
            model       = self.model_config.model,
            temperature = self.model_config.temperature,
            messages    = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": user_message},
            ],
            max_tokens  = self.model_config.max_tokens,
        )
        if not raw:
            raise ValueError(f"Empty response from model '{self.model_config.model}'")

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
            "[%s] verdict=%s confidence=%.2f cited=%d",
            self.role, opinion.verdict.value, opinion.confidence, len(opinion.cited_articles),
        )
        return opinion

    @abstractmethod
    def _role_instruction(self) -> str: ...