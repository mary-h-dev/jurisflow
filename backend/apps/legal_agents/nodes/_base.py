"""
Shared LLM client and base class for the three deliberation agents.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod

from openai import OpenAI

from apps.legal_agents.schemas import AgentOpinion, AuditorOut, Verdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

class LLMClient:
    """OpenAI-compatible client — pointed at OpenRouter."""

    def __init__(self, api_key: str, base_url: str) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, model: str, temperature: float, messages: list[dict]) -> str:
        response = self._client.chat.completions.create(
            model       = model,
            messages    = messages,
            temperature = temperature,
            max_tokens  = 1024,
        )
        return response.choices[0].message.content.strip()


_client: LLMClient | None = None


def init_client(client: LLMClient) -> None:
    global _client
    _client = client


def _get_client() -> LLMClient:
    if _client is None:
        raise RuntimeError("LLMClient not initialised — check apps.py AppConfig.ready()")
    return _client


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _format_applicable_articles(auditor_out: AuditorOut) -> str:
    lines: list[str] = []
    for art in auditor_out.verified_articles:
        if not art.is_applicable:
            continue
        satisfied = [c.condition for c in art.checklist if c.satisfied]
        lines.append(
            f"  • {art.article_ref}  (confidence: {art.auditor_confidence:.2f})\n"
            f"    Satisfied conditions: {'; '.join(satisfied) or 'none'}"
        )
    return "\n".join(lines) if lines else "  (no applicable articles)"


_SHARED_CONTEXT = """\
=== LEGAL QUERY ===
{query}

=== VERIFIED APPLICABLE ARTICLES ===
(Only these passed the Auditor checklist — do NOT cite anything else.)
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
    role:          str
    system_prompt: str
    model:         str  = "google/gemini-2.5-flash"
    temperature:   float = 0.2

    def run(self, auditor_out: AuditorOut) -> AgentOpinion:
        logger.info("[%s] starting — query=%r", self.role, auditor_out.query[:80])

        user_message = (
            _SHARED_CONTEXT.format(
                query       = auditor_out.query,
                articles    = _format_applicable_articles(auditor_out),
                prior_score = auditor_out.confidence.score,
                prior_level = auditor_out.confidence.level,
                prune_ratio = auditor_out.confidence.prune_ratio,
            )
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
            "[%s] verdict=%s confidence=%.2f cited=%d",
            self.role, opinion.verdict.value, opinion.confidence, len(opinion.cited_articles),
        )
        return opinion

    @abstractmethod
    def _role_instruction(self) -> str: ...