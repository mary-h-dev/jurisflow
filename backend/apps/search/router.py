from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from core.llm_client import get_client
from .apps_search_config import ROUTER_MODEL

logger = logging.getLogger(__name__)

_VALID_CHANNELS = {"feature", "ruling", "article"}
_VALID_CASE_TYPES = {"حقوقی", "کیفری"}


_SYSTEM_PROMPT = """You are a query router for a Persian legal search system covering Iranian civil and criminal law.

Given a user query written in Persian, return a single JSON object with the fields below, and nothing else — no markdown, no explanation, no code fences.

{
  "rewritten_query": string,        // the query rewritten in Persian: disambiguated, complete, precise legal terms
  "channels": array of strings,     // subset of ["feature", "ruling", "article"]
  "routing_confidence": float,      // 0.0 to 1.0, how confident you are in this channel selection
  "intent": string,                 // short label, e.g. "article_lookup", "legal_problem", "precedent_search"
  "case_type_hint": string or null, // must be exactly "حقوقی" or "کیفری", or null if unclear
  "ambiguity_flag": boolean         // true only if the query is too short or incomplete to determine intent
}

Channel selection guide:
- "feature": REQUIRED whenever the query names a specific legal concept, role, or
  action that depends on case facts to apply (e.g. نشوز, عسر و حرج, اکراه, تمکین,
  دفاع مشروع). These concepts are indexed separately from article text because
  their applicability is fact-dependent and interpreted case-by-case — the
  article text alone does not show how courts apply the concept.
- "ruling": user wants similar case rulings or judicial precedent.
- "article": user wants the text of a specific statutory article or provision.

Most real legal questions (not simple "what does article X say" lookups)
require BOTH "feature" and "article" together, because the user is asking
whether a named legal concept applies to a situation, which requires both
the statutory text AND how courts have factually applied that concept.

Example:
  query: "زوجه به دلیل نشوز آیا حق نفقه دارد؟"
  correct channels: ["feature", "ruling", "article"]
  wrong channels: ["article", "ruling"]  // missing "feature" — نشوز is a
    fact-dependent legal concept, not just an article reference

IMPORTANT — "ruling" is the rarest correct channel. Iranian courts are a
civil-law (statutory) system, not a precedent-based one: the vast majority
of legal reasoning is derived directly from statutory article text, not from
binding case-law (رأی وحدت رویه). Only select "ruling" when the query is
explicitly about finding similar precedent cases or when the legal question
is known to hinge on a specific binding judicial interpretation that is not
recoverable from the article text alone. Do NOT include "ruling" by default
just because a query is complex or fact-heavy — most complex legal questions
are still resolved with "feature" + "article" only.

Example (ruling NOT needed, despite being a substantial fact-heavy question):
  query: "در تصادف رانندگی که خودم مقصر بودم، همسرم فوت کرد. آیا به عنوان
    وارث می‌توانم دیه فوت او را مطالبه کنم؟"
  correct channels: ["feature", "article"]
  wrong channels: ["feature", "ruling", "article"]  // "ruling" added
    reflexively; the answer is fully determined by statutory text (قتل
    غیرعمد و حرمان قاتل از دیه), no binding precedent is needed.

ambiguity_flag should be true only for queries that lack enough content to
determine intent (e.g. a single bare term), not for queries that are merely
long or informally phrased.

Respond with the JSON object only.
"""

# The model can still wrap JSON in a ```json ... ``` fence occasionally.
# Strip it defensively before parsing rather than trusting the response
# format alone.
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class RoutingResult:
    rewritten_query:    str
    channels:            list[str]
    routing_confidence:  float
    intent:              str
    case_type_hint:      str | None
    ambiguity_flag:      bool
    raw_ok:              bool = True


def route_query(query: str) -> RoutingResult:
    try:
        raw = get_client().complete(
            model=ROUTER_MODEL.model,
            temperature=ROUTER_MODEL.temperature,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            max_tokens=ROUTER_MODEL.max_tokens,
        )
        data = json.loads(_strip_code_fence(raw))
        return _parse(data, query)
    except Exception as e:
        logger.warning(f"Router failed, falling back to all channels: {e}", exc_info=True)
        return RoutingResult(
            rewritten_query=query,
            channels=["feature", "ruling", "article"],
            routing_confidence=0.0,
            intent="unknown",
            case_type_hint=None,
            ambiguity_flag=True,
            raw_ok=False,
        )


def _strip_code_fence(raw: str) -> str:
    return _CODE_FENCE_RE.sub("", raw.strip()).strip()


def _parse(data: dict, original_query: str) -> RoutingResult:
    channels = [c for c in data.get("channels", []) if c in _VALID_CHANNELS]
    if not channels:
        channels = ["feature", "ruling", "article"]

    case_type_hint = data.get("case_type_hint")
    if case_type_hint not in _VALID_CASE_TYPES:
        case_type_hint = None

    return RoutingResult(
        rewritten_query=data.get("rewritten_query") or original_query,
        channels=channels,
        routing_confidence=float(data.get("routing_confidence", 0.5)),
        intent=data.get("intent", "unknown"),
        case_type_hint=case_type_hint,
        ambiguity_flag=bool(data.get("ambiguity_flag", False)),
        raw_ok=True,
    )