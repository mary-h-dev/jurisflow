from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from django.conf import settings

logger = logging.getLogger(__name__)

_VALID_CHANNELS = {"feature", "ruling", "article"}

# Must match the system's case_type taxonomy exactly (see database/config
# domain -> case_type mapping). These two literals are data values, not
# display text, so they are kept in Persian intentionally.
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

ambiguity_flag should be true only for queries that lack enough content to
determine intent (e.g. a single bare term), not for queries that are merely
long or informally phrased.

Respond with the JSON object only.
"""



_GROQ_MODEL = "llama-3.3-70b-versatile"
_GROQ_TIMEOUT_SECONDS = 10


@dataclass
class RoutingResult:
    rewritten_query:    str
    channels:            list[str]
    routing_confidence:  float
    intent:              str
    case_type_hint:      str | None
    ambiguity_flag:      bool
    raw_ok:              bool = True   # False if the LLM call failed and a fallback was used


def route_query(query: str) -> RoutingResult:
    try:
        raw = _call_groq(query)
        data = json.loads(raw)
        return _parse(data, query)
    except Exception as e:
        logger.warning(f"Router failed, falling back to all channels: {e}")
        # A fresh instance each time — never reuse or mutate a shared object,
        # to avoid a race condition under concurrent requests.
        return RoutingResult(
            rewritten_query=query,
            channels=["feature", "ruling", "article"],
            routing_confidence=0.0,
            intent="unknown",
            case_type_hint=None,
            ambiguity_flag=True,
            raw_ok=False,
        )


def _call_groq(query: str) -> str:
    from groq import Groq
    client = Groq(api_key=settings.GROQ_API_KEY)

    response = client.chat.completions.create(
        model=_GROQ_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
        timeout=_GROQ_TIMEOUT_SECONDS,
    )
    return response.choices[0].message.content


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