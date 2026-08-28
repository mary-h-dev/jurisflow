from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from .checklist_builder import ArticleEvidenceBundle
from core.llm_client import get_client
from .apps_auditor_config import AUDITOR_MODEL
from .schemas import ChecklistItemOut
from pathlib import Path

logger = logging.getLogger(__name__)


_CHECKLIST_DB_PATH = Path(__file__).resolve().parent / "data" / "checklist_db.json"
_checklist_db_cache: dict[str, list[str]] | None = None


_MAX_ARTICLE_TEXT_CHARS = 1200
_MAX_EVIDENCE_TEXT_CHARS = 300



def _load_checklist_db() -> dict[str, list[str]]:
    """
    Loads the offline-generated fixed checklist for annotation-dataset
    articles (see build_checklist_db.py). Cached in-process -- read once,
    reused for every verify_article call in this run. Missing file or
    parse error falls back to an empty dict, which just means every
    article goes through the normal (LLM-generates-its-own-checklist)
    path -- never a hard failure.
    """
    global _checklist_db_cache
    if _checklist_db_cache is not None:
        return _checklist_db_cache

    if not _CHECKLIST_DB_PATH.exists():
        logger.warning(f"checklist_db.json not found at {_CHECKLIST_DB_PATH}, "
                        f"all articles will use LLM-generated checklists")
        _checklist_db_cache = {}
        return _checklist_db_cache

    try:
        _checklist_db_cache = json.loads(_CHECKLIST_DB_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"checklist_db.json unreadable ({e}), falling back to LLM generation")
        _checklist_db_cache = {}

    return _checklist_db_cache



def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + " ..."


_PROMPT_TEMPLATE = """
You are a precise, cautious legal auditor. Your task is to audit whether one specific statutory article applies to the facts of a case.

=== CASE FACTS (query) ===
{query}

=== ARTICLE UNDER REVIEW: {article_ref} ===
{article_text}

=== SUPPORTING EVIDENCE (related rulings and extracted facts) ===
{evidence_block}

GOLDEN RULE 1 (critical): The checklist must be specific to THIS article, not a generic summary of the case facts.
Every condition must be derived from a legal element actually stated in this article's own text (a qualifier, an exception, a material or mental element the article itself names).
If a checklist you produce could be reused unchanged for a different article -- because it only restates generic case facts like "a fight occurred" or "the defendant took part" -- it is wrong. Rebuild it and find the elements specific to this article.

Example of the expected precision:
An article on legitimate self-defense -> a condition should read something like "the defense was proportionate to the established danger" or "no lawful alternative was available" -- not merely "the defendant acted in self-defense".
An article on aiding and abetting -> a condition should read something like "the defendant's assistance or incitement preceded the underlying offense" -- not merely "the defendant was present at the incident".

GOLDEN RULE 2 (critical -- condition polarity): Always phrase every condition so that satisfied=true means "this factor favors the article applying" -- never the reverse.
- If the article's text contains an exception or bar (e.g. "... unless X" or "... except where X"), do not write that exception verbatim as the condition. Invert it into a positive statement: instead of "the defendant has a prior conviction" (confusing when false), write "the prior-conviction exception does not apply", and set satisfied accordingly.
- If the article imposes a legal duty on the court/authority, and the case is precisely about a FAILURE of that duty (e.g. grounds for retrial that open up when a duty was not fulfilled), phrase the condition from the angle of "is this ground available for the present action [e.g. retrial]", not "was that duty fulfilled". Bad example: "the court is obligated to impose a substitute penalty" with satisfied=false when the court didn't -- this is inverted. Good example: "the court failed to impose the legally required substitute penalty" with satisfied=true.
- Before finalizing each condition, ask yourself: "if satisfied=true, does this genuinely move the case closer to the article applying?" If the answer is no, invert the wording.


GOLDEN RULE 3 (critical -- stay inside this article's own elements): The checklist's necessary conditions must come ONLY from the legal elements this article's own text actually requires (subject matter, actors, thresholds, procedural triggers, etc). Supporting evidence may be used to judge whether those elements are met -- never to introduce a NEW condition this article's text does not itself impose.
A competing or alternative legal basis mentioned in the evidence (e.g. a different article, or a contractual clause that might displace this one) is NEVER a condition of THIS article. Do not add a condition like "the parties did not otherwise agree on damages" or "no other legal basis applies" just because supporting evidence discusses a competing provision -- that is a separate legal question for deliberation, not an element this article's own text requires. If you catch yourself writing a condition phrased as "X does not apply instead" or "no other Y exists", stop -- that is almost always this exact mistake. Delete it.This is the single most common mistake in this task -- watch for it specifically.

YOUR TASK:
1. First, decide topical relevance: is this article about the same legal issue the query raises at all -- regardless of whether it ultimately applies? An article defining a threshold the case falls short of (e.g. "at least 3 participants required" when the case has 2) is topically relevant even though it will turn out inapplicable. An article about an unrelated matter (e.g. weapons possession, when the query is about phone harassment) is not topically relevant, even if it was retrieved.
2. Build a diagnostic checklist of the legal conditions required for THIS article to apply (per all three golden rules above) -- based only on the article text and the evidence shown, not outside knowledge. Every condition must trace back to a specific phrase in the article text itself -- if you cannot point to the words in the article that a condition comes from, it does not belong on this checklist.
   LIMIT: at most 3 necessary conditions. Most articles have 1-2 core elements; if you have written more than 3, you have almost certainly split one element into several, or smuggled in a condition from the case facts or a competing article rather than this article's own text. Merge or cut down to the 3 most essential.
3. Evaluate each condition against the case facts and evidence above, and determine whether it is satisfied.
4. For each condition, determine whether it is a necessary condition for the article to apply, or merely contextual/explanatory.
5. SELF-CHECK before finalizing: for every necessary condition, re-read the article text under review and confirm you can point to the specific words it comes from. If a condition instead comes from something the evidence says about a DIFFERENT article, a competing legal basis, or a general fact pattern -- delete it. This check overrides everything else: it is better to end up with 1 necessary condition than to keep one you cannot source to this article's own text.


LANGUAGE REQUIREMENT (critical): write every "condition" string in Persian (فارسی), using proper legal Persian terminology -- never English or a mix of languages within one checklist. This is because the output must stay directly comparable to Persian-language gold annotations and be read by Persian-speaking legal reviewers.

Return ONLY a single JSON object with this structure, no extra text. The value of satisfied must always be exactly true or false (never "unknown" or anything else); if uncertain, pick true or false based on your best reading of the evidence available:
{{
  "topically_relevant": true,
  "checklist": [
    {{"condition": "...", "necessary": true, "satisfied": true}}
  ]
}}
"""



_EVALUATION_ONLY_PROMPT_TEMPLATE = """
You are a precise, cautious legal auditor. Your task is to evaluate whether one specific statutory article applies to the facts of a case, using a FIXED checklist of conditions -- you do not create the checklist, only evaluate it.

=== CASE FACTS (query) ===
{query}

=== ARTICLE UNDER REVIEW: {article_ref} ===
{article_text}

=== SUPPORTING EVIDENCE (related rulings and extracted facts) ===
{evidence_block}

=== FIXED CHECKLIST (do not add, remove, or reword any condition) ===
{conditions_block}

GOLDEN RULE (critical -- polarity): For each condition, satisfied=true must always mean "this factor favors the article applying." If a condition is phrased as an exception or bar, satisfied=true means the exception does NOT apply (i.e. the bar is absent).

YOUR TASK:
1. Decide topical relevance: is this article about the same legal issue the query raises at all -- regardless of whether it ultimately applies?
2. For EACH of the fixed conditions listed above, evaluate it against the case facts and evidence, and determine whether it is satisfied. Every condition in the fixed checklist is necessary=true by construction -- do not change this.
3. Do NOT invent any additional condition, even if the evidence discusses a competing article or an alternative legal basis. Evaluate only the conditions given.

LANGUAGE REQUIREMENT: keep every "condition" string EXACTLY as given in the fixed checklist above -- do not translate, reword, or paraphrase it.

Return ONLY a single JSON object with this structure, no extra text. The value of satisfied must always be exactly true or false:
{{
  "topically_relevant": true,
  "checklist": [
    {{"condition": "...", "necessary": true, "satisfied": true}}
  ]
}}
"""



class VerificationError(Exception):
    """Base class for anything that stops verify_article from returning a
    usable checklist. Callers that don't need to distinguish the cause
    (e.g. AuditorService, which just skips the article) can catch this."""


class AuditorAPIError(VerificationError):
    """The LLM call itself failed: network error, invalid/missing API
    key, rate limit exhausted after the SDK's own retries, timeout, etc.
    Verification did not run at all -- distinct from the model running
    and returning something unusable (see AuditorParseError)."""

    def __init__(self, article_ref: str, cause: Exception):
        self.article_ref = article_ref
        self.cause = cause
        super().__init__(
            f"LLM call failed for {article_ref}: {type(cause).__name__}: {cause}"
        )


class AuditorParseError(VerificationError):
    """The LLM call succeeded but the response was not valid JSON, or
    didn't match the expected checklist schema after _MAX_PARSE_RETRIES
    attempts. Includes a truncated copy of the raw response to help
    debug prompt/schema drift."""

    def __init__(self, article_ref: str, raw_text: str, cause: Exception):
        self.article_ref = article_ref
        self.raw_text = raw_text
        self.cause = cause
        super().__init__(
            f"Could not parse response for {article_ref}: "
            f"{type(cause).__name__}: {cause}\n"
            f"raw response (truncated): {raw_text[:500]!r}"
        )


_MAX_PARSE_RETRIES = 2


@dataclass
class VerificationResult:
    topically_relevant: bool
    checklist: list[ChecklistItemOut]


def verify_article(query: str, bundle: ArticleEvidenceBundle) -> VerificationResult:
    """
    Single combined LLM call per article: decides topical relevance,
    builds the diagnostic checklist for `bundle.article_ref`, and
    evaluates every condition against `query` and
    `bundle.supporting_texts` in one pass. Checklist generation and
    item-wise verification are merged into one request to halve the
    number of LLM calls versus a two-step pipeline.

    Uses the shared LLMClient (see client.py, initialised in apps.py)
    pointed at OpenRouter, with model/temperature from config.AUDITOR_MODEL.

    Raises AuditorAPIError if the API call itself fails, or
    AuditorParseError if every parse attempt fails. Both subclass
    VerificationError.
    """
    evidence_block = "\n".join(
        f"- {_truncate(t, _MAX_EVIDENCE_TEXT_CHARS)}" for t in bundle.supporting_texts
    )
    evidence_block = evidence_block or "(شواهد پشتیبان مستقیمی یافت نشد)"

    article_text = bundle.article_text or "(متن ماده در دسترس نیست)"
    article_text = _truncate(article_text, _MAX_ARTICLE_TEXT_CHARS)


    predefined_conditions = _load_checklist_db().get(bundle.article_ref)

    if predefined_conditions:
        conditions_block = "\n".join(f"- {c}" for c in predefined_conditions)
        prompt = _EVALUATION_ONLY_PROMPT_TEMPLATE.format(
            query=query,
            article_ref=bundle.article_ref,
            article_text=article_text,
            evidence_block=evidence_block,
            conditions_block=conditions_block,
        )
    else:
        prompt = _PROMPT_TEMPLATE.format(
            query=query,
            article_ref=bundle.article_ref,
            article_text=article_text,
            evidence_block=evidence_block,
        )



    messages = [{"role": "user", "content": prompt}]

    last_parse_error: Exception | None = None
    last_raw_text = ""

    for attempt in range(_MAX_PARSE_RETRIES):
        try:
            raw_text = get_client().complete(
                model=AUDITOR_MODEL.model,
                temperature=AUDITOR_MODEL.temperature,
                messages=messages,
                max_tokens=AUDITOR_MODEL.max_tokens,
            )
        except Exception as e:
            logger.error(f"LLM call failed for {bundle.article_ref}: {e}")
            raise AuditorAPIError(bundle.article_ref, e) from e

        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        last_raw_text = raw_text

        try:
            raw = json.loads(raw_text)
            if not isinstance(raw, dict):
                raise ValueError(f"expected a JSON object, got {type(raw).__name__}")

            items = raw.get("checklist")
            if not isinstance(items, list) or not items:
                raise ValueError(f"expected a non-empty 'checklist' list, got {items!r}")

            topically_relevant = raw.get("topically_relevant")
            if not isinstance(topically_relevant, bool):
                logger.warning(
                    f"topically_relevant missing/invalid for {bundle.article_ref} "
                    f"({topically_relevant!r}), defaulting to True (conservative -- "
                    f"don't silently drop an article from downstream context)"
                )
                topically_relevant = True

            checklist: list[ChecklistItemOut] = []
            for item in items:
                try:
                    checklist.append(ChecklistItemOut(**item))
                except Exception as item_error:
                    # One malformed item doesn't need to discard an
                    # otherwise-usable checklist -- skip just that item.
                    logger.warning(
                        f"Skipping malformed checklist item for {bundle.article_ref}: "
                        f"{item_error} -- item was {item!r}"
                    )
            if not checklist:
                raise ValueError("every checklist item failed to parse")
            return VerificationResult(topically_relevant=topically_relevant, checklist=checklist)
        except Exception as e:
            last_parse_error = e
            logger.warning(
                f"Parse attempt {attempt + 1}/{_MAX_PARSE_RETRIES} failed for "
                f"{bundle.article_ref}: {e} -- retrying call"
                if attempt + 1 < _MAX_PARSE_RETRIES else
                f"Auditor response parsing failed for {bundle.article_ref}: {e}"
            )

    raise AuditorParseError(bundle.article_ref, last_raw_text, last_parse_error)