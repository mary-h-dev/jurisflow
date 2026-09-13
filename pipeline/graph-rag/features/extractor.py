"""
features/extractor.py — LLM-based Feature extraction from a ruling's text
                          (third graph layer)

How it works (current version — vocabulary is NOT sent in the prompt):
    1. The LLM freely (without seeing the closed vocabulary) extracts
       concept/action/role/object/fact from the ruling text — using the
       exact wording found in the text, not a paraphrase.
    2. Each extracted value (except fact, which is free-form) is mapped to
       the closest closed-vocabulary term via features/vocab_resolver.py.
       Resolution is deterministic only: exact/alias match, then
       token-containment (no fuzzy string matching, no embedding
       similarity — see vocab_resolver.py's docstring and ADR-006 for why).
       If the resolver finds no match, that feature is dropped.
    3. For every feature, an evidence quote is also requested; its
       position in the source text is located with str.find (or a fuzzy
       fallback for minor LLM rewording).

Why this change (not sending the vocabulary in the prompt)?
    The full vocabulary (several hundred terms) pushed the prompt past the
    token/TPM ceiling of most free-tier providers. The first alternative
    (lightweight retrieval via whole-case embedding) was tried and
    rejected — short, frequent terms like "خواهان" scored low similarity
    against a long paragraph and got dropped from the prompt. Current
    approach: let the LLM extract freely (prompt stays small), and match
    against the vocabulary *after* extraction, on short phrases (not the
    whole case) — which is both more accurate and keeps the prompt small.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from dotenv import load_dotenv

from features.configs import FACT_CATEGORY, LLM_MODEL, LLM_TEMPERATURE, VOCAB_CATEGORIES, get_llm_client
from features.schemas import Evidence, ExtractedFeature, FeatureExtractionResult
from features.vocab_resolver import VocabResolver

load_dotenv()

_HERE = Path(__file__).resolve().parent.parent  # graph-rag/

_client = get_llm_client()

# Safety cap on ruling text length sent to the LLM.
MAX_TEXT_CHARS = 12000
_HEAD_RATIO = 0.4  # 40% from the start of the text, 60% from the end (end matters more)

_CATEGORY_DESCRIPTIONS = "\n".join(
    f'- {c.key}: {c.label_fa} — {c.description}' for c in VOCAB_CATEGORIES.values()
)


def _truncate_keep_head_and_tail(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head_len = int(max_chars * _HEAD_RATIO)
    tail_len = max_chars - head_len
    return (
        text[:head_len]
        + "\n\n[...بخشی از میانه‌ی متن به‌خاطر طولانی بودن حذف شد...]\n\n"
        + text[-tail_len:]
    )


# NOTE: instructional text translated to English. The category
# descriptions embedded via {_CATEGORY_DESCRIPTIONS} come from
# configs.py and stay in Persian (they describe the actual Persian
# legal categories being extracted), and the ruling text itself is of
# course Persian legal text — both are content, not documentation.
def _build_prompt(ruling_text: str) -> str:
    return f"""
You are a legal assistant specialized in Iranian law. From the ruling
text below, extract legal Features according to the following
categories:

{_CATEGORY_DESCRIPTIONS}

⚠️ For concept/action/role/object: use exactly the term as it appears
in the text — do not paraphrase or summarize, write the exact form of
the word/phrase (since it will later be matched against an official
vocabulary).

For the "{FACT_CATEGORY.key}" category ({FACT_CATEGORY.label_fa}):
{FACT_CATEGORY.description}
Here you are free — write the key facts of the ruling as a short
sentence.
⚠️ The examples above (like "payment was not made" or "a forged
document was submitted") are only meant to show the *writing style* of
a fact, not a checklist that must always be filled. If one of these
examples doesn't apply in *this* ruling, don't mention it at all —
never return a Feature just because it was used as an example above,
with an empty evidence_quote and zero confidence; such cases will be
discarded entirely, so don't waste your time on them.

For *every* extracted Feature (in any category), evidence is required:
exactly the part of the ruling text this Feature was drawn from
(verbatim, without changing a single word, so its position in the text
can be located), plus a confidence number between 0 and 1.

⚠️ Critical note about evidence: the quote must **directly and
specifically** prove what is stated in "value" — not a general sentence
from the same area of the text that merely happens to be nearby. If you
cannot find precise, relevant evidence for a Feature, do not extract
that Feature at all.

⚠️ evidence_quote must never contain "..." or a summary/combination of
several separate pieces of text — it must be exactly one *contiguous*
and *complete* section of the original text, word-for-word, with
nothing omitted. If the full sentence is too long, pick only the
shortest contiguous portion of the text that on its own proves the
"value" claim — not the whole sentence truncated with "...".

⚠️ Precise guide for setting confidence (please follow exactly, not
just approximately — a constant number for every item, e.g. all 0.95,
is not acceptable):
- 0.95-1.0: the exact word/phrase in "value" appears verbatim, with no intermediary, in the evidence quote.
- 0.75-0.9: value doesn't appear directly in the quote, but its meaning is clearly and unambiguously inferable from it.
- 0.5-0.74: requires multi-step inference or context outside the quote.
- below 0.5: the connection is weak; in this case it's better not to extract this Feature at all.

Ruling text:
\"\"\"
{_truncate_keep_head_and_tail(ruling_text, MAX_TEXT_CHARS)}
\"\"\"

Return only JSON — no extra explanation — in exactly this form (any
category can be an empty list):
{{
  "concept": [{{"value": "...", "evidence_quote": "...", "confidence": 0.9}}],
  "action": [],
  "role": [],
  "object": [],
  "fact": [{{"value": "...", "evidence_quote": "...", "confidence": 0.85}}]
}}
"""


# Bidi text-direction control characters — fully invisible, so like the
# zero-width non-joiner they must be ignored during fuzzy matching too.
_BIDI_CONTROL_CHARS = "\u200e\u200f\u202a\u202b\u202c\u202d\u202e"
_SPLIT_SEPARATOR_RE = re.compile(r"[\s\u200c" + _BIDI_CONTROL_CHARS + r"]+")


def _fuzzy_find(quote: str, full_text: str) -> tuple[int | None, int | None]:
    tokens = [t for t in _SPLIT_SEPARATOR_RE.split(quote.strip()) if t]
    if not tokens:
        return None, None
    separator_pattern = r"[\s\u200c" + _BIDI_CONTROL_CHARS + r"]*"
    pattern = separator_pattern.join(re.escape(t) for t in tokens)
    m = re.search(pattern, full_text)
    if not m:
        return None, None
    return m.start(), m.end()


def _locate_evidence(quote: str, full_text: str, confidence: float) -> Evidence:
    quote = (quote or "").strip()
    if not quote:
        return Evidence(quote="", start_char=None, end_char=None, confidence=confidence)

    idx = full_text.find(quote)
    if idx != -1:
        return Evidence(quote=quote, start_char=idx, end_char=idx + len(quote), confidence=confidence)

    start, end = _fuzzy_find(quote, full_text)
    if start is not None:
        return Evidence(quote=quote, start_char=start, end_char=end, confidence=confidence)

    if "..." in quote or "…" in quote:
        print(f"  ⚠️ evidence contains '...' and was not found — likely a model paraphrase: «{quote[:50]}...»")
        return Evidence(quote=quote, start_char=None, end_char=None, confidence=confidence)

    return Evidence(quote=quote, start_char=None, end_char=None, confidence=confidence)


def extract_ruling(
    ruling_id: str, ruling_text: str, resolver: VocabResolver, retries: int = 2
) -> FeatureExtractionResult:
    """
    resolver: a VocabResolver instance — must be created *once* in
    main_features.py and reused across all cases (not created per case).
    """
    prompt = _build_prompt(ruling_text)

    last_error = None
    for attempt in range(retries + 1):
        try:
            response = _client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=LLM_TEMPERATURE,
            )
            text = response.choices[0].message.content.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            data = json.loads(text)
            return _to_result(ruling_id, ruling_text, data, resolver)
        except Exception as e:  # noqa: BLE001
            last_error = e
            if attempt < retries:
                print(f"  ⏳ Feature extraction error for {ruling_id} "
                      f"(attempt {attempt + 1}/{retries}): {e}")
                time.sleep(3)

    raise RuntimeError(f"❌ Feature extraction failed for {ruling_id}: {last_error}")


def _to_result(
    ruling_id: str, ruling_text: str, data: dict, resolver: VocabResolver
) -> FeatureExtractionResult:
    result = FeatureExtractionResult(ruling_id=ruling_id)
    target_lists = {
        "concept": result.concepts,
        "action": result.actions,
        "role": result.roles,
        "object": result.objects,
        "fact": result.facts,
    }

    for category_key, target_list in target_lists.items():
        is_free_category = category_key == "fact"  # fact is not resolved against the vocabulary
        seen_values = set()  # initialized here only — before the inner loop

        for item in data.get(category_key, []):
            raw_value = str(item.get("value", "")).strip()
            if not raw_value:
                continue

            if is_free_category:
                value = raw_value
            else:
                value = resolver.resolve(raw_value, category_key)
                if value is None:
                    print(f"  ⚠️ «{raw_value}» did not match any closed-vocabulary term "
                          f"— dropped ({category_key})")
                    continue

            if value in seen_values:  # duplicate check here, after resolving to canonical value
                continue
            seen_values.add(value)

            raw_quote = str(item.get("evidence_quote", "")).strip()
            if not raw_quote:
                print(f"  🚫 [{ruling_id}] dropped, no evidence ({category_key}): «{value}»")
                continue

            evidence = _locate_evidence(raw_quote, ruling_text, float(item.get("confidence", 0.0)))
            _warn_if_evidence_unrelated(ruling_id, category_key, value, evidence.quote)
            target_list.append(ExtractedFeature(category=category_key, value=value, evidence=evidence))

    return result


def _warn_if_evidence_unrelated(ruling_id: str, category_key: str, value: str, quote: str):
    value_tokens = [t for t in re.split(r"[\s\u200c]+", value) if len(t) > 1]
    if value_tokens and quote and not any(t in quote for t in value_tokens):
        print(f"  🔎 [{ruling_id}] possible evidence/value mismatch "
              f"({category_key}=«{value}»): «{quote[:60]}...»")