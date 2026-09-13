"""
features/vocab_resolver.py — maps a freely-extracted LLM value to the
closest closed-vocabulary term, using deterministic methods only
(no fuzzy matching, no embedding similarity).

Why no fuzzy matching (no difflib, no embedding)?
    Tested and rejected. "توقیف" (seizure of property) and "توقف"
    (coming to a stop) differ by a single character but have completely
    different meanings — difflib incorrectly matched them. To rule this
    out with a different tool too, embedding similarity (bge-m3) was
    tested directly: "توقیف" vs "توقیف خودرو" (should be similar) =
    0.889, "توقیف" vs "توقف" (should NOT be similar) = 0.846. A gap of
    only 0.043 between the correct and incorrect match means no
    threshold can reliably separate them — this is an inherent
    limitation of general-purpose embedding models on Persian
    near-homograph verbs, not a tunable parameter. See
    diagnose_confusable_pairs.py for the full test and ADR-006.

    So the guiding principle: better to correctly drop a feature than to
    snap it to a nearby-but-wrong term that silently sits in the graph —
    the same principle followed elsewhere in this project (rejecting
    evidence without a location, rejecting quotes containing "...").

Two allowed match types — both deterministic, not probabilistic:
    1. Exact / alias match: the value (after normalization) is exactly
       equal to a canonical term or one of its aliases.
    2. Token-containment: a *complete* canonical term (as a token
       sequence, not a raw substring) appears inside the extracted
       phrase. Example: "خوانده ردیف اول" contains the exact token
       "خوانده" → match. But "توقیف" never matches "توقف", because
       these are different strings, not because they're similar.

       Why at the token level, not raw substring? Because a raw
       substring check causes false positives — e.g. "رد" can appear as
       a substring inside "تجدیدنظرخواهی" without actually being that
       word. The same pattern used in features/vocab_gap_audit.py to
       solve an analogous problem ("ولی" inside "مسئولیت") is applied
       here too.

Every rejected feature (no exact or containment match) is logged — this
list is itself the source for discovering new aliases: if a rejected
phrase recurs often (e.g. "اقامه دعوا" instead of "اقامه دعوی"), it's
manually added to *_aliases.json. Vocabulary growth happens via manual
review, not automatic guessing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from features.configs import VOCAB_CATEGORIES

_HERE = Path(__file__).resolve().parent.parent
VOCAB_DIR = _HERE / "data" / "legal-vocabulary" / "categorized"

_MIN_TOKEN_LEN = 2  # matches MIN_TOKEN_LEN in vocab_gap_audit.py


def _normalize(s: str) -> str:
    return (
        s.replace("ء", "").replace("أ", "ا").replace("إ", "ا").replace("ة", "ه")
        .replace("ي", "ی").replace("ك", "ک").replace(" ", "").replace("‌", "")
    )


_TOKEN_SPLIT_RE = re.compile(r"[^\u0600-\u06FF\u200c]+")  # Persian/Arabic letters + ZWNJ only


def _tokenize(text: str) -> list[str]:
    """
    Matches tokenize in vocab_gap_audit.py — only counts Persian/Arabic
    runs as tokens (not raw substrings), to avoid false positives like
    "رد" inside "تجدیدنظرخواهی".
    """
    return [t for t in _TOKEN_SPLIT_RE.split(text) if len(t) >= _MIN_TOKEN_LEN]


def _token_sequence_contained(needle_tokens: list[str], haystack_tokens: list[str]) -> bool:
    """Is needle_tokens present as a complete, contiguous subsequence of haystack_tokens?"""
    n = len(needle_tokens)
    if n == 0:
        return False
    return any(
        haystack_tokens[i:i + n] == needle_tokens
        for i in range(len(haystack_tokens) - n + 1)
    )


def _load_alias_index(category_key: str) -> dict[str, str]:
    """Output: {normalized_alias_or_canonical: canonical_word}"""
    index: dict[str, str] = {}
    canon_path = VOCAB_DIR / f"{category_key}s.json"
    alias_path = VOCAB_DIR / f"{category_key}s_aliases.json"

    if canon_path.exists():
        for w in json.load(open(canon_path, encoding="utf-8")):
            index[_normalize(w)] = w

    if alias_path.exists():
        alias_map = json.load(open(alias_path, encoding="utf-8"))
        for canonical, raws in alias_map.items():
            for raw in raws:
                index.setdefault(_normalize(raw), canonical)

    return index


class VocabResolver:
    """
    One instance of this class is created *once*, at the start of
    main_features.py (not per case) — because loading the vocabulary has
    a cost and should only happen once, then be reused across all cases.
    """

    def __init__(self):
        self._alias_index: dict[str, dict[str, str]] = {
            key: _load_alias_index(key) for key in VOCAB_CATEGORIES
        }
        # For token-containment: each canonical/alias term along with its
        # own tokens (precomputed, for speed)
        self._tokenized_vocab: dict[str, list[tuple[list[str], str]]] = {}
        for key, alias_index in self._alias_index.items():
            entries = []
            for normalized_raw, canonical in alias_index.items():
                # tokenize from the original canonical/alias (before normalize)
                pass
            self._tokenized_vocab[key] = []

        # Since alias_index's key is the normalized form (no spaces), we
        # need the original raw string for tokenizing — so raw words are
        # kept separately.
        self._raw_words_by_category: dict[str, list[tuple[str, str]]] = {
            key: self._load_raw_words(key) for key in VOCAB_CATEGORIES
        }

    def _load_raw_words(self, category_key: str) -> list[tuple[str, str]]:
        """Output: [(raw word (alias or canonical), canonical), ...]"""
        pairs: list[tuple[str, str]] = []
        canon_path = VOCAB_DIR / f"{category_key}s.json"
        alias_path = VOCAB_DIR / f"{category_key}s_aliases.json"

        if canon_path.exists():
            for w in json.load(open(canon_path, encoding="utf-8")):
                pairs.append((w, w))

        if alias_path.exists():
            alias_map = json.load(open(alias_path, encoding="utf-8"))
            for canonical, raws in alias_map.items():
                for raw in raws:
                    pairs.append((raw, canonical))

        return pairs

    def resolve(self, raw_value: str, category_key: str) -> str | None:
        """
        raw_value: a value the LLM wrote freely (without seeing the
        vocabulary). Returns: the canonical closed-vocabulary term, or
        None if no deterministic match (exact or token-containment) was
        found.
        """
        if category_key not in VOCAB_CATEGORIES:
            return None

        # 1. Exact / alias match (fastest and safest path)
        normalized = _normalize(raw_value)
        alias_index = self._alias_index[category_key]
        if normalized in alias_index:
            return alias_index[normalized]

        # 2. Token-containment — is a complete vocabulary term present,
        #    as a token sequence, inside raw_value? If several different
        #    terms match, the longest (most specific) one is chosen —
        #    e.g. if "خوانده ردیف اول" matches both "خوانده" and some
        #    hypothetically longer term, the longer one is preferred.
        raw_tokens = _tokenize(raw_value)
        if not raw_tokens:
            return None

        best_match: tuple[int, str] | None = None  # (token length, canonical)
        for word, canonical in self._raw_words_by_category[category_key]:
            word_tokens = _tokenize(word)
            if not word_tokens:
                continue
            if _token_sequence_contained(word_tokens, raw_tokens):
                if best_match is None or len(word_tokens) > best_match[0]:
                    best_match = (len(word_tokens), canonical)

        return best_match[1] if best_match else None