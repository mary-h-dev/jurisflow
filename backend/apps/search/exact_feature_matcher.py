"""
Deterministic (exact/token-containment) matching of query text against
the closed legal vocabulary, before falling back to embedding similarity.

Mirrors features/vocab_resolver.py's approach: short legal terms (نشوز,
پرداخت, دفاع مشروع) are indistinguishable from near-synonyms in general
embedding space (see: توقیف vs توقف, cosine gap of only 0.043 — no
threshold separates them reliably). Since the vocabulary is closed and
finite, exact/token matching gives perfect precision where it fires,
with embedding reserved only as fallback for queries that don't
literally mention any vocabulary term.
"""

from __future__ import annotations

import re
from pathlib import Path
import json

VOCAB_CATEGORIES = ["concepts", "actions", "roles", "objects"]  

VOCAB_DIR = Path(__file__).resolve().parent / "vocabulary_data"


_MIN_TOKEN_LEN = 2
_TOKEN_SPLIT_RE = re.compile(r"[^\u0600-\u06FF\u200c]+")


# Terms too generic to be useful as exact-match anchors — they appear in
# nearly every legal ruling, so matching them returns hundreds of
# undifferentiated nodes with identical score=1.0, providing no signal.
# Identified empirically: "جرم" alone returned 24/30 evidence items in
# a single query. Extend this list as more generic terms are found.
_STOP_CONCEPTS = {
    "جرم", "قانونی", "حق", "دلیل", "قانون", "دادگاه", "حکم", "ماده",
    "دعوی", "دعوا", "طرفین", "خواسته", "رأی", "مقررات",
}



def _normalize(s: str) -> str:
    return (
        s.replace("ء", "").replace("أ", "ا").replace("إ", "ا").replace("ة", "ه")
        .replace("ي", "ی").replace("ك", "ک").replace(" ", "").replace("‌", "")
    )


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_SPLIT_RE.split(text) if len(t) >= _MIN_TOKEN_LEN]


class ExactFeatureMatcher:
    """
    One instance built once at startup (module-level singleton pattern,
    same as VocabResolver — vocabulary loading has a cost that should
    not repeat per query).
    """

    def __init__(self):
        self._canonical_words: list[tuple[list[str], str, str]] = []
        # each entry: (tokenized_word, canonical_word, category_key)
        for key in VOCAB_CATEGORIES:
            for word, canonical in self._load_raw_words(key):
                tokens = _tokenize(word)
                if tokens:
                    self._canonical_words.append((tokens, canonical, key))


    def _load_raw_words(self, category_key: str) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        canon_path = VOCAB_DIR / f"{category_key}.json"           
        alias_path = VOCAB_DIR / f"{category_key}_aliases.json"   

        if canon_path.exists():
            for w in json.load(open(canon_path, encoding="utf-8")):
                pairs.append((w, w))

        if alias_path.exists():
            alias_map = json.load(open(alias_path, encoding="utf-8"))
            for canonical, raws in alias_map.items():
                for raw in raws:
                    pairs.append((raw, canonical))

        return pairs

    def find_matches(self, query: str) -> list[tuple[str, str]]:
        """
        Returns [(canonical_word, category_key), ...] for every vocabulary
        term whose full token sequence appears in the query. Deterministic:
        either the exact word is there, or it isn't — no similarity score.
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []


        matches = []
        seen = set()
        for word_tokens, canonical, category_key in self._canonical_words:
            if canonical in _STOP_CONCEPTS:
                continue
            n = len(word_tokens)
            if any(query_tokens[i:i + n] == word_tokens for i in range(len(query_tokens) - n + 1)):
                key = (canonical, category_key)
                if key not in seen:
                    seen.add(key)
                    matches.append(key)
        return matches

exact_feature_matcher = ExactFeatureMatcher()