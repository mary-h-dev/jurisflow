"""
features/vocab_retrieval.py — DEPRECATED / NOT USED BY THE ACTIVE PIPELINE.

Kept for reference only. extractor.py does NOT import this module — see
extractor.py's own docstring for why this approach was tried and rejected.

Original purpose: select a likely subset of the closed vocabulary using
embeddings already stored on RulingSection nodes
(database/embedding_store.py) — instead of re-embedding the case text.

Why this seemed better than re-embedding:
    The same model (bge-m3/Ollama) and the same unit (each ruling section
    separately) had already been computed and stored in Neo4j for the
    whole corpus. Re-embedding the ruling text inside extractor.py would
    have been both redundant computation and a source of inconsistency
    (if chunk-size here ever diverged from the database's chunk-size).
    This way both sides (case and vocabulary) would come from exactly the
    same vector space.

Why it was rejected in favor of the current approach (free extraction +
post-hoc resolution): short, frequent terms (e.g. "خواهان") scored low
similarity against a long paragraph and got dropped from the candidate
set before the LLM ever saw them. See extractor.py's module docstring.
"""

import json
import math
from pathlib import Path

from features.configs import VOCAB_CATEGORIES

_HERE = Path(__file__).resolve().parent.parent
VOCAB_EMBED_CACHE = _HERE / "data" / "legal-vocabulary" / "gap_audit" / "vocab_embeddings_cache.json"

TOP_K_PER_SECTION = 15
MAX_PER_CATEGORY = 50


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _load_vocab_cache() -> dict[str, dict]:
    with open(VOCAB_EMBED_CACHE, encoding="utf-8") as f:
        return json.load(f)


def retrieve_relevant_vocab(section_embeddings: list[list[float]]) -> dict[str, list[str]]:
    """
    Input: embeddings of a case's sections (from
    EmbeddingStore.get_section_embeddings). Output: {category_key: [likely terms]}
    """
    if not section_embeddings:
        # If the case has no embeddings yet (e.g. embed_all.py for cases
        # hasn't run yet), return the whole vocabulary instead of failing
        # — a safe fallback, not a crash.
        vocab_cache = _load_vocab_cache()
        result = {key: [] for key in VOCAB_CATEGORIES}
        for entry in vocab_cache.values():
            if entry["category"] in result:
                result[entry["category"]].append(entry["word"])
        return result

    vocab_cache = _load_vocab_cache()
    best_score: dict[str, float] = {}
    word_category: dict[str, str] = {}

    for vec in section_embeddings:
        scored = [
            (entry["word"], entry["category"], _cosine(vec, entry["vector"]))
            for entry in vocab_cache.values()
        ]
        for category_key in VOCAB_CATEGORIES:
            cat_scored = sorted(
                (s for s in scored if s[1] == category_key), key=lambda x: -x[2]
            )[:TOP_K_PER_SECTION]
            for word, cat, score in cat_scored:
                if word not in best_score or score > best_score[word]:
                    best_score[word] = score
                    word_category[word] = cat

    result: dict[str, list[str]] = {key: [] for key in VOCAB_CATEGORIES}
    for word, cat in word_category.items():
        result[cat].append(word)
    for cat in result:
        result[cat] = sorted(result[cat], key=lambda w: -best_score[w])[:MAX_PER_CATEGORY]

    return result