"""
features/vocab_retrieval.py — انتخاب زیرمجموعه‌ی محتمل از closed vocabulary
با استفاده از embeddingهایی که *از قبل* روی RulingSection ذخیره شده‌اند
(database/embedding_store.py) — نه با embed کردن دوباره‌ی متن پرونده.

چرا این بهتر از embed کردن مجدد است؟
    چون همون مدل (bge-m3/Ollama) و همون واحد (هر بخش رأی جدا) از قبل
    برای کل کورپوس محاسبه و در Neo4j ذخیره شده. embed کردن دوباره‌ی
    متن رأی در extractor.py هم هزینه‌ی محاسباتی تکراری بود، هم فرصت
    ناسازگاری (اگه یک روز chunk-size اینجا با chunk-size دیتابیس فرق
    می‌کرد). حالا هر دو طرف (پرونده و واژگان) دقیقاً از یک فضای برداری
    می‌آیند.
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
    ورودی: embeddingهای بخش‌های یک پرونده (از EmbeddingStore.get_section_embeddings)
    خروجی: {category_key: [واژه‌های محتمل]}
    """
    if not section_embeddings:
        # اگه پرونده هنوز embedding نداره (مثلاً هنوز embed_all.py cases
        # روش اجرا نشده)، به‌جای شکست، کل واژه‌نامه رو برگردون —
        # fallback امن، نه crash.
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