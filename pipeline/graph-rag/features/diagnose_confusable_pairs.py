"""
features/diagnose_confusable_pairs.py — a cheap test before committing to
                                          an embedding-based resolver

Why was this script needed?
    Before running a full pilot again and seeing more incorrect snaps, we
    needed to directly ask: "can bge-m3 actually distinguish between
    'توقیف' and 'توقف' (or 'صدور حکم' and 'دوره محکومیت')?" If the
    answer is no, an embedding-based resolver is useless regardless, and
    the strategy needs to change entirely (e.g. exact/alias match only,
    no fuzzy matching at all). This test only needs a handful of
    embed_text calls (a few seconds), not a full 50-case pilot (several
    minutes + LLM cost).

Usage:
    uv run -m features.diagnose_confusable_pairs
"""

import math

from common.embedder import embed_text

# Pairs actually observed being incorrectly snapped together in previous
# outputs, plus a few control pairs (that should genuinely be similar, to
# confirm the threshold isn't meaninglessly strict)
CONFUSABLE_PAIRS = [
    # (model-extracted phrase, vocabulary term it was incorrectly snapped to, should they be similar?)
    ("توقیف", "توقف", False),                    # seizure of property  vs  coming to a stop — should NOT be similar
    ("صدور حکم بر رفع توقیف", "دوره محکومیت", False),  # completely unrelated
    ("بطلان عقد اجاره", "عقد اجاره", False),        # "بطلان" (nullity) must stay distinct, not get lost
    ("انقضاء مدت اجاره", "عقد اجاره", False),
    ("اعلان بطلان", "اعلام اینکه", False),           # not even a valid term at all
    # --- control pairs: these *should* be similar ---
    ("خوانده ردیف اول", "خوانده", True),
    ("اقامه دعوا", "اقامه دعوی", True),
    ("توقیف خودرو", "توقیف", True),
]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def run():
    print("🔬 Testing embedding distinguishability on previously problematic pairs\n")
    print(f"{'extracted phrase':<28} {'vocab term':<20} {'should be similar?':<19} {'actual similarity':<12}")
    print("-" * 84)

    problems = []
    for raw, vocab_word, should_be_similar in CONFUSABLE_PAIRS:
        v1 = embed_text(raw)
        v2 = embed_text(vocab_word)
        score = _cosine(v1, v2)

        verdict = "✅" if (score > 0.75) == should_be_similar else "🔴"
        print(f"{raw:<28} {vocab_word:<20} {'yes' if should_be_similar else 'no':<19} {score:.3f} {verdict}")

        if (score > 0.75) != should_be_similar:
            problems.append((raw, vocab_word, score, should_be_similar))

    print()
    if not problems:
        print("✅ Embedding correctly distinguished these pairs — an embedding-based "
              "resolver is probably reliable for this vocabulary.")
    else:
        print(f"🔴 {len(problems)} pairs were misclassified:")
        for raw, vocab_word, score, expected in problems:
            print(f"  - «{raw}» vs «{vocab_word}»: similarity={score:.3f} "
                  f"(expected them to be {'similar' if expected else 'dissimilar'})")
        print("\n⚠️ If most of these are 🔴, embedding is not reliable for these "
              "specific pairs either — the resolver should be restricted to "
              "exact/alias match only (no fuzzy semantic matching), even if that "
              "means lower recall.")


if __name__ == "__main__":
    run()