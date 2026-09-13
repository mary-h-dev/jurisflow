"""
features/vocab_gap_audit.py — Frequency Audit over the closed vocabulary

Goal: find frequently-occurring phrases across the full text of ~5000
cases that aren't yet covered by any of the 5 closed-vocabulary
categories (concepts/actions/roles/objects/principles) — without
removing or rewriting anything in the current vocabulary. This only adds
a *proposed* supplementary layer; nothing is automatically added to the
categorized/*.json files.

Usage:
    uv run -m features.vocab_gap_audit

    Or just the cleaning safety check (without a full run):
    uv run python -c "from features.vocab_gap_audit import check_cleaning_safety; check_cleaning_safety(200)"
"""

from __future__ import annotations

import csv
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from common.embedder import embed_text
from features.configs import VOCAB_CATEGORIES

_HERE = Path(__file__).resolve().parent.parent  # graph-rag/
CASES_DIR = _HERE / "data" / "cases"
VOCAB_DIR = _HERE / "data" / "legal-vocabulary" / "categorized"
GAP_AUDIT_DIR = _HERE / "data" / "legal-vocabulary" / "gap_audit"
VOCAB_EMBED_CACHE = GAP_AUDIT_DIR / "vocab_embeddings_cache.json"
CANDIDATE_EMBED_CHECKPOINT = GAP_AUDIT_DIR / "_candidate_embeddings.jsonl"
REPORT_PATH = GAP_AUDIT_DIR / "gap_report.csv"

# --- tunable settings ---
MIN_DISTINCT_RULINGS = 50  # starting threshold; adjust after seeing the real distribution
ALIAS_SIMILARITY_THRESHOLD = 0.87  # suggested starting point (0.85-0.9)
MIN_TOKEN_LEN = 2  # single-character terms are usually noise

# --- stopword filter (added later) ---
# Why was this needed? The first real run of gap_report.csv showed the
# top of the list (by distinct case count) full of Persian prepositions/
# conjunctions ("به", "است", "از", "در", "با", "که", "را", "این", "بر"...)
# that had been assigned to random categories (concept/role/object) with
# low similarity scores (0.60-0.70, well below
# ALIAS_SIMILARITY_THRESHOLD). These are in no way legal-vocabulary
# candidates; a small fixed list is enough to filter them — something
# more elaborate (like POS-tagging) isn't needed for this kind of noise.
#
# Why is a unigram/bigram only filtered when *all* of its tokens are
# stopwords, not if even one of them is? Because bigrams like "پس از"
# (itself just a preposition) should be filtered, but meaningful
# combinations like "دادگاه عمومی" or "تجدیدنظر استان" that happen to
# contain no stopword must stay untouched. Simple rule: filter only if
# and only if every token of the phrase is in STOPWORDS.
STOPWORDS: set[str] = {
    "به", "از", "در", "با", "که", "را", "این", "آن", "بر", "پس",
    "است", "شده", "شد", "باشد", "بود", "می", "را", "تا", "یا", "و",
    "برای", "چون", "اگر", "نیز", "هم", "یک", "خود", "دیگر", "همه",
    "هر", "چه", "کدام", "چون", "زیرا", "لذا", "بنابراین", "اما",
    "ولی", "چنانچه", "نمود", "نموده", "گردید", "گردیده", "کرد",
    "کرده", "دارد", "داشت", "است", "بایست", "باید", "نباید",
}


# ==================================================================
# Stage 1: text cleaning before frequency counting
# ==================================================================

FULL_LAW_NAMES = [
    "قانون مدنی",
    "قانون تجارت",
    "قانون مجازات اسلامی",
    "قانون اساسی",
    "قانون آیین دادرسی مدنی",
    "قانون آئین دادرسی مدنی",
    "قانون آیین دادرسی دادگاه های عمومی و انقلاب در امور مدنی",
    "قانون آیین دادرسی کیفری",
    "قانون آئین دادرسی کیفری",
    "قانون آیین دادرسی دادگاه های عمومی و انقلاب در امور کیفری",
    "قانون مسئولیت مدنی",
    "قانون کار",
    "قانون تجارت الکترونیک",
    "قانون صدور چک",
    "قانون چک",
    "قانون ثبت اسناد و املاک",
]

_PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"

_ARTICLE_PATTERN = r"ماده\s*[\d" + _PERSIAN_DIGITS + r"]+"
_ARTICLE_REF_RE = re.compile(r"^ماده\s*[\d" + _PERSIAN_DIGITS + r"]+$")

_CLEAN_PATTERNS = [
    *[re.escape(name) for name in sorted(FULL_LAW_NAMES, key=len, reverse=True)],
    _ARTICLE_PATTERN,
    r"شعبه\s*[\d" + _PERSIAN_DIGITS + r"]+",
    r"[\d" + _PERSIAN_DIGITS + r"]+([/\-][\d" + _PERSIAN_DIGITS + r"]+)+",
    r"[\d" + _PERSIAN_DIGITS + r"]{3,}",
    r"(?<![\u0600-\u06FF])(?:[\u0600-\u06FF]\.){1,4}",
]
_CLEAN_REGEX = re.compile("|".join(_CLEAN_PATTERNS))

_REMOVED_SENTINEL = " ‹removed› "

# Known caveat: phrases that sometimes disappear due to overlap with
# FULL_LAW_NAMES (e.g. "طبق قانون" before "قانون مدنی"). Reviewed and
# confirmed this is inherent text ambiguity, not a bug — its occurrence
# rate is very low.
KNOWN_ACCEPTABLE_OVERLAPS = {"طبق قانون"}


def clean_text_for_frequency(text: str) -> str:
    return _CLEAN_REGEX.sub(_REMOVED_SENTINEL, text)


def _is_cataloged_article_reference(word: str) -> bool:
    return bool(_ARTICLE_REF_RE.match(word.strip()))


# ==================================================================
# Stage 2: frequency counting (unigram + bigram) — aware of real adjacency
# ==================================================================

_SCAN_RE = re.compile(r"[\u0600-\u06FF\u200c]+|[^\u0600-\u06FF\u200c]+")
_WS_ONLY_RE = re.compile(r"^\s*$")


def _is_persian_run(s: str) -> bool:
    return bool(s) and ("\u0600" <= s[0] <= "\u06FF" or s[0] == "\u200c")


def tokenize(text: str) -> list[str]:
    return [
        m.group(0) for m in _SCAN_RE.finditer(text)
        if _is_persian_run(m.group(0)) and len(m.group(0)) >= MIN_TOKEN_LEN
    ]


def _tokens_with_real_adjacency(text: str) -> tuple[list[str], list[bool]]:
    tokens: list[str] = []
    adjacency: list[bool] = []
    gap_is_pure_whitespace = True

    for m in _SCAN_RE.finditer(text):
        val = m.group(0)
        if _is_persian_run(val):
            if len(val) >= MIN_TOKEN_LEN:
                if tokens:
                    adjacency.append(gap_is_pure_whitespace)
                tokens.append(val)
                gap_is_pure_whitespace = True
            else:
                gap_is_pure_whitespace = False
        else:
            if not _WS_ONLY_RE.match(val):
                gap_is_pure_whitespace = False

    return tokens, adjacency


def _is_all_stopwords(gram: str) -> bool:
    """Are *all* tokens of this phrase stopwords? (not just one — otherwise
    meaningful combinations like "دادگاه عمومی" would also be filtered)"""
    return all(tok in STOPWORDS for tok in gram.split(" "))


def ngrams_for_ruling(text: str) -> set[str]:
    cleaned = clean_text_for_frequency(text)
    tokens, adjacency = _tokens_with_real_adjacency(cleaned)
    grams = {t for t in tokens if not _is_all_stopwords(t)}
    grams.update(
        f"{tokens[i]} {tokens[i+1]}"
        for i in range(len(tokens) - 1)
        if adjacency[i] and not _is_all_stopwords(f"{tokens[i]} {tokens[i+1]}")
    )
    return grams


def count_frequencies(ruling_texts: dict[str, str]) -> dict[str, dict]:
    freq: dict[str, dict] = defaultdict(lambda: {"total": 0, "ruling_ids": set()})
    for ruling_id, text in ruling_texts.items():
        cleaned = clean_text_for_frequency(text)
        tokens, adjacency = _tokens_with_real_adjacency(cleaned)
        all_grams = (
            [t for t in tokens if not _is_all_stopwords(t)]
            + [
                f"{tokens[i]} {tokens[i+1]}"
                for i in range(len(tokens) - 1)
                if adjacency[i] and not _is_all_stopwords(f"{tokens[i]} {tokens[i+1]}")
            ]
        )
        seen_in_this_ruling = set()
        for g in all_grams:
            freq[g]["total"] += 1
            seen_in_this_ruling.add(g)
        for g in seen_in_this_ruling:
            freq[g]["ruling_ids"].add(ruling_id)
    return dict(freq)


def load_all_ruling_texts() -> dict[str, str]:
    texts = {}
    bad_files = []
    paths = glob.glob(str(CASES_DIR / "*" / "*.json"))
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                ruling = json.load(f)
            texts[ruling["ruling_id"]] = "\n\n".join(s["text"] for s in ruling.get("sections", []))
        except (json.JSONDecodeError, KeyError, OSError) as e:
            bad_files.append((path, str(e)))

    if bad_files:
        print(f"⚠️ Skipped {len(bad_files)} corrupted/empty files (out of {len(paths)} total):")
        for path, err in bad_files[:20]:
            print(f"  - {path}: {err}")
        if len(bad_files) > 20:
            print(f"  ... and {len(bad_files) - 20} more")

    return texts


# ==================================================================
# Stage 3: comparison against the current vocabulary
# ==================================================================

def _normalize(s: str) -> str:
    return (
        s.replace("ء", "").replace("أ", "ا").replace("إ", "ا").replace("ة", "ه")
        .replace("ي", "ی").replace("ك", "ک").replace(" ", "").replace("‌", "")
    )


def load_existing_vocab() -> dict[str, tuple[str, str]]:
    index: dict[str, tuple[str, str]] = {}
    for key in VOCAB_CATEGORIES:
        canon_path = VOCAB_DIR / f"{key}s.json"
        alias_path = VOCAB_DIR / f"{key}s_aliases.json"
        if canon_path.exists():
            for w in json.load(open(canon_path, encoding="utf-8")):
                index[_normalize(w)] = (w, key)
        if alias_path.exists():
            alias_map = json.load(open(alias_path, encoding="utf-8"))
            for root, raws in alias_map.items():
                for raw in raws:
                    index.setdefault(_normalize(raw), (root, key))
    return index


def find_gap_candidates(
    freq: dict[str, dict], vocab_index: dict[str, tuple[str, str]], min_distinct: int
) -> list[str]:
    return [
        g for g, stats in freq.items()
        if len(stats["ruling_ids"]) >= min_distinct
        and _normalize(g) not in vocab_index
        and not _is_all_stopwords(g)  # second safety layer — even if something slipped past stage-2 filtering
    ]


# ==================================================================
# Stage 4: semantic similarity check (prevents duplicates/drift)
# ==================================================================

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def build_vocab_embeddings(vocab_index: dict[str, tuple[str, str]]) -> dict[str, dict]:
    GAP_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    cache = {}
    if VOCAB_EMBED_CACHE.exists():
        cache = json.load(open(VOCAB_EMBED_CACHE, encoding="utf-8"))

    to_embed = [(norm, word, cat) for norm, (word, cat) in vocab_index.items() if norm not in cache]
    print(f"📚 Embedding vocabulary: {len(cache)} already cached, {len(to_embed)} remaining")

    for i, (norm, word, cat) in enumerate(to_embed, start=1):
        vector = embed_text(word)
        cache[norm] = {"word": word, "category": cat, "vector": vector}
        if i % 50 == 0:
            json.dump(cache, open(VOCAB_EMBED_CACHE, "w", encoding="utf-8"))
            print(f"  ... {i}/{len(to_embed)} embedded (checkpoint saved)")

    json.dump(cache, open(VOCAB_EMBED_CACHE, "w", encoding="utf-8"))
    print(f"✅ Total vocabulary embedding cache: {len(cache)} terms")
    return cache


def load_candidate_embeddings_checkpoint() -> dict[str, list[float]]:
    if not CANDIDATE_EMBED_CHECKPOINT.exists():
        return {}
    done = {}
    with open(CANDIDATE_EMBED_CHECKPOINT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                done[row["candidate"]] = row["vector"]
    return done


def embed_candidates(candidates: list[str]) -> dict[str, list[float]]:
    done = load_candidate_embeddings_checkpoint()
    remaining = [c for c in candidates if c not in done]
    print(f"🔎 Embedding candidates: {len(done)} already cached, {len(remaining)} remaining")
    with open(CANDIDATE_EMBED_CHECKPOINT, "a", encoding="utf-8") as f:
        for i, c in enumerate(remaining, start=1):
            vector = embed_text(c)
            done[c] = vector
            f.write(json.dumps({"candidate": c, "vector": vector}, ensure_ascii=False) + "\n")
            if i % 50 == 0:
                print(f"  ... {i}/{len(remaining)} embedded")
    return done


def find_best_vocab_match(
    candidate_vector: list[float], vocab_cache: dict[str, dict]
) -> tuple[str, str, float]:
    best_word, best_cat, best_score = None, None, -1.0
    for entry in vocab_cache.values():
        score = _cosine(candidate_vector, entry["vector"])
        if score > best_score:
            best_word, best_cat, best_score = entry["word"], entry["category"], score
    return best_word, best_cat, best_score


# ==================================================================
# Cleaning safety test (mandatory precondition before run)
# ==================================================================

def _phrase_present_with_real_adjacency(
    tokens: list[str], adjacency: list[bool], word_tokens: list[str]
) -> bool:
    n = len(word_tokens)
    if n == 0:
        return False
    if n == 1:
        return word_tokens[0] in tokens
    for i in range(len(tokens) - n + 1):
        if tokens[i:i + n] == word_tokens and all(adjacency[i:i + n - 1]):
            return True
    return False


def check_cleaning_safety(sample_size: int = 200, ruling_texts: dict[str, str] | None = None) -> list[dict]:
    vocab_index = load_existing_vocab()
    vocab_words = sorted({w for w, _ in vocab_index.values()}, key=len, reverse=True)
    vocab_word_tokens = {w: tokenize(w) for w in vocab_words}

    if ruling_texts is None:
        ruling_texts = load_all_ruling_texts()
    sample_ids = list(ruling_texts.keys())[:sample_size]

    expected_within_law_names = {
        w for w in vocab_words
        if any(
            _phrase_present_with_real_adjacency(*_tokens_with_real_adjacency(law), vocab_word_tokens[w])
            for law in FULL_LAW_NAMES
        )
    }
    expected_article_refs = {w for w in vocab_words if _is_cataloged_article_reference(w)}
    all_expected = expected_within_law_names | expected_article_refs | KNOWN_ACCEPTABLE_OVERLAPS

    expected_problems, unexpected_problems = [], []
    for ruling_id in sample_ids:
        raw = ruling_texts[ruling_id]
        cleaned = clean_text_for_frequency(raw)
        raw_tokens, raw_adjacency = _tokens_with_real_adjacency(raw)
        cleaned_tokens, cleaned_adjacency = _tokens_with_real_adjacency(cleaned)
        for word in vocab_words:
            word_tokens = vocab_word_tokens[word]
            was_present = _phrase_present_with_real_adjacency(raw_tokens, raw_adjacency, word_tokens)
            if not was_present:
                continue
            still_present = _phrase_present_with_real_adjacency(cleaned_tokens, cleaned_adjacency, word_tokens)
            if not still_present:
                entry = {"ruling_id": ruling_id, "word": word}
                if word in all_expected:
                    expected_problems.append(entry)
                else:
                    unexpected_problems.append(entry)

    if expected_problems:
        distinct_words = sorted({p["word"] for p in expected_problems})
        print(f"ℹ️ {len(distinct_words)} expected terms (part of a law name, a cataloged "
              f"article reference, or a known overlap) were intentionally removed — "
              f"this is normal, not a bug: {distinct_words}")

    if not unexpected_problems:
        print(f"✅ Across {len(sample_ids)} sample cases, no vocab term was unexpectedly removed.")
    else:
        distinct_unexpected = sorted({p["word"] for p in unexpected_problems})
        print(f"🔴 Found {len(unexpected_problems)} *unexpected* cleaning cases "
              f"({len(distinct_unexpected)} distinct terms):")
        for p in unexpected_problems[:30]:
            print(f"  - «{p['word']}» removed in case {p['ruling_id']}")
        if len(unexpected_problems) > 30:
            print(f"  ... and {len(unexpected_problems) - 30} more")
        print("⚠️ Fix these in FULL_LAW_NAMES/_CLEAN_PATTERNS before running run().")

    return unexpected_problems


# ==================================================================
# Stages 5 & 6: final report + coverage
# ==================================================================

def run(min_distinct_rulings: int = MIN_DISTINCT_RULINGS, safety_sample_size: int = 200):
    print("📂 Loading full ruling texts...")
    ruling_texts = load_all_ruling_texts()
    print(f"   {len(ruling_texts)} cases loaded")

    print("\n🛡️ Running mandatory precondition: check_cleaning_safety...")
    problems = check_cleaning_safety(sample_size=safety_sample_size, ruling_texts=ruling_texts)
    if problems:
        print("\n❌ run() stopped because cleaning is problematic. Fix these first, then re-run.")
        return
    print()

    print("🔢 Counting unigram/bigram frequency (stopwords filtered)...")
    freq = count_frequencies(ruling_texts)
    print(f"   {len(freq)} unique phrases (after cleaning and stopword filtering)")

    vocab_index = load_existing_vocab()
    print(f"📖 Current vocabulary: {len(vocab_index)} unique terms/aliases (normalized)")

    candidates = find_gap_candidates(freq, vocab_index, min_distinct_rulings)
    print(f"🕳️ Found {len(candidates)} gap candidates (threshold: at least "
          f"{min_distinct_rulings} distinct cases)")

    if not candidates:
        print("✅ No gap found above the threshold.")
        return

    vocab_cache = build_vocab_embeddings(vocab_index)
    candidate_vectors = embed_candidates(candidates)

    GAP_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "عبارت", "فرکانس کل", "تعداد پرونده‌ی متمایز",
            "دسته‌ی پیشنهادی", "نزدیک‌ترین واژه‌ی موجود (امتیاز)", "پیشنهاد نهایی",
        ])
        for c in sorted(candidates, key=lambda x: -len(freq[x]["ruling_ids"])):
            best_word, best_cat, score = find_best_vocab_match(candidate_vectors[c], vocab_cache)
            if score >= ALIAS_SIMILARITY_THRESHOLD:
                verdict = f"alias پیشنهادی برای «{best_word}»"
            else:
                verdict = "واژه‌ی جدید"
            writer.writerow([
                c,
                freq[c]["total"],
                len(freq[c]["ruling_ids"]),
                best_cat or "نامشخص",
                f"{best_word} ({score:.2f})" if best_word else "-",
                verdict,
            ])
    print(f"💾 Report saved: {REPORT_PATH}")

    total_freq_all = sum(stats["total"] for stats in freq.values())
    covered_freq = sum(
        stats["total"] for g, stats in freq.items() if _normalize(g) in vocab_index
    )
    gap_freq = sum(freq[c]["total"] for c in candidates)

    coverage_before = covered_freq / total_freq_all if total_freq_all else 0
    coverage_after = (covered_freq + gap_freq) / total_freq_all if total_freq_all else 0

    print("\n" + "=" * 60)
    print("📊 Coverage Report")
    print("=" * 60)
    print(f"  Before audit: {coverage_before:.2%}")
    print(f"  After audit (assuming all gaps are approved): {coverage_after:.2%}")
    print(f"  (The denominator includes all corpus n-grams *after* stopword filtering —")
    print(f"   not just frequent candidates; that's why even 'after' doesn't reach 100%.)")


if __name__ == "__main__":
    run()