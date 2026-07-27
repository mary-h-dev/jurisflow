"""
features/vocab_gap_audit.py — Frequency Audit روی closed vocabulary

هدف: پیدا کردن عبارات پرتکرار توی متن کامل ۵۰۰۰ پرونده که هنوز در
هیچ‌کدوم از ۵ دسته‌ی closed vocabulary (concepts/actions/roles/objects/
principles) پوشش داده نشدن — بدون این‌که چیزی از واژه‌نامه‌ی فعلی حذف
یا بازنویسی بشه. این فقط یک لایه‌ی تکمیلیِ *پیشنهادی* اضافه می‌کنه؛
هیچ‌چیز خودکار به فایل‌های categorized/*.json اضافه نمی‌شه.

نحوه‌ی اجرا:
    uv run -m features.vocab_gap_audit

    یا فقط تست ایمنیِ پاک‌سازی (بدون اجرای کامل):
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

# --- تنظیمات قابل‌تغییر ---
MIN_DISTINCT_RULINGS = 50  # آستانه‌ی اولیه؛ بعد از دیدن توزیع واقعی تنظیمش کن
ALIAS_SIMILARITY_THRESHOLD = 0.87  # نقطه‌ی شروع پیشنهادی (۰.۸۵-۰.۹)
MIN_TOKEN_LEN = 2  # واژه‌های تک‌کاراکتری معمولاً نویزن

# --- فیلتر stopword (جدید) ---
# چرا لازم شد؟ اولین اجرای واقعی gap_report.csv رو دیدیم: صدر لیست
# (بر اساس تعداد پرونده‌ی متمایز) پر بود از حروف‌اضافه/ربط فارسی («به»،
# «است»، «از»، «در»، «با»، «که»، «را»، «این»، «بر»...) که به دسته‌های
# تصادفی (concept/role/object) با امتیاز شباهت پایین (۰.۶۰-۰.۷۰، خیلی
# زیر ALIAS_SIMILARITY_THRESHOLD) نسبت داده شده بودن. این‌ها به هیچ‌وجه
# candidate واژه‌نامه‌ی حقوقی نیستن؛ یک لیست ثابت و کوچیک برای فیلترشون
# کافیه — پیچیده‌تر (مثل POS-tagging) برای این نوع نویز لازم نیست.
#
# چرا یک unigram/bigram فقط وقتی فیلتر می‌شه که *همه‌ی* توکن‌هاش
# stopword باشن، نه اگه حتی یکی از توکن‌هاش باشه؟ چون bigramهایی مثل
# «پس از» (که خودش هم صرفاً حرف‌اضافه‌ست) باید فیلتر بشه، ولی ترکیب‌های
# معنادار مثل «دادگاه عمومی» یا «تجدیدنظر استان» که تصادفاً یک stopword
# ندارن دست‌نخورده می‌مونن. یک قانون ساده: اگر و فقط اگر تمام توکن‌های
# عبارت در STOPWORDS باشن، حذف می‌شه.
STOPWORDS: set[str] = {
    "به", "از", "در", "با", "که", "را", "این", "آن", "بر", "پس",
    "است", "شده", "شد", "باشد", "بود", "می", "را", "تا", "یا", "و",
    "برای", "چون", "اگر", "نیز", "هم", "یک", "خود", "دیگر", "همه",
    "هر", "چه", "کدام", "چون", "زیرا", "لذا", "بنابراین", "اما",
    "ولی", "چنانچه", "نمود", "نموده", "گردید", "گردیده", "کرد",
    "کرده", "دارد", "داشت", "است", "بایست", "باید", "نباید",
}


# ==================================================================
# مرحله ۱: پاک‌سازی متن قبل از شمارش فرکانس
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

# استثنای شناخته‌شده: عبارت‌هایی که گاهی به‌خاطر همپوشانی با
# FULL_LAW_NAMES ناپدید می‌شن (مثل «طبق قانون» قبل از «قانون مدنی»).
# بررسی و تأیید شده که ابهام ذاتی متنه، نه باگ — نرخ وقوعش خیلی پایینه.
KNOWN_ACCEPTABLE_OVERLAPS = {"طبق قانون"}


def clean_text_for_frequency(text: str) -> str:
    return _CLEAN_REGEX.sub(_REMOVED_SENTINEL, text)


def _is_cataloged_article_reference(word: str) -> bool:
    return bool(_ARTICLE_REF_RE.match(word.strip()))


# ==================================================================
# مرحله ۲: شمارش فرکانس (unigram + bigram) — با آگاهی از مجاورت واقعی
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
    """آیا *همه‌ی* توکن‌های این عبارت stopword ان؟ (نه فقط یکی‌شون —
    وگرنه ترکیب‌های معنادار مثل «دادگاه عمومی» هم فیلتر می‌شدن)"""
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
        print(f"⚠️ {len(bad_files)} فایل خراب/خالی رد شد (از {len(paths)} کل):")
        for path, err in bad_files[:20]:
            print(f"  - {path}: {err}")
        if len(bad_files) > 20:
            print(f"  ... و {len(bad_files) - 20} مورد دیگر")

    return texts


# ==================================================================
# مرحله ۳: مقایسه با واژه‌نامه‌ی فعلی
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
        and not _is_all_stopwords(g)  # لایه‌ی دوم اطمینان — حتی اگر جایی از فیلترِ مرحله‌ی ۲ رد شده باشه
    ]


# ==================================================================
# مرحله ۴: چک شباهت معنایی (جلوگیری از duplicate/drift)
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
    print(f"📚 embedding واژه‌نامه: {len(cache)} از قبل در کش، {len(to_embed)} باقی‌مونده")

    for i, (norm, word, cat) in enumerate(to_embed, start=1):
        vector = embed_text(word)
        cache[norm] = {"word": word, "category": cat, "vector": vector}
        if i % 50 == 0:
            json.dump(cache, open(VOCAB_EMBED_CACHE, "w", encoding="utf-8"))
            print(f"  ... {i}/{len(to_embed)} embed شد (ذخیره‌ی موقت انجام شد)")

    json.dump(cache, open(VOCAB_EMBED_CACHE, "w", encoding="utf-8"))
    print(f"✅ کل کش embedding واژه‌نامه: {len(cache)} واژه")
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
    print(f"🔎 embedding کاندیداها: {len(done)} از قبل در کش، {len(remaining)} باقی‌مونده")
    with open(CANDIDATE_EMBED_CHECKPOINT, "a", encoding="utf-8") as f:
        for i, c in enumerate(remaining, start=1):
            vector = embed_text(c)
            done[c] = vector
            f.write(json.dumps({"candidate": c, "vector": vector}, ensure_ascii=False) + "\n")
            if i % 50 == 0:
                print(f"  ... {i}/{len(remaining)} embed شد")
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
# تست ایمنیِ پاک‌سازی (پیش‌شرط اجباری قبل از run)
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
        print(f"ℹ️ {len(distinct_words)} واژه‌ی مورد انتظار (بخشی از نام قانون، ارجاع به ماده‌ی "
              f"کاتالوگ‌شده، یا همپوشانیِ شناخته‌شده) عمداً حذف شدن — این طبیعیه، نه باگ: "
              f"{distinct_words}")

    if not unexpected_problems:
        print(f"✅ روی {len(sample_ids)} پرونده‌ی نمونه، هیچ واژه‌ی vocab غیرمنتظره‌ای به‌اشتباه پاک نشد.")
    else:
        distinct_unexpected = sorted({p["word"] for p in unexpected_problems})
        print(f"🔴 {len(unexpected_problems)} مورد پاک‌سازیِ *غیرمنتظره* پیدا شد "
              f"({len(distinct_unexpected)} واژه‌ی یکتا):")
        for p in unexpected_problems[:30]:
            print(f"  - «{p['word']}» در پرونده‌ی {p['ruling_id']} حذف شد")
        if len(unexpected_problems) > 30:
            print(f"  ... و {len(unexpected_problems) - 30} مورد دیگر")
        print("⚠️ قبل از اجرای run() این‌ها را در FULL_LAW_NAMES/_CLEAN_PATTERNS اصلاح کن.")

    return unexpected_problems


# ==================================================================
# مرحله ۵ و ۶: گزارش نهایی + Coverage
# ==================================================================

def run(min_distinct_rulings: int = MIN_DISTINCT_RULINGS, safety_sample_size: int = 200):
    print("📂 بارگذاری متن کامل رأی‌ها...")
    ruling_texts = load_all_ruling_texts()
    print(f"   {len(ruling_texts)} پرونده بارگذاری شد")

    print("\n🛡️ اجرای پیش‌شرط اجباری: check_cleaning_safety...")
    problems = check_cleaning_safety(sample_size=safety_sample_size, ruling_texts=ruling_texts)
    if problems:
        print("\n❌ run() متوقف شد چون پاک‌سازی مشکل‌دار است. اول این‌ها را اصلاح کن، بعد دوباره اجرا کن.")
        return
    print()

    print("🔢 شمارش فرکانس unigram/bigram (stopwordها فیلتر شدن)...")
    freq = count_frequencies(ruling_texts)
    print(f"   {len(freq)} عبارت یکتا (بعد از پاک‌سازی و فیلتر stopword)")

    vocab_index = load_existing_vocab()
    print(f"📖 واژه‌نامه‌ی فعلی: {len(vocab_index)} واژه/alias یکتا (نرمال‌شده)")

    candidates = find_gap_candidates(freq, vocab_index, min_distinct_rulings)
    print(f"🕳️ {len(candidates)} کاندیدای gap پیدا شد (آستانه: حداقل "
          f"{min_distinct_rulings} پرونده‌ی متمایز)")

    if not candidates:
        print("✅ هیچ gapی بالای آستانه پیدا نشد.")
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
    print(f"💾 گزارش ذخیره شد: {REPORT_PATH}")

    total_freq_all = sum(stats["total"] for stats in freq.values())
    covered_freq = sum(
        stats["total"] for g, stats in freq.items() if _normalize(g) in vocab_index
    )
    gap_freq = sum(freq[c]["total"] for c in candidates)

    coverage_before = covered_freq / total_freq_all if total_freq_all else 0
    coverage_after = (covered_freq + gap_freq) / total_freq_all if total_freq_all else 0

    print("\n" + "=" * 60)
    print("📊 گزارش Coverage")
    print("=" * 60)
    print(f"  قبل از audit : {coverage_before:.2%}")
    print(f"  بعد از audit (به‌شرط تأیید همه‌ی gapها): {coverage_after:.2%}")
    print(f"  (مخرج شامل کل n-gramهای کورپوس، *بعد از فیلتر stopword*، است —")
    print(f"   نه فقط کاندیداهای پرتکرار؛ برای همین حتی «بعد» هم به ۱۰۰٪ نمی‌رسه.)")


if __name__ == "__main__":
    run()