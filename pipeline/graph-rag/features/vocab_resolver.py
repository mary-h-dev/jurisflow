"""
features/vocab_resolver.py — نگاشت مقدار آزادِ استخراج‌شده توسط LLM به
نزدیک‌ترین واژه‌ی closed vocabulary — فقط با روش‌های قطعی (deterministic)،
بدون هیچ fuzzy matching احتمالی (نه character-similarity، نه embedding).

چرا هیچ fuzzy matching (نه difflib، نه embedding)؟
    تست شد و رد شد. «توقیف» (مصادره‌ی مال) و «توقف» (متوقف‌شدن) فقط یک
    حرف فرق دارن ولی معنایشان کاملاً متفاوته — difflib این دو رو
    اشتباهی یکی گرفت. برای رد کردن این ریسک با ابزار دیگه، شباهت
    embedding (bge-m3) هم مستقیم تست شد: «توقیف» در برابر «توقیف
    خودرو» = ۰.۸۸۹ (باید شبیه باشه)، «توقیف» در برابر «توقف» = ۰.۸۴۶
    (نباید شبیه باشه). فاصله‌ی فقط ۰.۰۴۳ بین match درست و غلط یعنی
    هیچ threshold ای نمی‌تونه این دو رو قابل‌اعتماد از هم جدا کنه —
    این یک محدودیت ذاتی مدل embedding عمومیه برای افعال هم‌ریشه‌ی
    فارسی، نه یک پارامتر قابل‌تنظیم.

    پس فلسفه: بهتره یک Feature درست رد بشه، تا این‌که به یک واژه‌ی
    نزدیک ولی معنایاً غلط snap بشه و بی‌صدا در گراف بشینه — دقیقاً
    همون فلسفه‌ای که در کل این پروژه (رد کردن evidence بدون موقعیت،
    رد کردن quote های حاوی «...») دنبال کردیم.

دو نوع match مجاز — هردو قطعی، نه احتمالی:
    ۱. Exact / Alias match: مقدار (بعد از normalize) دقیقاً برابر یک
       canonical یا یکی از aliasهاش باشد.
    ۲. Token-containment: یک واژه‌ی canonical *کامل* (به‌صورت
       دنباله‌ی توکن، نه substring خام) داخل عبارت استخراج‌شده باشد.
       مثال: «خوانده ردیف اول» شامل توکن دقیق «خوانده» است → match.
       ولی «توقیف» هرگز به «توقف» match نمی‌شود، چون این دو رشته‌ی
       متفاوتند، نه چون شبیه‌اند.

       چرا سطح توکن، نه substring خام؟ چون substring خام باعث
       false-positive می‌شود — مثلاً «رد» می‌تواند به‌عنوان substring
       داخل «تجدیدنظرخواهی» ظاهر شود بدون این‌که واقعاً همان کلمه
       باشد. همان الگویی که در features/vocab_gap_audit.py برای حل
       مشکل مشابه («ولی» داخل «مسئولیت») استفاده شد، اینجا هم به‌کار
       رفته.

هر Feature رد شده (نه match شد، نه exact نه containment) لاگ می‌شود —
این لیست خودش منبع کشف alias های جدید است: اگر یک عبارت رد شده زیاد
تکرار شد (مثلاً «اقامه دعوا» به‌جای «اقامه دعوی»)، دستی به
*_aliases.json اضافه می‌شود. رشد واژه‌نامه با مرور دستی، نه حدس خودکار.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from features.configs import VOCAB_CATEGORIES

_HERE = Path(__file__).resolve().parent.parent
VOCAB_DIR = _HERE / "data" / "legal-vocabulary" / "categorized"

_MIN_TOKEN_LEN = 2  # هم‌راستا با MIN_TOKEN_LEN در vocab_gap_audit.py


def _normalize(s: str) -> str:
    return (
        s.replace("ء", "").replace("أ", "ا").replace("إ", "ا").replace("ة", "ه")
        .replace("ي", "ی").replace("ك", "ک").replace(" ", "").replace("‌", "")
    )


_TOKEN_SPLIT_RE = re.compile(r"[^\u0600-\u06FF\u200c]+")  # فقط حروف فارسی/عربی + نیم‌فاصله


def _tokenize(text: str) -> list[str]:
    """
    هم‌راستا با tokenize در vocab_gap_audit.py — فقط دنباله‌های فارسی/
    عربی را توکن حساب می‌کند (نه substring خام)، تا از false-positive
    مثل «رد» داخل «تجدیدنظرخواهی» جلوگیری شود.
    """
    return [t for t in _TOKEN_SPLIT_RE.split(text) if len(t) >= _MIN_TOKEN_LEN]


def _token_sequence_contained(needle_tokens: list[str], haystack_tokens: list[str]) -> bool:
    """آیا دنباله‌ی needle_tokens به‌طور کامل و پیوسته در haystack_tokens حاضر است؟"""
    n = len(needle_tokens)
    if n == 0:
        return False
    return any(
        haystack_tokens[i:i + n] == needle_tokens
        for i in range(len(haystack_tokens) - n + 1)
    )


def _load_alias_index(category_key: str) -> dict[str, str]:
    """خروجی: {normalized_alias_or_canonical: canonical_word}"""
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
    یک نمونه از این کلاس فقط *یک‌بار* در ابتدای main_features.py ساخته
    می‌شود (نه هر پرونده) — چون بارگذاری واژه‌نامه هزینه دارد و باید
    فقط یک‌بار انجام شود، بعد برای همه‌ی پرونده‌ها استفاده مجدد شود.
    """

    def __init__(self):
        self._alias_index: dict[str, dict[str, str]] = {
            key: _load_alias_index(key) for key in VOCAB_CATEGORIES
        }
        # برای token-containment: هر واژه‌ی canonical/alias به‌همراه
        # توکن‌های خودش (پیش‌محاسبه‌شده، برای سرعت)
        self._tokenized_vocab: dict[str, list[tuple[list[str], str]]] = {}
        for key, alias_index in self._alias_index.items():
            entries = []
            for normalized_raw, canonical in alias_index.items():
                # از خودِ canonical/alias اصلی (قبل از normalize) توکنایز می‌کنیم
                pass
            self._tokenized_vocab[key] = []

        # چون alias_index کلیدش نسخه‌ی normalize شده است (بدون فاصله)،
        # برای توکنایز کردن باید از خودِ رشته‌ی اصلی (قبل از normalize)
        # استفاده کنیم — پس raw wordها را جدا نگه می‌داریم.
        self._raw_words_by_category: dict[str, list[tuple[str, str]]] = {
            key: self._load_raw_words(key) for key in VOCAB_CATEGORIES
        }

    def _load_raw_words(self, category_key: str) -> list[tuple[str, str]]:
        """خروجی: [(واژه‌ی خام (alias یا canonical)، canonical), ...]"""
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
        raw_value: مقداری که LLM آزادانه (بدون دیدن واژه‌نامه) نوشته.
        خروجی: واژه‌ی canonical در closed vocabulary، یا None اگر
        هیچ match قطعی (exact یا token-containment) پیدا نشد.
        """
        if category_key not in VOCAB_CATEGORIES:
            return None

        # ۱. Exact / Alias match (سریع‌ترین و امن‌ترین مسیر)
        normalized = _normalize(raw_value)
        alias_index = self._alias_index[category_key]
        if normalized in alias_index:
            return alias_index[normalized]

        # ۲. Token-containment — آیا یک واژه‌ی کامل واژه‌نامه، به‌صورت
        #    دنباله‌ی توکن، داخل raw_value هست؟ اگه چند واژه‌ی مختلف
        #    match شدن، طولانی‌ترین (دقیق‌ترین) رو انتخاب می‌کنیم —
        #    مثلاً «خوانده ردیف اول» اگه هم به «خوانده» هم به یک واژه‌ی
        #    فرضی طولانی‌تر match بشه، طولانی‌تر ترجیح داده می‌شود.
        raw_tokens = _tokenize(raw_value)
        if not raw_tokens:
            return None

        best_match: tuple[int, str] | None = None  # (طول توکن، canonical)
        for word, canonical in self._raw_words_by_category[category_key]:
            word_tokens = _tokenize(word)
            if not word_tokens:
                continue
            if _token_sequence_contained(word_tokens, raw_tokens):
                if best_match is None or len(word_tokens) > best_match[0]:
                    best_match = (len(word_tokens), canonical)

        return best_match[1] if best_match else None