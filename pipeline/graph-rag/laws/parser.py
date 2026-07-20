"""
laws/parser.py — پارسر مخصوص ۵ قانون مادر (Rule Graph)

این فایل زیر laws/ است چون منطقش (تشخیص «ماده»/«تبصره»، چک‌لیست تشخیصی که
بعداً اضافه می‌شود) فقط برای قوانین معنا دارد. پارسر پرونده‌ها (که بعداً
زیر cases/ ساخته می‌شود) کاملاً منطق دیگری خواهد داشت (استخراج نام متهم،
موضوع اتهام، تاریخ رأی و ...) و نباید با این فایل قاطی شود.
"""

import re
from dataclasses import dataclass
from bs4 import BeautifulSoup
from laws.schemas import Law, Article, Note
from laws.case_types import case_type_of


ARTICLE_PATTERN_DEFAULT = re.compile(r"^ماده\s*(\d+)")
NOTE_PATTERN_DEFAULT    = re.compile(r"^تبصره\s*(\d*)")
REFERENCE_PATTERN       = re.compile(r"ماده\s+(\d+)")

# کاراکترهای نامرئی رایج در متن‌های دیجیتایز/OCR‌شده‌ی قدیمی (مثل قوانین
# مصوب سال‌های خیلی قدیم روی qavanin.ir): ZWNJ, ZWJ, LRM, RLM, BOM.
# اگر این‌ها را پاک نکنیم، regex با ^ (ابتدای رشته) شکست می‌خورد و آن ماده
# به‌جای این‌که یک ماده‌ی جدید تشخیص داده شود، به متن ماده‌ی قبلی می‌چسبد.
_INVISIBLE_CHARS_PATTERN = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\ufeff]")


def _clean_text(text: str) -> str:
    """حذف کاراکترهای نامرئی + یکدست‌کردن فاصله‌ها، قبل از هر تشخیص الگو"""
    text = _INVISIBLE_CHARS_PATTERN.sub("", text)
    return text.strip()


@dataclass
class LawParseConfig:
    """
    تنظیمات مخصوص هر قانون.
    اگر بعد از بررسیِ HTML واقعیِ یک قانون دیدید کلاس یا الگوی متفاوتی دارد،
    فقط همین‌جا یک نمونه‌ی جدید با مقادیر متفاوت بسازید — parser.py دست‌نخورده می‌ماند.

    توجه: case_type اینجا تایپ نمی‌شود — همیشه به‌صورت خودکار از روی domain
    محاسبه می‌شود (نگاه کنید به laws/case_types.py) تا هیچ‌وقت این دو
    ناهماهنگ نشوند.
    """
    law_name: str                         # "قانون مدنی"
    domain: str                           # "مدنی"
    article_tag: str = "p"
    article_class: str = "SecTex"
    article_pattern: re.Pattern = ARTICLE_PATTERN_DEFAULT
    note_pattern: re.Pattern = NOTE_PATTERN_DEFAULT

    @property
    def case_type(self) -> str:
        """حقوقی | کیفری — همیشه مشتق از domain، هرگز دستی وارد نمی‌شود"""
        return case_type_of(self.domain)


def _detect_status(text: str) -> str:
    if re.search(r"اصلاح[ىي]|الحاق[ىي]", text):
        return "amended"
    if re.search(r"منسوخ|موقوف", text):
        return "abolished"
    if re.search(r"تفسیر|تفسير", text):
        return "interpreted"
    return "active"


def _extract_references(text: str, self_num: int) -> list[int]:
    return [int(n) for n in REFERENCE_PATTERN.findall(text) if int(n) != self_num]


def parse_law(soup: BeautifulSoup, config: LawParseConfig, url: str) -> Law:
    """
    پارس یک قانون بر اساس config مخصوص همان قانون.
    خروجی: Law با لیست تخت از Article ها (بدون هیچ سلسله‌مراتبی).
    """
    law = Law(name=config.law_name, domain=config.domain, case_type=config.case_type, url=url)
    current_article: Article | None = None

    elements = soup.find_all(config.article_tag, class_=config.article_class)

    for el in elements:
        text = _clean_text(el.get_text(separator=" ", strip=True))
        if not text:
            continue

        article_match = config.article_pattern.match(text)
        note_match    = config.note_pattern.match(text)

        if article_match:
            num = int(article_match.group(1))
            current_article = Article(
                article_number=num,
                content=text,
                law_name=config.law_name,
                domain=config.domain,
                case_type=config.case_type,
                status=_detect_status(text),
                references=_extract_references(text, num),
            )
            law.articles.append(current_article)

        elif note_match and current_article:
            note_num = int(note_match.group(1)) if note_match.group(1) else 1
            current_article.notes.append(Note(
                note_number=note_num,
                content=text,
                status=_detect_status(text),
            ))

        elif current_article:
            if current_article.notes:
                current_article.notes[-1].content += " " + text
            else:
                current_article.content += " " + text

    return law