"""
laws/parser.py — Parser dedicated to the 5 core statutes (Rule Graph)

This file lives under laws/ because its logic (detecting "Article"/"Note",
and the diagnostic checklist to be added later) only makes sense for statutes.
The parser for case rulings (to be built later under cases/) will have
completely different logic (extracting defendant name, charge subject,
verdict date, etc.) and must not be mixed with this file.
"""

import re
from dataclasses import dataclass
from bs4 import BeautifulSoup
from laws.schemas import Law, Article, Note
from laws.case_types import case_type_of


ARTICLE_PATTERN_DEFAULT = re.compile(r"^ماده\s*(\d+)")
NOTE_PATTERN_DEFAULT    = re.compile(r"^تبصره\s*(\d*)")
REFERENCE_PATTERN       = re.compile(r"ماده\s+(\d+)")

# Common invisible characters found in old digitized/OCR'd texts (such as
# statutes enacted many years ago on qavanin.ir): ZWNJ, ZWJ, LRM, RLM, BOM.
# If these are not stripped, the regex anchored at ^ (start of string) will
# fail to match, and that article will get appended to the previous
# article's text instead of being recognized as a new article.
_INVISIBLE_CHARS_PATTERN = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\ufeff]")


def _clean_text(text: str) -> str:
    """Strip invisible characters and normalize whitespace before any pattern matching."""
    text = _INVISIBLE_CHARS_PATTERN.sub("", text)
    return text.strip()


@dataclass
class LawParseConfig:
    """
    Configuration specific to each statute.
    If, after inspecting the actual HTML of a statute, you find it uses a
    different tag/class or pattern, just create a new instance here with
    different values — parser.py itself stays untouched.

    Note: case_type is not typed here — it is always computed automatically
    from domain (see laws/case_types.py) so the two can never become
    inconsistent.
    """
    law_name: str                         # e.g. "Civil Code"
    domain: str                           # e.g. "Civil"
    article_tag: str = "p"
    article_class: str = "SecTex"
    article_pattern: re.Pattern = ARTICLE_PATTERN_DEFAULT
    note_pattern: re.Pattern = NOTE_PATTERN_DEFAULT

    @property
    def case_type(self) -> str:
        """Civil | Criminal — always derived from domain, never entered manually."""
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
    Parse a single statute using the config specific to that statute.
    Output: a Law with a flat list of Articles (no hierarchy at all).
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