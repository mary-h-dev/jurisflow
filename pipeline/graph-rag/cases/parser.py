"""
cases/parser.py — parser dedicated to rulings from the National Rulings
                   System (ara.jri.ac.ir)

General logic:
    - container index 0: title + "message" + case metadata (verdict number/
      date/group) + "citations" (statutory articles) + select#Judge_ID
      (related rulings)
    - div#treeText: since "aggregated case text" is enabled by default,
      this div can contain several separate <h1> elements (each one a
      litigation tier: first instance, appeal, supreme court, ...). Since
      the id of these h1 elements is duplicated and unreliable on the site
      (observed: two h1 with id="titr6"), they are separated purely based
      on their actual order of appearance on the page (not by id).
"""

import re
from bs4 import BeautifulSoup
from cases.schemas import Ruling, RulingSection, CitedArticle


PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

# Full names ("قانون ...") of our 5 core statutes and their common
# synonyms as they appear on this site.
FULL_LAW_NAMES = [
    "قانون مدنی",
    "قانون تجارت",
    "قانون مجازات اسلامی",
    "قانون آیین دادرسی مدنی",
    "قانون آئین دادرسی مدنی",
    "قانون آیین دادرسی دادگاه های عمومی و انقلاب در امور مدنی",
    "قانون آیین دادرسی کیفری",
    "قانون آئین دادرسی کیفری",
    "قانون آیین دادرسی دادگاه های عمومی و انقلاب در امور کیفری",
]

# Some rulings only give the bare compound phrase without the "قانون"
# prefix (e.g. "مواد ۳۴۸ آیین دادرسی مدنی"). We only accept this for
# phrases that are specific enough (not single words like "مدنی" alone,
# which would be far too ambiguous and too common).
BARE_LAW_NAMES = [
    "آیین دادرسی مدنی",
    "آئین دادرسی مدنی",
    "آیین دادرسی کیفری",
    "آئین دادرسی کیفری",
]

# Maps every name/synonym seen on the rulings system to the same
# canonical name we used for scraping the statute itself in laws/configs.py.
LAW_NAME_TO_CANONICAL = {
    "قانون مدنی": "قانون مدنی",
    "قانون تجارت": "قانون تجارت",
    "قانون مجازات اسلامی": "قانون مجازات اسلامی",
    "قانون آیین دادرسی مدنی": "قانون آیین دادرسی مدنی",
    "قانون آئین دادرسی مدنی": "قانون آیین دادرسی مدنی",
    "قانون آیین دادرسی دادگاه های عمومی و انقلاب در امور مدنی": "قانون آیین دادرسی مدنی",
    "قانون آیین دادرسی کیفری": "قانون آیین دادرسی کیفری",
    "قانون آئین دادرسی کیفری": "قانون آیین دادرسی کیفری",
    "قانون آیین دادرسی دادگاه های عمومی و انقلاب در امور کیفری": "قانون آیین دادرسی کیفری",
    "آیین دادرسی مدنی": "قانون آیین دادرسی مدنی",
    "آئین دادرسی مدنی": "قانون آیین دادرسی مدنی",
    "آیین دادرسی کیفری": "قانون آیین دادرسی کیفری",
    "آئین دادرسی کیفری": "قانون آیین دادرسی کیفری",
}


def _normalize_law_name(name: str) -> str:
    """Maps a name to its canonical form if known, otherwise returns it unchanged."""
    return LAW_NAME_TO_CANONICAL.get(name.strip(), name.strip())


_INVISIBLE_CHARS_PATTERN = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\ufeff]")


def _clean_free_text(text: str) -> str:
    """
    Normalize free text before any pattern matching: convert Persian
    digits to Latin, unify ZWNJ/spacing (which was causing compound names
    like "دادگاه‌های" to fail to match across different spelling
    conventions), and collapse consecutive whitespace.
    """
    text = text.translate(PERSIAN_DIGITS)
    text = _INVISIBLE_CHARS_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text


_REFERENTIAL_PATTERN = re.compile(
    r"^قانون(?:\s+[\u0600-\u06FF]+){0,4}?\s+"
    r"(?:مذکور|موصوف|مارالذکر|فوق[\s-]?الذکر|یادشده|اخیرالذکر|لاحق|سابق)\b"
)
# "آن قانون" / "همان قانون" — unlike the other referential words, here the
# demonstrative word ("آن"/"همان") comes *before* "قانون", not after — so
# we need to check the text preceding the "قانون" occurrence, not after it.
_BACKWARD_REFERENTIAL_PATTERN = re.compile(r"(?:آن|همان)\s*$")

_NUMBER_GROUP_PATTERN = re.compile(r"(?:ماده|مواد)\s*((?:\d+\s*(?:و|،|,)?\s*)+)")
# Special case: "مواد X ... و Y قانون دیگر" — Y comes without "ماده" in
# front of it but is joined to the previous list with "و". We only accept
# this number when it's *exactly at the start of the window* (immediately
# after the previous law boundary), so it doesn't get mixed up with other
# unrelated numbers (dates, amounts, case numbers).
_LEADING_CONTINUATION_PATTERN = re.compile(r"^\s*(?:و|،)\s*(\d+)\b")


# Detects "تبصره N از" (or without "از") immediately before "ماده" — only
# treated as a note of that same article when there is exactly one article
# number in the same group (so it doesn't get mixed up with a multi-article
# list whose note is ambiguous).
_TABSARE_PREFIX_PATTERN = re.compile(r"تبصره\s*(\d+)\s*(?:از\s+)?$")


def _numbers_in_window(window: str) -> list[tuple[int, int | None, int]]:
    """Output: a list of (article_number, note_number_or_None, match_position_in_window)"""
    results = []

    for m in _NUMBER_GROUP_PATTERN.finditer(window):
        numbers = [int(n) for n in re.findall(r"\d+", m.group(1))]

        note_number = None
        if len(numbers) == 1:
            preceding = window[:m.start()]
            tabsare_match = _TABSARE_PREFIX_PATTERN.search(preceding)
            if tabsare_match:
                note_number = int(tabsare_match.group(1))

        for num in numbers:
            results.append((num, note_number if len(numbers) == 1 else None, m.start()))

    lead = _LEADING_CONTINUATION_PATTERN.match(window)
    if lead:
        results.append((int(lead.group(1)), None, lead.start()))

    return results


def _find_law_fences(text: str) -> list[tuple[int, int, str | None, bool]]:
    """
    Finds all "fences" (boundaries) in the text: every occurrence of the
    word "قانون" (whether one of our 5 core statutes or a completely
    unknown statute like "قانون تشکیل دادگاه‌های عمومی") + every occurrence
    of a bare compound name without the prefix.

    Why do unknown statutes count as fences too?
    Because without fencing them off, the backward-looking window used to
    find the "ماده" belonging to the *next* known statute could reach back
    through the middle of another, unknown statute and incorrectly steal
    its article number.

    Output for each fence: (start, end, canonical_name_or_None, is_referential)
    """
    fences = []
    full_spans: list[tuple[int, int]] = []

    for m in re.finditer(r"قانون", text):
        start = m.start()
        ahead = text[start:start + 90]
        behind = text[max(0, start - 6):start]

        matched_full = None
        for name in sorted(FULL_LAW_NAMES, key=len, reverse=True):
            if ahead.startswith(name):
                matched_full = LAW_NAME_TO_CANONICAL[name]
                end = start + len(name)
                break

        if matched_full:
            fences.append((start, end, matched_full, False))
            full_spans.append((start, end))
            continue

        ref_match = _REFERENTIAL_PATTERN.match(ahead)
        if ref_match:
            fences.append((start, start + ref_match.end(), None, True))
            continue

        if _BACKWARD_REFERENTIAL_PATTERN.search(behind):
            fences.append((start, start + len("قانون"), None, True))
            continue

        # Unknown statute — a fence only, with no attribution
        fences.append((start, start + len("قانون"), None, False))

    for bare in BARE_LAW_NAMES:
        for m in re.finditer(re.escape(bare), text):
            start = m.start()
            end = start + len(bare)
            # If this phrase falls inside the span of a "قانون ..." fence
            # we already found, it's part of that same fence — don't add
            # it again. (Note: this is an exact containment check, not a
            # distance heuristic like checking N characters before —
            # because that would incorrectly drop this fence whenever an
            # unrelated "قانون" happened to be nearby, letting the article
            # number leak into the wrong subsequent fence.)
            if any(fs <= start < fe for fs, fe in full_spans):
                continue
            fences.append((start, end, LAW_NAME_TO_CANONICAL[bare], False))

    fences.sort(key=lambda f: f[0])
    return fences


def _extract_articles_from_free_text(
    text: str, window_size: int = 400, forward_window: int = 60
) -> list[CitedArticle]:
    """
    "Best-effort" extraction of statutory articles from the ruling's free
    text — complementing the official "citations" box, which is sometimes
    left empty. Handles:
        - traditional/long statute names (anchored on FULL_LAW_NAMES)
        - implicit references ("قانون مذکور", "قانون موصوف", "آن قانون", ...)
        - the plural word "مواد" with several numbers separated by "و"/"،"
        - correct boundaries between different statutes (even when one of
          them is unknown)
        - a specific note of an article (not the whole article)
        - both sentence directions: "ماده N ... قانون X" (backward) and
          "قانون X ... ماده N" (forward, e.g. "قانون مدنی، ماده ۱۹۹")

    To avoid double-counting when both directions reach the same number,
    the backward pass (the more common one) runs first and records its
    positions; the forward pass only adds numbers that haven't already
    been claimed.
    """
    text = _clean_free_text(text)
    fences = _find_law_fences(text)

    resolved: list[tuple[int, int, str | None]] = []
    last_known_law: str | None = None
    for start, end, law, is_referential in fences:
        if law:
            attributed = law
            last_known_law = law
        elif is_referential and last_known_law:
            attributed = last_known_law
        else:
            attributed = None
        resolved.append((start, end, attributed))

    results: list[CitedArticle] = []
    claimed_positions: set[int] = set()   # absolute start position of every number already used

    # Pass 1 — backward (the main, default behavior)
    prev_fence_end = 0
    for start, end, attributed in resolved:
        if attributed:
            window_start = max(prev_fence_end, start - window_size, 0)
            window = text[window_start:start]
            for num, note_num, match_pos in _numbers_in_window(window):
                abs_pos = window_start + match_pos
                claimed_positions.add(abs_pos)
                results.append(CitedArticle(article_number=num, law_name=attributed, note_number=note_num))
        prev_fence_end = end

    # Pass 2 — forward (complementary, only for "قانون X ... ماده N")
    for idx, (start, end, attributed) in enumerate(resolved):
        if not attributed:
            continue
        next_start = resolved[idx + 1][0] if idx + 1 < len(resolved) else len(text)
        window_end = min(next_start, end + forward_window)
        window = text[end:window_end]
        for num, note_num, match_pos in _numbers_in_window(window):
            abs_pos = end + match_pos
            if abs_pos in claimed_positions:
                continue
            claimed_positions.add(abs_pos)
            results.append(CitedArticle(article_number=num, law_name=attributed, note_number=note_num))

    return results


def _to_latin_digits(text: str) -> str:
    return text.translate(PERSIAN_DIGITS)


def _parse_cited_articles(mostanadat_text: str) -> list[CitedArticle]:
    """
    Two formats observed in the "citations" ("مستندات") field:
        "ماده 312 قانون تجارت-ماده 2 قانون اصلاح موادی از قانون صدور چک-"
            → each segment (split on -) is one article and one separate statute
        "ماده 399 ماده 401 قانون مدنی-"
            → several consecutive articles sharing one common statute name
              (with no - between them)
    Solution: first group by "-"; within each group, extract all article
    numbers separately and take the statute name from the end of the group
    (after the last "ماده <number>") — this covers both formats.
    """
    text = _to_latin_digits(mostanadat_text)
    articles = []
    for chunk in text.split("-"):
        chunk = chunk.strip()
        if not chunk:
            continue

        article_numbers = [int(n) for n in re.findall(r"ماده\s*(\d+)", chunk)]
        if not article_numbers:
            continue

        law_name_match = re.search(r"(?:ماده\s*\d+\s*)+(.+)", chunk)
        law_name = law_name_match.group(1).strip() if law_name_match else ""
        law_name = _normalize_law_name(law_name)

        # If this segment had exactly one article, check whether it was
        # immediately preceded by "تبصره N از" (meaning the citation was to
        # a specific note, not the whole article)
        note_number = None
        if len(article_numbers) == 1:
            prefix = chunk[:re.search(r"ماده\s*\d+", chunk).start()]
            tabsare_match = _TABSARE_PREFIX_PATTERN.search(prefix)
            if tabsare_match:
                note_number = int(tabsare_match.group(1))

        for num in article_numbers:
            articles.append(CitedArticle(
                article_number=num, law_name=law_name,
                note_number=note_number if len(article_numbers) == 1 else None,
            ))

    return articles


def _extract_metadata_box(container0) -> dict:
    result = {"verdict_number": "", "verdict_date": "", "case_type": ""}

    meta_td = container0.find("td", class_="font-size-small")
    if not meta_td:
        return result

    text = _to_latin_digits(meta_td.get_text(separator=" ", strip=True))

    num_match = re.search(r"شماره دادنامه قطعی\s*:\s*(\d+)", text)
    if num_match:
        result["verdict_number"] = num_match.group(1)

    date_match = re.search(r"تاریخ دادنامه قطعی\s*:\s*([\d/]+)", text)
    if date_match:
        result["verdict_date"] = date_match.group(1)

    group_match = re.search(r"گروه رأی\s*:\s*(حقوقی|کیفری)", text)
    if group_match:
        result["case_type"] = group_match.group(1)

    return result


def _extract_related_ruling_ids(container0) -> list[str]:
    select = container0.find("select", id="Judge_ID")
    if not select:
        return []
    return [opt.get("value") for opt in select.find_all("option") if opt.get("value")]


def _extract_title_and_summary(container0) -> tuple[str, str]:
    title = ""
    summary = ""

    h1 = container0.find("h1", class_="Title3D")
    if h1:
        title = re.sub(r"^عنوان\s*:\s*", "", h1.get_text(separator=" ", strip=True)).strip()

    for b_tag in container0.find_all("b"):
        if "پیام" in b_tag.get_text():
            parent_span = b_tag.parent
            summary = parent_span.get_text(separator=" ", strip=True)
            summary = re.sub(r"^پیام\s*:\s*", "", summary).strip()
            break

    return title, summary


def _extract_cited_articles_text(container0) -> str:
    for b_tag in container0.find_all("b"):
        if "مستندات" in b_tag.get_text():
            parent_span = b_tag.parent
            text = parent_span.get_text(separator=" ", strip=True)
            return re.sub(r"^مستندات\s*:\s*", "", text).strip()
    return ""


def _split_into_sections(tree_text_div) -> list[RulingSection]:
    """
    Splits div#treeText into separate sections based on <h1> elements.
    Since the id of these h1 elements is duplicated/unreliable, only the
    actual object identity is used to detect "we've reached the next h1",
    not its id or text.
    """
    headers = tree_text_div.find_all("h1")
    if not headers:
        return []

    header_object_ids = {id(h) for h in headers}
    sections = []

    for h1 in headers:
        level = h1.get_text(strip=True)
        parts = []
        for sib in h1.next_siblings:
            if id(sib) in header_object_ids:
                break
            # A <div> usually means we've reached the "index" menu at the
            # end of the page; the actual ruling content is always plain
            # text + <br>, never a nested div.
            if getattr(sib, "name", None) == "div":
                break
            if hasattr(sib, "get_text"):
                parts.append(sib.get_text(separator=" ", strip=True))
            else:
                parts.append(str(sib).strip())

        text = " ".join(p for p in parts if p)
        text = _to_latin_digits(text)
        sections.append(RulingSection(court_level=level, text=text))

    return sections


def parse_ruling(soup: BeautifulSoup, ruling_id: str, url: str) -> Ruling:
    containers = soup.find_all("div", class_="container")
    container0 = containers[0] if containers else None

    title, summary = "", ""
    meta = {"verdict_number": "", "verdict_date": "", "case_type": ""}
    related_ids: list[str] = []
    cited_text = ""

    if container0:
        title, summary = _extract_title_and_summary(container0)
        meta = _extract_metadata_box(container0)
        related_ids = _extract_related_ruling_ids(container0)
        cited_text = _extract_cited_articles_text(container0)

    tree_text_div = soup.find("div", id="treeText")
    sections = _split_into_sections(tree_text_div) if tree_text_div else []

    cited_articles = _parse_cited_articles(cited_text)

    # Runs on the combined text of all sections together (not each one
    # separately), because an article might be mentioned in one section
    # while the statute name appears in the next sentence/another section
    full_text_all_sections = " ".join(s.text for s in sections)
    text_cited_articles = _extract_articles_from_free_text(full_text_all_sections)

    return Ruling(
        ruling_id=ruling_id,
        title=title,
        legal_factual_summary=summary,
        verdict_number=meta["verdict_number"],
        verdict_date=meta["verdict_date"],
        case_type=meta["case_type"],
        cited_articles=cited_articles,
        text_cited_articles=text_cited_articles,
        related_ruling_ids=related_ids,
        sections=sections,
        url=url,
    )