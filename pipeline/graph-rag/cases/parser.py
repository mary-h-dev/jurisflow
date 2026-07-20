"""
cases/parser.py — پارسر مخصوص رأی‌های سامانه ملی آراء (ara.jri.ac.ir)

منطق کلی:
    - container شماره ۰: عنوان + «پیام» + متادیتای پرونده (شماره دادنامه/
      تاریخ/گروه) + «مستندات» (مواد قانونی) + select#Judge_ID (پرونده‌های مرتبط)
    - div#treeText: چون «متن تجمیعی پرونده» به‌صورت پیش‌فرض فعاله، این div
      می‌تونه چند <h1> جدا داشته باشه (هر کدوم یک لایه‌ی دادرسی: بدوی،
      تجدیدنظر، دیوان عالی، ...). چون id این h1 ها روی سایت تکراری و
      غیرقابل‌اعتماده (دیده شده: دو h1 با id="titr6")، فقط بر اساس
      ترتیب واقعی ظاهرشدن‌شون در صفحه (نه id) از هم جدا می‌شن.
"""

import re
from bs4 import BeautifulSoup
from cases.schemas import Ruling, RulingSection, CitedArticle


PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

# نام‌های کامل («قانون ...») ۵ قانون مادر و مترادف‌های رایج‌شون روی این سایت.
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

# بعضی رأی‌ها فقط عبارت ترکیبی رو بدون پیشوند «قانون» می‌آرن (مثلاً
# «مواد ۳۴۸ آیین دادرسی مدنی»). فقط برای عبارت‌های به‌قدر کافی مشخص
# (نه کلمات تک مثل «مدنی» به‌تنهایی، که خیلی مبهم و پرتکرارن) این حالت
# رو هم می‌پذیریم.
BARE_LAW_NAMES = [
    "آیین دادرسی مدنی",
    "آئین دادرسی مدنی",
    "آیین دادرسی کیفری",
    "آئین دادرسی کیفری",
]

# نگاشت هر نام/مترادفی که روی سامانه‌ی آراء دیده می‌شه، به همون نام رسمی
# که در laws/configs.py برای اسکرپ خودِ قانون استفاده کردیم.
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

# برای سازگاری با کد قدیمی/بقیه‌ی فایل
KNOWN_LAW_NAMES = FULL_LAW_NAMES


def _normalize_law_name(name: str) -> str:
    """اگر نام شناخته‌شده بود به نام رسمی نگاشت می‌کنه، وگرنه دست‌نخورده برمی‌گردونه."""
    return LAW_NAME_TO_CANONICAL.get(name.strip(), name.strip())


_INVISIBLE_CHARS_PATTERN = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\ufeff]")


def _clean_free_text(text: str) -> str:
    """
    نرمال‌سازی متن آزاد قبل از هر تشخیص الگو: تبدیل رقم فارسی به لاتین،
    یکدست‌کردن نیم‌فاصله/فاصله (که باعث می‌شد نام‌های ترکیبی مثل
    «دادگاه‌های» با رسم‌الخط‌های مختلف match نشن)، و جمع‌کردن فاصله‌های
    پشت‌سرهم.
    """
    text = text.translate(PERSIAN_DIGITS)
    text = _INVISIBLE_CHARS_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text


_REFERENTIAL_PATTERN = re.compile(
    r"^قانون(?:\s+[\u0600-\u06FF]+){0,4}?\s+"
    r"(?:مذکور|موصوف|مارالذکر|فوق[\s-]?الذکر|یادشده|اخیرالذکر|لاحق|سابق)\b"
)
# «آن قانون» / «همان قانون» — برخلاف بقیه‌ی کلمات ارجاعی، اینجا کلمه‌ی
# اشاره («آن»/«همان») *قبل* از «قانون» می‌آد، نه بعدش — پس باید متنِ
# پیش از رخداد «قانون» رو چک کنیم، نه بعدش.
_BACKWARD_REFERENTIAL_PATTERN = re.compile(r"(?:آن|همان)\s*$")

_NUMBER_GROUP_PATTERN = re.compile(r"(?:ماده|مواد)\s*((?:\d+\s*(?:و|،|,)?\s*)+)")
# حالت خاص: «مواد X ... و Y قانون دیگر» — Y بدون «ماده» جلوش میاد ولی با
# «و» به فهرست قبلی وصله. فقط وقتی این عدد *دقیقاً ابتدای پنجره* باشه
# (بلافاصله بعد از مرز قانون قبلی) در نظر می‌گیریم — تا با اعداد نامرتبط
# دیگه (تاریخ، مبلغ، شماره پرونده) قاطی نشه.
_LEADING_CONTINUATION_PATTERN = re.compile(r"^\s*(?:و|،)\s*(\d+)\b")


# تشخیص «تبصره N از» (یا بدون «از») بلافاصله قبل از «ماده» — فقط وقتی
# دقیقاً یک شماره ماده در همون گروه باشه به‌عنوان تبصره‌ی همون ماده در
# نظر گرفته می‌شه (تا با فهرست چندماده‌ای که تبصره‌شون نامشخصه قاطی نشه).
_TABSARE_PREFIX_PATTERN = re.compile(r"تبصره\s*(\d+)\s*(?:از\s+)?$")


def _numbers_in_window(window: str) -> list[tuple[int, int | None, int]]:
    """خروجی: لیستی از (شماره_ماده, شماره_تبصره_یا_None, موقعیت_تطبیق_در_پنجره)"""
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
    پیدا کردن همه‌ی «مرزها» در متن: هر رخداد کلمه‌ی «قانون» (چه یکی از
    ۵ قانون مادرمون باشه، چه قانونی کاملاً ناشناخته مثل «قانون تشکیل
    دادگاه‌های عمومی») + هر رخداد نام ترکیبی بدون پیشوند.

    چرا قانون‌های ناشناخته هم مرز حساب می‌شن؟
    چون اگه مرزبندی نکنیم، پنجره‌ی عقب‌گرد برای پیدا کردن «ماده» یک
    قانونِ شناخته‌شده‌ی *بعدی* می‌تونه از وسط یک قانون ناشناخته‌ی دیگه رد
    بشه و شماره‌ماده‌ی اون رو به اشتباه بدزده.

    خروجی هر مرز: (start, end, نام_رسمی_یا_None, آیا_ارجاعی_است)
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

        # قانونِ ناشناخته — فقط مرز، بدون انتساب
        fences.append((start, start + len("قانون"), None, False))

    for bare in BARE_LAW_NAMES:
        for m in re.finditer(re.escape(bare), text):
            start = m.start()
            end = start + len(bare)
            # اگه این عبارت داخل محدوده‌ی یکی از مرزهای «قانون ...» که
            # قبلاً پیدا کردیم افتاده، یعنی جزئی از همون مرزه، دوباره
            # اضافه نکن. (توجه: این بررسیِ دقیقِ containment است، نه یک
            # حدسِ فاصله‌ای مثل چک‌کردن N کاراکتر قبل — چون آن حالت باعث
            # می‌شد اگه یک «قانون» نامرتبط تصادفاً نزدیک بود، این fence
            # به‌اشتباه حذف بشه و شماره‌ماده به مرز غلط بعدی نشت کنه.)
            if any(fs <= start < fe for fs, fe in full_spans):
                continue
            fences.append((start, end, LAW_NAME_TO_CANONICAL[bare], False))

    fences.sort(key=lambda f: f[0])
    return fences


def _extract_articles_from_free_text(
    text: str, window_size: int = 400, forward_window: int = 60
) -> list[CitedArticle]:
    """
    استخراج «حدسی» مواد قانونی از متن آزاد رأی — مکمل جعبه‌ی رسمی
    «مستندات» که گاهی خالی می‌مونه. پشتیبانی می‌کنه از:
        - نام‌های سنتی/طولانی قانون (با لنگرِ FULL_LAW_NAMES)
        - ارجاعات ضمنی («قانون مذکور»، «قانون موصوف»، «آن قانون»، ...)
        - کلمه‌ی جمع «مواد» با چند شماره‌ی جدا با «و»/«،»
        - مرزبندی درست بین قانون‌های مختلف (حتی وقتی یکی‌شون ناشناخته باشه)
        - تبصره‌ی مشخصِ یک ماده (نه کل ماده)
        - هر دو جهت جمله: «ماده N ... قانون X» (عقب‌گرد) و
          «قانون X ... ماده N» (جلوگرد، مثلاً «قانون مدنی، ماده ۱۹۹»)

    برای جلوگیری از دوبار-شمارش وقتی هر دو جهت به یک عدد می‌رسن، ابتدا
    پاسِ عقب‌گرد (که پرکاربردتره) اجرا و موقعیت‌هاش ثبت می‌شه؛ پاسِ
    جلوگرد فقط عددهایی رو اضافه می‌کنه که قبلاً claim نشدن.
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
    claimed_positions: set[int] = set()   # موقعیت مطلقِ شروع هر عدد که قبلاً استفاده شده

    # پاس ۱ — عقب‌گرد (رفتار اصلی و پیش‌فرض)
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

    # پاس ۲ — جلوگرد (مکمل، فقط برای «قانون X ... ماده N»)
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
    دو فرمت دیده شده در «مستندات»:
        "ماده 312 قانون تجارت-ماده 2 قانون اصلاح موادی از قانون صدور چک-"
            → هر بخش (جداشده با -) یک ماده و یک قانون مجزا
        "ماده 399 ماده 401 قانون مدنی-"
            → چند ماده‌ی متوالی که یک نام قانون مشترک دارن (بدون - بین‌شون)
    راه‌حل: ابتدا با "-" گروه‌بندی می‌کنیم؛ داخل هر گروه همه‌ی شماره‌ماده‌ها
    را جدا استخراج می‌کنیم و نام قانون را از انتهای گروه (بعد از آخرین
    "ماده <عدد>") می‌گیریم — این هر دو فرمت را پوشش می‌دهد.
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

        # اگر دقیقاً یک ماده در این بخش بود، چک کن آیا بلافاصله قبلش
        # «تبصره N از» اومده (یعنی استناد به یک تبصره‌ی خاص بوده، نه کل ماده)
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
    تقسیم div#treeText به بخش‌های جدا بر اساس <h1> ها.
    چون id این h1 ها تکراری/غیرقابل‌اعتماده، فقط از شیء واقعی (identity)
    برای تشخیص «به h1 بعدی رسیدیم» استفاده می‌کنیم، نه از id یا متنش.
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
            # یک <div> معمولاً یعنی به منوی «فهرست» انتهای صفحه رسیدیم؛
            # محتوای واقعی رأی همیشه متن ساده + <br> است، نه div تودرتو.
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

    # روی متن همه‌ی بخش‌ها با هم (نه هرکدوم جدا) اجرا می‌شه، چون یک ماده
    # ممکنه در یک بخش ذکر بشه ولی نام قانون در جمله‌ی بعدی/بخش دیگه بیاد
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