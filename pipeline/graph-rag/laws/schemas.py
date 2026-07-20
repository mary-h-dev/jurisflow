
from dataclasses import dataclass, field


@dataclass
class Note:
    """تبصره‌ی یک ماده"""
    note_number: int
    content: str
    status: str = "active"        # active | amended | abolished | interpreted


@dataclass
class Article:
    """یک ماده‌ی قانونی — واحد اصلی Rule Graph"""
    article_number: int
    content: str
    law_name: str                 # مثلاً "قانون مدنی"
    domain: str                   # مدنی | کیفری | تجاری | دادرسی مدنی | دادرسی کیفری
    case_type: str                # حقوقی | کیفری — همیشه مشتق از domain، دستی وارد نمی‌شود
    notes: list[Note] = field(default_factory=list)
    status: str = "active"
    references: list[int] = field(default_factory=list)


@dataclass
class Law:
    """یک قانون کامل، به صورت لیست تخت (flat) از مواد"""
    name: str
    domain: str
    case_type: str                # حقوقی | کیفری — همیشه مشتق از domain
    url: str
    articles: list[Article] = field(default_factory=list)