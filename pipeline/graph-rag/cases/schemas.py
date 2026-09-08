"""
cases/schemas.py — data structures dedicated to the Fact Graph (rulings/cases)

Why under cases/ and not a generic models/?
    Exactly for the same reason laws/schemas.py is separate: Ruling has a
    completely different structure from Article and the two must not be
    mixed together.

Why do we only keep court_level + text and not finer-grained fields (case
number, judges) per section?
    Because the format of these fields varies drastically between court
    tiers (first instance/appeal/supreme court/opinion) and between
    different years (e.g. "case number" in one place, "docket number"
    ("پرونده کلاسه") in another, and sometimes values are censored with
    "*"). Instead of writing a fragile regex for every case, we keep the
    full text of each section so it can be processed more precisely in a
    later stage (concept extraction with an LLM for the Ontology Graph).

Why can a single page (one Ruling) have several "sections" (RulingSection)?
    Because the site keeps "aggregated case text" enabled by default —
    meaning that when a case went through several rounds of appeal/retrial,
    all of its historical layers (each separated by its own <h1>) are
    concatenated together on a single page.
"""

from dataclasses import dataclass, field


@dataclass
class CitedArticle:
    """A statutory article mentioned in this ruling's "citations" field"""
    article_number: int
    law_name: str          # statute name exactly as written on the site, e.g. "قانون مدنی"
    note_number: int | None = None   # set if the citation was to a specific note of this article (not the whole article)


@dataclass
class RulingSection:
    """One section of the aggregated text — one litigation tier (first instance | appeal | supreme court | opinion | ...)"""
    court_level: str       # the exact h1 text, e.g. "رأی دادگاه بدوی" or "نظریه ..."
    text: str               # the full text of this section, with no further processing


@dataclass
class Ruling:
    """One page/ruling on the National Rulings System (key = ruling_id from the URL)"""
    ruling_id: str                          # from the URL, e.g. "32208"
    title: str                              # the case's official title on the site
    legal_factual_summary: str              # "پیام" — the official summary written by the research institute
    verdict_number: str                     # final verdict number (for this ruling_id)
    verdict_date: str                       # final verdict date
    case_type: str                          # civil | criminal (from "گروه رأی")
    cited_articles: list[CitedArticle] = field(default_factory=list)
    # Statutory articles not mentioned in the official "citations" box
    # (since that field is manually edited and sometimes left empty), but
    # which the ruling's own text explicitly cites. Only extracted for our
    # 5 core statutes (since only those exist in the Rule Graph and can be
    # linked). Because it's extracted from free text, it's less reliable
    # than cited_articles — the Auditor stage must ultimately validate it.
    text_cited_articles: list[CitedArticle] = field(default_factory=list)
    related_ruling_ids: list[str] = field(default_factory=list)   # other tiers of the same case
    sections: list[RulingSection] = field(default_factory=list)   # aggregated text, split by h1
    url: str = ""