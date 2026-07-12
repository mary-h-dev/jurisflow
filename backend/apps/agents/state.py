from __future__ import annotations
from typing import TypedDict


# ── Sub-structures ────────────────────────────────────────────────────────────

class Party(TypedDict):
    role:    str   # خواهان | خوانده | متهم | شاکی | ثالث
    name:    str
    details: str


class RelatedLaw(TypedDict):
    article_ref: str    # مثل "ماده ۱۹۰" یا "ماده ۱۴۱ مکرر"
    law_name:    str
    summary:     str    # خلاصه — نه متن کامل
    relevance:   float


class TeammateOpinion(TypedDict):
    """نظر هر teammate بعد از debate"""
    role:       str        # defender | prosecutor | judge
    position:   str        # موضع کلی
    arguments:  list[str]  # استدلال‌ها
    cited_articles: list[str]  # مواد قانونی مستند
    confidence: str        # بالا | متوسط | پایین


class Risk(TypedDict):
    title:       str
    description: str
    severity:    str   # بالا | متوسط | پایین


# ── Main State ────────────────────────────────────────────────────────────────

class CaseState(TypedDict):
    """
    State مشترک بین همه agents.

    طراحی:
    - raw_text بعد از analyzer پاک میشه
    - teammate ها به صورت parallel کار می‌کنن
    - lead نتیجه نهایی رو از opinions می‌گیره
    """

    # ── ورودی ────────────────────────────────────────────────────────────────
    raw_text:   str
    user_id:    int
    session_id: str

    # ── خروجی Analyzer ───────────────────────────────────────────────────────
    case_subject:    str | None
    case_type:       str | None
    parties:         list[Party] | None
    key_facts:       list[str]   | None
    legal_keywords:  list[str]   | None

    # ── خروجی Searcher ───────────────────────────────────────────────────────
    related_laws:    list[RelatedLaw] | None
    cited_articles:  list[str]        | None

    # ── خروجی Teammates (parallel) ───────────────────────────────────────────
    defender_opinion:   TeammateOpinion | None   # وکیل مدافع
    prosecutor_opinion: TeammateOpinion | None   # مخالف/دادستان
    judge_opinion:      TeammateOpinion | None   # قاضی بی‌طرف

    # ── خروجی Lead Agent ─────────────────────────────────────────────────────
    strengths:      list[str] | None
    weaknesses:     list[str] | None
    risks:          list[Risk] | None
    win_chance:     str | None   # بالا | متوسط | پایین
    strategy:       str | None   # تهاجمی | دفاعی | مصالحه
    recommendation: str | None
    next_steps:     list[str] | None

    # ── Metadata ──────────────────────────────────────────────────────────────
    error:           str | None
    completed_nodes: list[str]