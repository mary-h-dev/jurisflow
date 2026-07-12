from __future__ import annotations

import logging

from django.conf import settings
from neo4j import GraphDatabase

from apps.agents.state import CaseState, RelatedLaw
from apps.search.services import SearchService

logger = logging.getLogger(__name__)

_search_service = SearchService()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_article_ref(article_number: int, law_name: str) -> str:
    """
    تبدیل عدد ماده به فرمت استاندارد فارسی.
    مثال: 141 → "ماده ۱۴۱"
    برای مواد مکرر باید از Neo4j بخونیم — اینجا base format رو می‌سازیم.
    """
    persian_digits = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
    num_persian    = str(article_number).translate(persian_digits)
    return f"ماده {num_persian} — {law_name}"


def _get_article_summary(article_number: int, law: str) -> str:
    """خلاصه ماده رو از Neo4j می‌گیره — نه متن کامل"""
    with _search_service.driver.session() as session:
        result = session.run(
            """
            MATCH (a:Article {article_number: $num, law: $law})
            RETURN coalesce(a.summary, substring(a.content, 0, 150)) AS summary
            """,
            num=article_number,
            law=law,
        )
        record = result.single()
        return record["summary"] if record else ""


# ── Main Node ─────────────────────────────────────────────────────────────────

def search_laws(state: CaseState) -> dict:
    """
    Agent 2 — Law Searcher

    وظایف:
    - از legal_keywords و case_subject برای جستجو استفاده می‌کنه
    - چند query مختلف می‌زنه و نتایج رو dedup می‌کنه
    - فقط summary ماده رو ذخیره می‌کنه (نه متن کامل) برای کاهش توکن
    """
    logger.info(f"[searcher] session={state['session_id']} started")

    if state.get("error"):
        logger.warning("[searcher] skipping due to previous error")
        return {"completed_nodes": state.get("completed_nodes", []) + ["search_laws"]}

    try:
        # ── ساخت query های جستجو ─────────────────────────────────────────────
        queries: list[str] = []

        # query اصلی از موضوع پرونده
        if state.get("case_subject"):
            queries.append(state["case_subject"])

        # query های اضافه از کلیدواژه‌ها — حداکثر ۳ تا
        keywords = state.get("legal_keywords") or []
        for kw in keywords[:3]:
            queries.append(kw)

        if not queries:
            logger.warning("[searcher] no queries to search")
            return {
                "related_laws":    [],
                "cited_articles":  [],
                "completed_nodes": state.get("completed_nodes", []) + ["search_laws"],
            }

        # ── جستجو برای هر query ───────────────────────────────────────────────
        law_name = "قانون مدنی"  # بعداً از case_type می‌گیریم
        seen_nums: set[int] = set()
        related_laws: list[RelatedLaw] = []
        cited_articles: list[str]      = []

        for query in queries:
            embedding = _search_service._embed(query)
            results   = _search_service._vector_search(
                embedding=embedding,
                law=law_name,
                top_k=5,
            )

            for r in results:
                if r.num in seen_nums:
                    continue
                seen_nums.add(r.num)

                # فقط summary بگیر — نه متن کامل
                summary = _get_article_summary(r.num, r.law)

                related_laws.append(RelatedLaw(
                    article_ref=_format_article_ref(r.num, r.law),
                    law_name=r.law,
                    summary=summary,
                    relevance=round(r.vector_score, 3),
                ))
                cited_articles.append(_format_article_ref(r.num, r.law))

        # مرتب‌سازی بر اساس relevance
        related_laws.sort(key=lambda x: x["relevance"], reverse=True)

        # حداکثر ۱۵ ماده — برای جلوگیری از context overflow
        related_laws   = related_laws[:15]
        cited_articles = cited_articles[:15]

        logger.info(
            f"[searcher] session={state['session_id']} "
            f"found={len(related_laws)} articles"
        )

        return {
            "related_laws":    related_laws,
            "cited_articles":  cited_articles,
            "completed_nodes": state.get("completed_nodes", []) + ["search_laws"],
            "error":           None,
        }

    except Exception as e:
        logger.error(f"[searcher] error: {e}", exc_info=True)
        return {
            "related_laws":    [],
            "cited_articles":  [],
            "error":           f"خطا در جستجوی قوانین: {str(e)}",
            "completed_nodes": state.get("completed_nodes", []) + ["search_laws"],
        }