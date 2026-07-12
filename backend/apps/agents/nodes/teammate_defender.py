from __future__ import annotations

import json
import logging

from django.conf import settings
from groq import Groq

from apps.agents.state import CaseState, TeammateOpinion

logger = logging.getLogger(__name__)

_groq = Groq(api_key=settings.GROQ_API_KEY)

PROMPT = """
تو یک وکیل مدافع متخصص و باتجربه در قوانین ایران هستی.
وظیفه تو دفاع از موکل و یافتن بهترین استدلال‌های حقوقی به نفع اوست.

اطلاعات پرونده:
- موضوع: {case_subject}
- نوع: {case_type}
- طرفین: {parties}
- وقایع کلیدی: {key_facts}
- قوانین مرتبط: {related_laws}

وظیفه تو:
۱. قوی‌ترین استدلال‌های دفاعی را بیاور
۲. به مواد قانونی که به نفع موکل است استناد کن
۳. موضع کلی دفاعی خود را مشخص کن
۴. میزان اطمینان خود را اعلام کن

فقط JSON برگردون — بدون هیچ توضیح اضافه:
{{
    "position": "موضع کلی دفاعی در یک جمله",
    "arguments": [
        "استدلال دفاعی اول",
        "استدلال دفاعی دوم",
        "استدلال دفاعی سوم"
    ],
    "cited_articles": [
        "ماده ۱۹۰ — قانون مدنی",
        "ماده ۱۴۱ مکرر — قانون مدنی"
    ],
    "confidence": "بالا | متوسط | پایین"
}}
"""


def defend(state: CaseState) -> dict:
    """
    Teammate 1 — وکیل مدافع

    وظایف:
    - قوی‌ترین استدلال‌های دفاعی رو پیدا می‌کنه
    - به مواد قانونی به نفع موکل استناد می‌کنه
    - با prosecutor و judge در debate شرکت می‌کنه
    """
    logger.info(f"[defender] session={state['session_id']} started")

    if state.get("error"):
        logger.warning("[defender] skipping due to previous error")
        return {
            "completed_nodes": state.get("completed_nodes", []) + ["defender"]
        }

    try:
        # ── آماده‌سازی context ────────────────────────────────────────────────
        parties_text = "\n".join([
            f"- {p['role']}: {p['name']}"
            for p in (state.get("parties") or [])
        ])

        key_facts_text = "\n".join([
            f"{i}. {fact}"
            for i, fact in enumerate((state.get("key_facts") or []), 1)
        ])

        related_laws_text = "\n".join([
            f"- {law['article_ref']}: {law['summary']}"
            for law in (state.get("related_laws") or [])[:8]  # حداکثر ۸ ماده
        ])

        response = _groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role":    "user",
                "content": PROMPT.format(
                    case_subject=state.get("case_subject", ""),
                    case_type=state.get("case_type", ""),
                    parties=parties_text or "نامشخص",
                    key_facts=key_facts_text or "نامشخص",
                    related_laws=related_laws_text or "موردی یافت نشد",
                ),
            }],
            temperature=0.2,  # کمی بالاتر برای خلاقیت در استدلال
        )

        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)

        opinion = TeammateOpinion(
            role="defender",
            position=data.get("position", ""),
            arguments=data.get("arguments", []),
            cited_articles=data.get("cited_articles", []),
            confidence=data.get("confidence", "متوسط"),
        )

        logger.info(
            f"[defender] session={state['session_id']} "
            f"confidence={opinion['confidence']} "
            f"arguments={len(opinion['arguments'])}"
        )

        return {
            "defender_opinion": opinion,
            "completed_nodes":  state.get("completed_nodes", []) + ["defender"],
            "error":            None,
        }

    except json.JSONDecodeError as e:
        logger.error(f"[defender] JSON parse error: {e}")
        return {
            "defender_opinion": None,
            "error":            "خطا در تجزیه پاسخ وکیل مدافع",
            "completed_nodes":  state.get("completed_nodes", []) + ["defender"],
        }

    except Exception as e:
        logger.error(f"[defender] unexpected error: {e}", exc_info=True)
        return {
            "defender_opinion": None,
            "error":            f"خطای غیرمنتظره در تحلیل دفاعی: {str(e)}",
            "completed_nodes":  state.get("completed_nodes", []) + ["defender"],
        }