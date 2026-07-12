from __future__ import annotations

import json
import logging

from django.conf import settings
from groq import Groq

from apps.agents.state import CaseState, TeammateOpinion

logger = logging.getLogger(__name__)

_groq = Groq(api_key=settings.GROQ_API_KEY)

PROMPT = """
تو یک دادستان یا وکیل طرف مقابل متخصص در قوانین ایران هستی.
وظیفه تو یافتن نقاط ضعف پرونده و قوی‌ترین استدلال‌های علیه موکل است.
باید سخت‌گیرانه و واقع‌بینانه نقاط آسیب‌پذیر را شناسایی کنی.

اطلاعات پرونده:
- موضوع: {case_subject}
- نوع: {case_type}
- طرفین: {parties}
- وقایع کلیدی: {key_facts}
- قوانین مرتبط: {related_laws}

وظیفه تو:
۱. ضعیف‌ترین نقاط پرونده از دیدگاه طرف مقابل را شناسایی کن
۲. به مواد قانونی که علیه موکل است استناد کن
۳. ریسک‌های اصلی را مشخص کن
۴. میزان اطمینان خود را اعلام کن

فقط JSON برگردون — بدون هیچ توضیح اضافه:
{{
    "position": "موضع کلی مخالف در یک جمله",
    "arguments": [
        "استدلال مخالف اول",
        "استدلال مخالف دوم",
        "استدلال مخالف سوم"
    ],
    "cited_articles": [
        "ماده X — قانون Y"
    ],
    "confidence": "بالا | متوسط | پایین"
}}
"""


def prosecute(state: CaseState) -> dict:
    """
    Teammate 2 — دادستان/مخالف

    وظایف:
    - نقاط ضعف پرونده رو شناسایی می‌کنه
    - به مواد قانونی علیه موکل استناد می‌کنه
    - ریسک‌های واقعی رو مشخص می‌کنه
    - با defender و judge در debate شرکت می‌کنه
    """
    logger.info(f"[prosecutor] session={state['session_id']} started")

    if state.get("error"):
        logger.warning("[prosecutor] skipping due to previous error")
        return {
            "completed_nodes": state.get("completed_nodes", []) + ["prosecutor"]
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
            for law in (state.get("related_laws") or [])[:8]
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
            temperature=0.2,
        )

        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)

        opinion = TeammateOpinion(
            role="prosecutor",
            position=data.get("position", ""),
            arguments=data.get("arguments", []),
            cited_articles=data.get("cited_articles", []),
            confidence=data.get("confidence", "متوسط"),
        )

        logger.info(
            f"[prosecutor] session={state['session_id']} "
            f"confidence={opinion['confidence']} "
            f"arguments={len(opinion['arguments'])}"
        )

        return {
            "prosecutor_opinion": opinion,
            "completed_nodes":    state.get("completed_nodes", []) + ["prosecutor"],
            "error":              None,
        }

    except json.JSONDecodeError as e:
        logger.error(f"[prosecutor] JSON parse error: {e}")
        return {
            "prosecutor_opinion": None,
            "error":              "خطا در تجزیه پاسخ دادستان",
            "completed_nodes":    state.get("completed_nodes", []) + ["prosecutor"],
        }

    except Exception as e:
        logger.error(f"[prosecutor] unexpected error: {e}", exc_info=True)
        return {
            "prosecutor_opinion": None,
            "error":              f"خطای غیرمنتظره در تحلیل مخالف: {str(e)}",
            "completed_nodes":    state.get("completed_nodes", []) + ["prosecutor"],
        }