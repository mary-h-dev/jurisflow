from __future__ import annotations

import json
import logging

from django.conf import settings
from groq import Groq

from apps.agents.state import CaseState, TeammateOpinion

logger = logging.getLogger(__name__)

_groq = Groq(api_key=settings.GROQ_API_KEY)

PROMPT = """
تو یک قاضی بی‌طرف و باتجربه در قوانین ایران هستی.
وظیفه تو ارزیابی واقع‌بینانه و بی‌طرفانه پرونده است.
نه جانب موکل را می‌گیری نه جانب طرف مقابل را.
فقط بر اساس قانون و واقعیت قضاوت می‌کنی.

اطلاعات پرونده:
- موضوع: {case_subject}
- نوع: {case_type}
- طرفین: {parties}
- وقایع کلیدی: {key_facts}
- قوانین مرتبط: {related_laws}
- استدلال وکیل مدافع: {defender_position}
- استدلال دادستان: {prosecutor_position}

وظیفه تو:
۱. هر دو دیدگاه را بررسی کن
۲. بر اساس قانون و واقعیت قضاوت کن
۳. احتمال موفقیت پرونده را ارزیابی کن
۴. به مواد قانونی کلیدی استناد کن

فقط JSON برگردون — بدون هیچ توضیح اضافه:
{{
    "position": "ارزیابی بی‌طرفانه در یک جمله",
    "arguments": [
        "نکته کلیدی اول از دیدگاه قاضی",
        "نکته کلیدی دوم از دیدگاه قاضی",
        "نکته کلیدی سوم از دیدگاه قاضی"
    ],
    "cited_articles": [
        "ماده X — قانون Y"
    ],
    "confidence": "بالا | متوسط | پایین"
}}
"""


def judge(state: CaseState) -> dict:
    """
    Teammate 3 — قاضی بی‌طرف

    وظایف:
    - هر دو دیدگاه defender و prosecutor رو بررسی می‌کنه
    - بی‌طرفانه بر اساس قانون قضاوت می‌کنه
    - احتمال موفقیت پرونده رو ارزیابی می‌کنه

    نکته: judge بعد از defender و prosecutor اجرا میشه
    تا بتونه نظر هر دو رو ببینه.
    در parallel flow، state هر دو opinion رو داره.
    """
    logger.info(f"[judge] session={state['session_id']} started")

    if state.get("error"):
        logger.warning("[judge] skipping due to previous error")
        return {
            "completed_nodes": state.get("completed_nodes", []) + ["judge"]
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

        # نظر defender و prosecutor برای judge
        defender_position = (
            state["defender_opinion"]["position"]
            if state.get("defender_opinion")
            else "نظر وکیل مدافع در دسترس نیست"
        )
        prosecutor_position = (
            state["prosecutor_opinion"]["position"]
            if state.get("prosecutor_opinion")
            else "نظر دادستان در دسترس نیست"
        )

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
                    defender_position=defender_position,
                    prosecutor_position=prosecutor_position,
                ),
            }],
            temperature=0.1,  # پایین‌ترین temperature — قاضی باید منطقی باشه
        )

        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)

        opinion = TeammateOpinion(
            role="judge",
            position=data.get("position", ""),
            arguments=data.get("arguments", []),
            cited_articles=data.get("cited_articles", []),
            confidence=data.get("confidence", "متوسط"),
        )

        logger.info(
            f"[judge] session={state['session_id']} "
            f"confidence={opinion['confidence']} "
            f"arguments={len(opinion['arguments'])}"
        )

        return {
            "judge_opinion":   opinion,
            "completed_nodes": state.get("completed_nodes", []) + ["judge"],
            "error":           None,
        }

    except json.JSONDecodeError as e:
        logger.error(f"[judge] JSON parse error: {e}")
        return {
            "judge_opinion":   None,
            "error":           "خطا در تجزیه پاسخ قاضی",
            "completed_nodes": state.get("completed_nodes", []) + ["judge"],
        }

    except Exception as e:
        logger.error(f"[judge] unexpected error: {e}", exc_info=True)
        return {
            "judge_opinion":   None,
            "error":           f"خطای غیرمنتظره در ارزیابی قاضی: {str(e)}",
            "completed_nodes": state.get("completed_nodes", []) + ["judge"],
        }