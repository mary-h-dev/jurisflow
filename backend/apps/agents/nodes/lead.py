from __future__ import annotations

import json
import logging

from django.conf import settings
from groq import Groq

from apps.agents.state import CaseState, Risk

logger = logging.getLogger(__name__)

_groq = Groq(api_key=settings.GROQ_API_KEY)

PROMPT = """
تو Lead Agent یک تیم حقوقی هستی.
سه همتیمی تو (وکیل مدافع، دادستان، قاضی) پرونده را بررسی کرده‌اند.
وظیفه تو جمع‌بندی نهایی، تصمیم‌گیری و ارائه توصیه عملی است.

اطلاعات پرونده:
- موضوع: {case_subject}
- نوع: {case_type}
- طرفین: {parties}

نظر وکیل مدافع:
موضع: {defender_position}
استدلال‌ها: {defender_arguments}
مواد استنادی: {defender_articles}
اطمینان: {defender_confidence}

نظر دادستان/مخالف:
موضع: {prosecutor_position}
استدلال‌ها: {prosecutor_arguments}
مواد استنادی: {prosecutor_articles}
اطمینان: {prosecutor_confidence}

نظر قاضی بی‌طرف:
موضع: {judge_position}
استدلال‌ها: {judge_arguments}
مواد استنادی: {judge_articles}
اطمینان: {judge_confidence}

وظیفه تو:
۱. نقاط قوت پرونده را از debate استخراج کن
۲. نقاط ضعف پرونده را مشخص کن
۳. ریسک‌های اصلی را شناسایی کن
۴. احتمال موفقیت کلی را ارزیابی کن
۵. استراتژی مناسب را انتخاب کن
۶. توصیه نهایی عملی و کاربردی بده
۷. اقدامات پیشنهادی را به ترتیب اولویت بنویس

فقط JSON برگردون — بدون هیچ توضیح اضافه:
{{
    "strengths": [
        "نقطه قوت اول",
        "نقطه قوت دوم"
    ],
    "weaknesses": [
        "نقطه ضعف اول",
        "نقطه ضعف دوم"
    ],
    "risks": [
        {{
            "title": "عنوان ریسک",
            "description": "توضیح ریسک",
            "severity": "بالا | متوسط | پایین"
        }}
    ],
    "win_chance": "بالا | متوسط | پایین",
    "strategy": "تهاجمی | دفاعی | مصالحه",
    "recommendation": "توصیه نهایی کامل و کاربردی در چند جمله",
    "next_steps": [
        "اقدام اول و فوری",
        "اقدام دوم",
        "اقدام سوم"
    ]
}}
"""


def _format_arguments(arguments: list[str]) -> str:
    return "\n".join(f"  {i}. {arg}" for i, arg in enumerate(arguments, 1))


def lead_decision(state: CaseState) -> dict:
    """
    Lead Agent — تصمیم‌گیر نهایی

    وظایف:
    - نظرات سه teammate رو جمع‌بندی می‌کنه
    - نقاط قوت و ضعف رو استخراج می‌کنه
    - استراتژی نهایی رو انتخاب می‌کنه
    - توصیه عملی و next steps میده
    """
    logger.info(f"[lead] session={state['session_id']} started")

    if state.get("error") and not any([
        state.get("defender_opinion"),
        state.get("prosecutor_opinion"),
        state.get("judge_opinion"),
    ]):
        logger.warning("[lead] no teammate opinions available")
        return {
            "recommendation":  "به دلیل خطا در پردازش، امکان ارائه توصیه وجود ندارد.",
            "completed_nodes": state.get("completed_nodes", []) + ["lead"],
        }

    try:
        # ── آماده‌سازی نظرات teammates ───────────────────────────────────────
        defender   = state.get("defender_opinion")   or {}
        prosecutor = state.get("prosecutor_opinion") or {}
        judge_op   = state.get("judge_opinion")      or {}

        parties_text = "\n".join([
            f"- {p['role']}: {p['name']}"
            for p in (state.get("parties") or [])
        ])

        response = _groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role":    "user",
                "content": PROMPT.format(
                    case_subject=state.get("case_subject", ""),
                    case_type=state.get("case_type", ""),
                    parties=parties_text or "نامشخص",

                    # defender
                    defender_position=defender.get("position", "نامشخص"),
                    defender_arguments=_format_arguments(
                        defender.get("arguments", [])
                    ),
                    defender_articles=", ".join(
                        defender.get("cited_articles", [])
                    ),
                    defender_confidence=defender.get("confidence", "نامشخص"),

                    # prosecutor
                    prosecutor_position=prosecutor.get("position", "نامشخص"),
                    prosecutor_arguments=_format_arguments(
                        prosecutor.get("arguments", [])
                    ),
                    prosecutor_articles=", ".join(
                        prosecutor.get("cited_articles", [])
                    ),
                    prosecutor_confidence=prosecutor.get("confidence", "نامشخص"),

                    # judge
                    judge_position=judge_op.get("position", "نامشخص"),
                    judge_arguments=_format_arguments(
                        judge_op.get("arguments", [])
                    ),
                    judge_articles=", ".join(
                        judge_op.get("cited_articles", [])
                    ),
                    judge_confidence=judge_op.get("confidence", "نامشخص"),
                ),
            }],
            temperature=0.1,
        )

        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)

        risks: list[Risk] = [
            Risk(
                title=r.get("title", ""),
                description=r.get("description", ""),
                severity=r.get("severity", "متوسط"),
            )
            for r in data.get("risks", [])
        ]

        # جمع‌آوری همه cited_articles از teammates
        all_cited = list({
            *state.get("cited_articles", []),
            *defender.get("cited_articles", []),
            *prosecutor.get("cited_articles", []),
            *judge_op.get("cited_articles", []),
        })

        logger.info(
            f"[lead] session={state['session_id']} "
            f"win_chance={data.get('win_chance')} "
            f"strategy={data.get('strategy')} "
            f"risks={len(risks)}"
        )

        return {
            "strengths":       data.get("strengths", []),
            "weaknesses":      data.get("weaknesses", []),
            "risks":           risks,
            "win_chance":      data.get("win_chance", "متوسط"),
            "strategy":        data.get("strategy", "دفاعی"),
            "recommendation":  data.get("recommendation", ""),
            "next_steps":      data.get("next_steps", []),
            "cited_articles":  all_cited,
            "completed_nodes": state.get("completed_nodes", []) + ["lead"],
            "error":           None,
        }

    except json.JSONDecodeError as e:
        logger.error(f"[lead] JSON parse error: {e}")
        return {
            "recommendation":  "خطا در جمع‌بندی نهایی",
            "error":           "خطا در تجزیه پاسخ Lead Agent",
            "completed_nodes": state.get("completed_nodes", []) + ["lead"],
        }

    except Exception as e:
        logger.error(f"[lead] unexpected error: {e}", exc_info=True)
        return {
            "recommendation":  "خطای غیرمنتظره در جمع‌بندی نهایی",
            "error":           f"خطای Lead Agent: {str(e)}",
            "completed_nodes": state.get("completed_nodes", []) + ["lead"],
        }