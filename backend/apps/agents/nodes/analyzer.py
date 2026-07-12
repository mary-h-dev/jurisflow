from __future__ import annotations

import json
import logging

from django.conf import settings
from groq import Groq

from apps.agents.state import CaseState, Party

logger = logging.getLogger(__name__)

_groq = Groq(api_key=settings.GROQ_API_KEY)

PROMPT = """
تو یک دستیار حقوقی متخصص در قوانین ایران هستی.
متن زیر از یک پرونده حقوقی استخراج شده است.

وظیفه تو:
۱. موضوع اصلی پرونده را در یک جمله کوتاه بنویس
۲. نوع پرونده را مشخص کن
۳. طرف‌های دعوا را شناسایی کن
۴. حداکثر ۷ واقعه کلیدی را استخراج کن
۵. کلیدواژه‌های حقوقی مرتبط را بنویس

متن پرونده:
{raw_text}

فقط JSON برگردون — بدون هیچ توضیح اضافه:
{{
    "case_subject": "موضوع اصلی پرونده در یک جمله",
    "case_type": "یکی از: حقوقی | کیفری | خانواده | تجاری | ملکی",
    "parties": [
        {{
            "role": "خواهان | خوانده | متهم | شاکی | ثالث",
            "name": "نام طرف",
            "details": "اطلاعات تکمیلی مختصر"
        }}
    ],
    "key_facts": [
        "واقعه کلیدی اول",
        "واقعه کلیدی دوم"
    ],
    "legal_keywords": [
        "کلیدواژه حقوقی اول",
        "کلیدواژه حقوقی دوم"
    ]
}}
"""


def analyze_document(state: CaseState) -> dict:
    """
    Agent 1 — Document Analyzer

    وظایف:
    - استخراج موضوع، نوع، طرفین، وقایع کلیدی
    - تولید کلیدواژه برای جستجوی قوانین در مرحله بعد
    - بعد از این مرحله raw_text رو پاک می‌کنه تا overhead توکن کم بشه
    """
    logger.info(f"[analyzer] session={state['session_id']} started")

    # برای کاهش توکن، فقط ۳۰۰۰ کاراکتر اول رو میفرستیم
    raw_text = state["raw_text"][:3000]

    try:
        response = _groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role":    "user",
                "content": PROMPT.format(raw_text=raw_text),
            }],
            temperature=0.1,
        )

        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)

        parties: list[Party] = [
            Party(
                role=p.get("role", ""),
                name=p.get("name", ""),
                details=p.get("details", ""),
            )
            for p in data.get("parties", [])
        ]

        completed = state.get("completed_nodes", []) + ["analyzer"]

        logger.info(
            f"[analyzer] session={state['session_id']} "
            f"type={data.get('case_type')} "
            f"parties={len(parties)}"
        )

        return {
            # خروجی analyzer
            "case_subject":   data.get("case_subject", ""),
            "case_type":      data.get("case_type", ""),
            "parties":        parties,
            "key_facts":      data.get("key_facts", [])[:7],
            "legal_keywords": data.get("legal_keywords", []),

            # پاک کردن raw_text برای کاهش overhead توکن
            "raw_text": "",

            "completed_nodes": completed,
            "error":           None,
        }

    except json.JSONDecodeError as e:
        logger.error(f"[analyzer] JSON parse error: {e}")
        return {
            "error":           "خطا در تجزیه پاسخ مدل",
            "completed_nodes": state.get("completed_nodes", []) + ["analyzer"],
        }

    except Exception as e:
        logger.error(f"[analyzer] unexpected error: {e}", exc_info=True)
        return {
            "error":           f"خطای غیرمنتظره در تحلیل سند: {str(e)}",
            "completed_nodes": state.get("completed_nodes", []) + ["analyzer"],
        }