"""
One-time offline script: for every article_ref in the annotation dataset,
extract a fixed necessary-conditions checklist and save it to
apps/auditor/data/checklist_db.json. Resumable -- re-running skips refs
already in the file, so an interrupted run only costs the remaining articles.

Run:
    DJANGO_SETTINGS_MODULE=config.settings.development python -m apps.auditor.build_checklist_db
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from core.llm_client import get_client

from .apps_auditor_config import AUDITOR_MODEL
from .checklist_builder import _fetch_article_texts, _parse_article_ref

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_ANNOTATION_PATH = Path("apps/search/calibration/data/annotations/case_grounded.json")
_OUTPUT_PATH = Path("apps/auditor/data/checklist_db.json")
_REQUEST_DELAY_SECONDS = 0.5

_EXTRACTION_PROMPT_TEMPLATE = """
متن ماده:
{article_text}

شرط‌های قانونی لازم (necessary) برای اعمال این ماده را استخراج کن.

قواعد:
- هر شرط باید مستقیماً از متن همین ماده بیاید -- نه از واقعیت‌های یک پرونده‌ی خاص، نه از یک ماده‌ی رقیب یا جایگزین.
- هر شرط یک عنصر حقوقی مشخص باشد (نه خلاصه‌ی کلی ماده).
- هر شرط را طوری بنویس که satisfied=true یعنی "این عامل به‌نفع اعمال ماده است" -- هرگز برعکس (اگر متن ماده استثنا یا مانعی دارد مثل "مگر اینکه X"، آن را وارونه کن: "استثنای X برقرار نیست").
- حداکثر ۳ شرط. بیشتر مواد فقط ۱ تا ۲ عنصر اصلی دارند.

فقط یک JSON object برگردان، بدون متن اضافه:
{{"conditions": ["شرط ۱", "شرط ۲"]}}
"""


def _collect_unique_article_refs() -> list[str]:
    """
    Every article_ref that could plausibly reach the Auditor: gold
    articles, excluded_articles, and anything already keyed in
    gold_checklist_by_article, across the whole annotation file.
    Deduplicated, order-stable.
    """
    raw = json.loads(_ANNOTATION_PATH.read_text(encoding="utf-8"))
    refs: list[str] = []
    seen = set()

    def _add(ref: str | None):
        if ref and ref not in seen:
            seen.add(ref)
            refs.append(ref)

    for sample in raw:
        for ref in sample.get("gold_articles", []):
            _add(ref)
        for ex in sample.get("excluded_articles", []):
            law = ex.get("law_name")
            num = ex.get("article_number")
            if law and num:
                _add(f"{law} - ماده {num}")
        for ref in sample.get("gold_checklist_by_article", {}):
            _add(ref)

    return refs


def _load_existing_db() -> dict[str, list[str]]:
    if not _OUTPUT_PATH.exists():
        return {}
    try:
        return json.loads(_OUTPUT_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"checklist_db.json unreadable ({e}), starting fresh")
        return {}


def _save_db(db: dict[str, list[str]]) -> None:
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(
        json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _extract_conditions(article_text: str) -> list[str] | None:
    prompt = _EXTRACTION_PROMPT_TEMPLATE.format(article_text=article_text)
    try:
        raw_text = get_client().complete(
            model=AUDITOR_MODEL.model,
            temperature=AUDITOR_MODEL.temperature,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
        )
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None

    raw_text = raw_text.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(raw_text)
        conditions = parsed.get("conditions")
        if not isinstance(conditions, list) or not conditions:
            raise ValueError(f"bad conditions: {conditions!r}")
        return [str(c).strip() for c in conditions if str(c).strip()]
    except Exception as e:
        logger.warning(f"Parse failed: {e} -- raw: {raw_text[:300]!r}")
        return None


def run() -> None:
    all_refs = _collect_unique_article_refs()
    logger.info(f"{len(all_refs)} unique article refs found in annotation data")

    db = _load_existing_db()
    logger.info(f"{len(db)} already in checklist_db.json, skipping those")

    pending = [ref for ref in all_refs if ref not in db]
    logger.info(f"{len(pending)} remaining to process")

    if not pending:
        logger.info("nothing to do")
        return

    article_texts = _fetch_article_texts(pending)

    for i, ref in enumerate(pending):
        text = article_texts.get(ref, "")
        if not text:
            logger.warning(f"[{i+1}/{len(pending)}] {ref}: no article text found in DB, skipping")
            continue

        if i > 0 and _REQUEST_DELAY_SECONDS:
            time.sleep(_REQUEST_DELAY_SECONDS)

        conditions = _extract_conditions(text)
        if conditions is None:
            logger.warning(f"[{i+1}/{len(pending)}] {ref}: extraction failed, skipping")
            continue

        db[ref] = conditions
        _save_db(db)  # save after every article -- resumable, no batch loss on crash
        logger.info(f"[{i+1}/{len(pending)}] {ref}: {len(conditions)} conditions saved")

    logger.info(f"done -- {len(db)} total articles in checklist_db.json")


if __name__ == "__main__":
    run()