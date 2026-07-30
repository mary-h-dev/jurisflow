"""
features/extractor.py — استخراج Feature از متن یک رأی با LLM (لایه‌ی سوم گراف)

نحوه‌ی کار (نسخه‌ی جدید — بدون فرستادن واژه‌نامه در prompt):
    ۱. LLM آزادانه (بدون دیدن closed vocabulary) از متن رأی، concept/
       action/role/object/fact استخراج می‌کند — دقیقاً با همان کلماتی
       که در متن آمده، نه پارافریز.
    ۲. هر value استخراج‌شده (به‌جز fact که آزاد است) با
       features/vocab_resolver.py به نزدیک‌ترین واژه‌ی closed
       vocabulary نگاشت می‌شود (normalize → alias → fuzzy → embedding).
       اگر resolver هیچ match نزدیکی پیدا نکرد، آن Feature رد می‌شود.
    ۳. برای هر Feature یک evidence-quote هم خواسته می‌شود؛ موقعیتش در
       متن اصلی با str.find (یا fuzzy fallback) پیدا می‌شود.

چرا این تغییر (نه فرستادن واژه‌نامه در prompt)؟
    چون واژه‌نامه‌ی کامل (چند صد واژه) prompt را از سقف token/TPM
    بیشتر providerهای رایگان رد می‌کرد. راه‌حل اول (retrieval سبک با
    embedding کل پرونده) امتحان و رد شد — چون کلمات کوتاه/پرتکرار مثل
    «خواهان» امتیاز شباهت پایینی به یک پاراگراف طولانی می‌گرفتند و از
    prompt حذف می‌شدند. راه‌حل فعلی: بگذار LLM آزاد استخراج کند (prompt
    کوچک می‌ماند)، و match با واژه‌نامه را *بعد از* استخراج و روی
    عبارات کوتاه (نه کل پرونده) انجام بده — که هم دقیق‌تر است هم prompt
    کوچک می‌ماند.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from dotenv import load_dotenv

from features.configs import FACT_CATEGORY, LLM_MODEL, LLM_TEMPERATURE, VOCAB_CATEGORIES, get_llm_client
from features.schemas import Evidence, ExtractedFeature, FeatureExtractionResult
from features.vocab_resolver import VocabResolver

load_dotenv()

_HERE = Path(__file__).resolve().parent.parent  # graph-rag/

_client = get_llm_client()

# سقف امنیتی طول متن رأی که به LLM فرستاده می‌شود.
MAX_TEXT_CHARS = 12000
_HEAD_RATIO = 0.4  # ۴۰٪ به ابتدای متن، ۶۰٪ به انتها (چون انتها مهم‌تره)

_CATEGORY_DESCRIPTIONS = "\n".join(
    f'- {c.key}: {c.label_fa} — {c.description}' for c in VOCAB_CATEGORIES.values()
)


def _truncate_keep_head_and_tail(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head_len = int(max_chars * _HEAD_RATIO)
    tail_len = max_chars - head_len
    return (
        text[:head_len]
        + "\n\n[...بخشی از میانه‌ی متن به‌خاطر طولانی بودن حذف شد...]\n\n"
        + text[-tail_len:]
    )


def _build_prompt(ruling_text: str) -> str:
    return f"""
تو یک دستیار حقوقی متخصص در قوانین ایران هستی. از متن رأی زیر، Featureهای
حقوقی را طبق دسته‌های زیر استخراج کن:

{_CATEGORY_DESCRIPTIONS}

⚠️ برای concept/action/role/object: دقیقاً همان اصطلاحی که در متن آمده
استفاده کن — پارافریز یا خلاصه‌سازی نکن، همان شکل دقیق کلمه/عبارت را
بنویس (چون بعداً با یک واژه‌نامه‌ی رسمی تطبیق داده می‌شود).

برای دسته‌ی "{FACT_CATEGORY.key}" ({FACT_CATEGORY.label_fa}):
{FACT_CATEGORY.description}
اینجا آزاد هستی — واقعیات کلیدیِ رأی را با یک جمله‌ی کوتاه بنویس.
⚠️ مثال‌های بالا (مثل «پرداخت انجام نشده» یا «سند جعلی ارائه شده») فقط
برای نشان‌دادن *سبک نوشتن* fact هستند، نه فهرستی که باید همیشه پر شود.
اگر یکی از این مثال‌ها در *این* رأی مصداق ندارد، اصلاً ذکرش نکن — هرگز
یک Feature را فقط به این دلیل که در توضیحات بالا مثال زده شده، با
evidence_quote خالی و confidence صفر برنگردان؛ چنین مواردی به‌طور کامل
حذف خواهند شد، پس اصلاً وقتت را صرفشان نکن.

برای *هر* Feature استخراج‌شده (در هر دسته‌ای)، یک evidence لازم است:
دقیقاً همان بخشی از متن رأی که این Feature از آن برداشت شده (عیناً،
بدون تغییر کلمه، تا بشه موقعیتش را در متن پیدا کرد)، به‌علاوه‌ی یک عدد
اطمینان بین 0 و 1.

⚠️ نکته‌ی حیاتی درباره‌ی evidence: quote باید **مستقیماً و مشخصاً**
همان چیزی را ثابت کند که در «value» گفته شده — نه یک جمله‌ی کلی از
همان بخش از متن که فقط نزدیک آن قسمت آمده. اگر برای یک Feature
نمی‌توانی evidence دقیق و مرتبط پیدا کنی، آن Feature را اصلاً استخراج
نکن.

⚠️ evidence_quote هرگز نباید شامل «...» یا خلاصه‌سازی/ترکیبِ چند تکه‌ی
جدا از متن باشد — باید عیناً یک بخشِ *پیوسته* و *کامل* از متن اصلی
باشد، کلمه‌به‌کلمه، بدون هیچ حذفیات. اگر جمله‌ی کامل خیلی طولانی است،
فقط کوتاه‌ترین بخشِ پیوسته‌ای از متن را انتخاب کن که به‌تنهایی ادعای
«value» را ثابت می‌کند — نه کل جمله با «...» بریده‌شده.

⚠️ راهنمای دقیق برای تعیین confidence (لطفاً دقیقاً رعایت کن، نه فقط
تقریبی؛ یک عدد ثابت برای همه‌ی موارد، مثلاً همه 0.95، قابل‌قبول نیست):
- 0.95-1.0: کلمه یا عبارت «value» عیناً و بدون هیچ واسطه‌ای در evidence quote آمده است.
- 0.75-0.9: value مستقیماً در quote نیامده، ولی مفهومش به‌وضوح و بدون ابهام از آن استنباط می‌شود.
- 0.5-0.74: نیاز به استنباط چندمرحله‌ای یا context بیرون از quote دارد.
- زیر 0.5: ارتباط ضعیف است؛ در این حالت بهتر است اصلاً این Feature را استخراج نکنی.

متن رأی:
\"\"\"
{_truncate_keep_head_and_tail(ruling_text, MAX_TEXT_CHARS)}
\"\"\"

فقط JSON برگردون — بدون هیچ توضیح اضافه، دقیقاً به این فرم (هر دسته
می‌تواند لیست خالی باشد):
{{
  "concept": [{{"value": "...", "evidence_quote": "...", "confidence": 0.9}}],
  "action": [],
  "role": [],
  "object": [],
  "fact": [{{"value": "...", "evidence_quote": "...", "confidence": 0.85}}]
}}
"""


# کاراکترهای کنترلی جهت‌دهیِ متن (bidi control characters) — کاملاً
# نامرئی‌ان، پس هم مثل نیم‌فاصله باید در fuzzy-match نادیده گرفته بشن.
_BIDI_CONTROL_CHARS = "\u200e\u200f\u202a\u202b\u202c\u202d\u202e"
_SPLIT_SEPARATOR_RE = re.compile(r"[\s\u200c" + _BIDI_CONTROL_CHARS + r"]+")


def _fuzzy_find(quote: str, full_text: str) -> tuple[int | None, int | None]:
    tokens = [t for t in _SPLIT_SEPARATOR_RE.split(quote.strip()) if t]
    if not tokens:
        return None, None
    separator_pattern = r"[\s\u200c" + _BIDI_CONTROL_CHARS + r"]*"
    pattern = separator_pattern.join(re.escape(t) for t in tokens)
    m = re.search(pattern, full_text)
    if not m:
        return None, None
    return m.start(), m.end()


def _locate_evidence(quote: str, full_text: str, confidence: float) -> Evidence:
    quote = (quote or "").strip()
    if not quote:
        return Evidence(quote="", start_char=None, end_char=None, confidence=confidence)

    idx = full_text.find(quote)
    if idx != -1:
        return Evidence(quote=quote, start_char=idx, end_char=idx + len(quote), confidence=confidence)

    start, end = _fuzzy_find(quote, full_text)
    if start is not None:
        return Evidence(quote=quote, start_char=start, end_char=end, confidence=confidence)

    if "..." in quote or "…" in quote:
        print(f"  ⚠️ evidence حاوی «...» و پیدا نشد — احتمالاً پارافریزِ مدل: «{quote[:50]}...»")
        return Evidence(quote=quote, start_char=None, end_char=None, confidence=confidence)

    return Evidence(quote=quote, start_char=None, end_char=None, confidence=confidence)


def extract_ruling(
    ruling_id: str, ruling_text: str, resolver: VocabResolver, retries: int = 2
) -> FeatureExtractionResult:
    """
    resolver: یک نمونه‌ی VocabResolver — باید *یک‌بار* در main_features.py
    ساخته و برای همه‌ی پرونده‌ها استفاده مجدد شود (نه هر پرونده جدا).
    """
    prompt = _build_prompt(ruling_text)

    last_error = None
    for attempt in range(retries + 1):
        try:
            response = _client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=LLM_TEMPERATURE,
            )
            text = response.choices[0].message.content.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            data = json.loads(text)
            return _to_result(ruling_id, ruling_text, data, resolver)
        except Exception as e:  # noqa: BLE001
            last_error = e
            if attempt < retries:
                print(f"  ⏳ خطا در استخراج Feature برای {ruling_id} "
                      f"(تلاش {attempt + 1}/{retries}): {e}")
                time.sleep(3)

    raise RuntimeError(f"❌ استخراج Feature برای {ruling_id} شکست خورد: {last_error}")


def _to_result(
    ruling_id: str, ruling_text: str, data: dict, resolver: VocabResolver
) -> FeatureExtractionResult:
    result = FeatureExtractionResult(ruling_id=ruling_id)
    target_lists = {
        "concept": result.concepts,
        "action": result.actions,
        "role": result.roles,
        "object": result.objects,
        "fact": result.facts,
    }



    for category_key, target_list in target_lists.items():
            is_free_category = category_key == "fact"  # fact از واژه‌نامه resolve نمی‌شه
            seen_values = set()  # فقط همینجا مقداردهی اولیه — قبل از حلقه‌ی داخلی

            for item in data.get(category_key, []):
                raw_value = str(item.get("value", "")).strip()
                if not raw_value:
                    continue

                if is_free_category:
                    value = raw_value
                else:
                    value = resolver.resolve(raw_value, category_key)
                    if value is None:
                        print(f"  ⚠️ «{raw_value}» به هیچ واژه‌ی closed vocabulary نزدیک نبود "
                            f"— رد شد ({category_key})")
                        continue

                if value in seen_values:  # ← چک تکرار اینجا، بعد از resolve شدن value
                    continue
                seen_values.add(value)

                raw_quote = str(item.get("evidence_quote", "")).strip()
                if not raw_quote:
                    print(f"  🚫 [{ruling_id}] بدون evidence رد شد ({category_key}): «{value}»")
                    continue

                evidence = _locate_evidence(raw_quote, ruling_text, float(item.get("confidence", 0.0)))
                _warn_if_evidence_unrelated(ruling_id, category_key, value, evidence.quote)
                target_list.append(ExtractedFeature(category=category_key, value=value, evidence=evidence))

    return result


def _warn_if_evidence_unrelated(ruling_id: str, category_key: str, value: str, quote: str):
    value_tokens = [t for t in re.split(r"[\s\u200c]+", value) if len(t) > 1]
    if value_tokens and quote and not any(t in quote for t in value_tokens):
        print(f"  🔎 [{ruling_id}] احتمال عدم تطابق evidence با value "
              f"({category_key}=«{value}»): «{quote[:60]}...»")