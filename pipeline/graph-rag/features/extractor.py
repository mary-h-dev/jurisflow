"""
features/extractor.py — استخراج Feature از متن یک رأی با LLM (لایه‌ی سوم گراف)

نحوه‌ی کار:
    ۱. متن رأی همراه با closed vocabulary هر دسته (خروجی
       vocabulary_categorizer.py، از data/legal-vocabulary/categorized/)
       به LLM داده می‌شود.
    ۲. برای concept/action/role/object/principle، LLM موظف است **فقط**
       از بین همان واژه‌ها انتخاب کند؛ برای fact آزاد است.
    ۳. برای هر Feature یک evidence-quote هم خواسته می‌شود؛ موقعیتش در
       متن اصلی با str.find پیدا می‌شود (توضیح کامل در schemas.py).

چرا هر واژه‌ی خارج از closed vocabulary رد می‌شود (نه اصلاح/نگه‌داری)؟
    چون کل فلسفه‌ی closed vocabulary همینه: اگر یک‌بار اجازه بدیم LLM
    چیزی بیرون از لیست برگردونه و بی‌صدا قبولش کنیم، دقیقاً همون مشکلی
    برمی‌گرده که closed vocabulary قرار بود جلوش رو بگیره (گراف پر از
    node های نامنسجم). به‌جاش رد می‌شه و لاگ می‌شه — تا اگه زیاد تکرار
    شد، بفهمیم واژه‌نامه ناقصه و باید بهش اضافه کنیم.

چرا کل closed vocabulary (نه فقط بخش مرتبط) در پرامپت می‌رود؟
    چون فعلاً اندازه‌اش کوچک است (چند صد واژه بعد از canonicalization).
    اگر بعداً این تعداد خیلی زیاد شد، باید یک مرحله‌ی retrieval
    (embedding-based، شبیه bm25_search.py/embedder.py موجود در پروژه)
    قبل از این اضافه شود تا فقط واژه‌های محتمل به پرامپت بروند — این را
    همین‌جا یادداشت می‌کنیم تا فراموش نشود، ولی برای پایلوت فعلی لازم
    نیست.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from dotenv import load_dotenv

from features.configs import FACT_CATEGORY, LLM_MODEL, LLM_TEMPERATURE, VOCAB_CATEGORIES, get_llm_client
from features.schemas import Evidence, ExtractedFeature, FeatureExtractionResult

load_dotenv()

_HERE = Path(__file__).resolve().parent.parent  # graph-rag/
VOCAB_DIR = _HERE / "data" / "legal-vocabulary" / "categorized"

_client = get_llm_client()

# سقف امنیتی طول متن رأی که به LLM فرستاده می‌شود — مثل الگوی
# embedder.py/analyzer.py، برای جلوگیری از overflow روی context window
# و کنترل هزینه؛ اکثر رأی‌ها خیلی کوتاه‌ترند.
MAX_TEXT_CHARS = 12000


def load_closed_vocab(category_key: str) -> list[str]:
    path = VOCAB_DIR / f"{category_key}s.json"
    if not path.exists():
        raise FileNotFoundError(
            f"واژه‌نامه‌ی دسته‌بندی‌شده‌ی «{category_key}» پیدا نشد ({path}). "
            f"اول uv run -m features.vocabulary_categorizer را اجرا کن."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _build_prompt(ruling_text: str, vocab: dict[str, list[str]]) -> str:
    vocab_lines = []
    for key, words in vocab.items():
        words_str = "، ".join(words) if words else "(خالی)"
        vocab_lines.append(f'- {VOCAB_CATEGORIES[key].label_fa} ("{key}"): {words_str}')
    vocab_block = "\n".join(vocab_lines)

    return f"""
تو یک دستیار حقوقی متخصص در قوانین ایران هستی. از متن رأی زیر، Featureهای
حقوقی را طبق دسته‌های مشخص‌شده استخراج کن.

برای دسته‌های concept / action / role / object / principle: **فقط** از
بین واژه‌های همان دسته در لیست زیر انتخاب کن (هیچ واژه‌ی جدیدی نساز؛
اگر هیچ‌کدام مرتبط نبود، آن دسته را خالی بگذار):

{vocab_block}

برای دسته‌ی "{FACT_CATEGORY.key}" ({FACT_CATEGORY.label_fa}):
{FACT_CATEGORY.description}
اینجا آزاد هستی — واقعیات کلیدیِ رأی را با یک جمله‌ی کوتاه بنویس.

برای *هر* Feature استخراج‌شده (در هر دسته‌ای)، یک evidence لازم است:
دقیقاً همان بخشی از متن رأی که این Feature از آن برداشت شده (عیناً،
بدون تغییر کلمه، تا بشه موقعیتش را در متن پیدا کرد)، به‌علاوه‌ی یک عدد
اطمینان بین 0 و 1.

⚠️ نکته‌ی حیاتی درباره‌ی evidence: quote باید **مستقیماً و مشخصاً**
همان چیزی را ثابت کند که در «value» گفته شده — نه یک جمله‌ی کلی از
همان بخش از متن که فقط نزدیک آن قسمت آمده. مثال غلط: اگر value برابر
«آیین دادرسی» است، evidence نباید جمله‌ای درباره‌ی یک قرارداد فروش
باشد (حتی اگر آن جمله در همان پاراگراف آمده باشد) — این‌جور evidence
نامرتبط رد خواهد شد. اگر برای یک Feature نمی‌توانی evidence دقیق و
مرتبط پیدا کنی، آن Feature را اصلاً استخراج نکن.

متن رأی:
\"\"\"
{ruling_text[:MAX_TEXT_CHARS]}
\"\"\"

فقط JSON برگردون — بدون هیچ توضیح اضافه، دقیقاً به این فرم (هر دسته
می‌تواند لیست خالی باشد):
{{
  "concept": [{{"value": "...", "evidence_quote": "...", "confidence": 0.9}}],
  "action": [],
  "role": [],
  "object": [],
  "principle": [],
  "fact": [{{"value": "...", "evidence_quote": "...", "confidence": 0.85}}]
}}
"""


def _fuzzy_find(quote: str, full_text: str) -> tuple[int | None, int | None]:
    """
    fallback برای وقتی str.find شکست می‌خوره. چرا لازم است؟
    چون توی نمونه‌های واقعی دیدیم حدود نصفِ evidence ها فقط به‌خاطر
    تفاوت نیم‌فاصله/فاصله (مثلاً «مبایعه‌نامه» با یا بدون ZWNJ بین
    «مبایعه» و «نامه») در متن اصلی پیدا نمی‌شدند، نه چون quote واقعاً
    غلط بود. اینجا quote را از روی فاصله/نیم‌فاصله می‌شکنیم و اجازه
    می‌دیم بین تکه‌ها هر مقدار فاصله/نیم‌فاصله (حتی صفر) باشد؛ ولی
    محتوای واقعی کلمه‌ها باید عیناً یکی باشد — این با حدس زدن یا
    fuzzy-matching معنایی فرق دارد؛ فقط نویسه‌های جداکننده را نادیده
    می‌گیرد.
    """
    tokens = [t for t in re.split(r"[\s\u200c]+", quote.strip()) if t]
    if not tokens:
        return None, None
    pattern = r"[\s\u200c]*".join(re.escape(t) for t in tokens)
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

    # حتی با fuzzy match هم پیدا نشد — مدل کمی متن را واقعاً تغییر داده
    # (نه فقط فاصله/نیم‌فاصله). متن مدرک را نگه می‌داریم ولی بی‌صدا
    # موقعیت را حدس نمی‌زنیم (نگاه کن به schemas.py).
    return Evidence(quote=quote, start_char=None, end_char=None, confidence=confidence)


def extract_ruling(ruling_id: str, ruling_text: str, retries: int = 2) -> FeatureExtractionResult:
    vocab = {key: load_closed_vocab(key) for key in VOCAB_CATEGORIES}
    prompt = _build_prompt(ruling_text, vocab)

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
            return _to_result(ruling_id, ruling_text, data, vocab)
        except Exception as e:  # noqa: BLE001 — خطای JSON/schema هم باید اینجا گرفته بشه
            last_error = e
            if attempt < retries:
                print(f"  ⏳ خطا در استخراج Feature برای {ruling_id} "
                      f"(تلاش {attempt + 1}/{retries}): {e}")
                time.sleep(3)

    raise RuntimeError(f"❌ استخراج Feature برای {ruling_id} شکست خورد: {last_error}")


def _to_result(
    ruling_id: str, ruling_text: str, data: dict, vocab: dict[str, list[str]]
) -> FeatureExtractionResult:
    result = FeatureExtractionResult(ruling_id=ruling_id)
    target_lists = {
        "concept": result.concepts,
        "action": result.actions,
        "role": result.roles,
        "object": result.objects,
        "principle": result.principles,
        "fact": result.facts,
    }

    for category_key, target_list in target_lists.items():
        allowed = set(vocab[category_key]) if category_key in vocab else None  # None یعنی fact (آزاد)
        for item in data.get(category_key, []):
            value = str(item.get("value", "")).strip()
            if not value:
                continue
            if allowed is not None and value not in allowed:
                print(f"  ⚠️ مقدار خارج از closed vocabulary رد شد ({category_key}): «{value}»")
                continue
            evidence = _locate_evidence(
                item.get("evidence_quote", ""), ruling_text, float(item.get("confidence", 0.0))
            )
            _warn_if_evidence_unrelated(ruling_id, category_key, value, evidence.quote)
            target_list.append(ExtractedFeature(category=category_key, value=value, evidence=evidence))

    return result


def _warn_if_evidence_unrelated(ruling_id: str, category_key: str, value: str, quote: str):
    """
    چک نرم (نه رد کردن): اگر evidence هیچ کلمه‌ی مشترکی با value نداشته
    باشد، احتمال زیاد یعنی مدل evidence نامرتبطی چسبانده (دقیقاً همون
    الگویی که در بررسی دستیِ پایلوت دیدیم: «آیین دادرسی» با evidence
    درباره‌ی یک قرارداد فروش). این رد نمی‌کند چون گاهی evidence واقعاً
    درست است ولی هم‌پوشانی کلمه‌ای ندارد (پارافریز مجاز) — فقط لاگ
    می‌کند تا در بازبینی دستیِ پایلوت راحت‌تر پیدا بشه.
    """
    value_tokens = [t for t in re.split(r"[\s\u200c]+", value) if len(t) > 1]
    if value_tokens and quote and not any(t in quote for t in value_tokens):
        print(f"  🔎 [{ruling_id}] احتمال عدم تطابق evidence با value "
              f"({category_key}=«{value}»): «{quote[:60]}...»")