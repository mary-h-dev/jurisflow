from __future__ import annotations

import json
import logging
import time

import requests
from django.conf import settings

from .checklist_builder import ArticleEvidenceBundle
from .schemas import ChecklistItemOut

logger = logging.getLogger(__name__)

_GEMINI_MODEL = "gemini-2.5-flash" 
_GROQ_MODEL = "llama-3.3-70b-versatile"
_MAX_ARTICLE_TEXT_CHARS = 1200
_MAX_EVIDENCE_TEXT_CHARS = 300




def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + " ..."


_PROMPT_TEMPLATE = """
تو یک حقوقدان دقیق و محتاط هستی که وظیفه‌ی audit یک ماده‌ی قانونی خاص را نسبت به شرح واقعیت یک پرونده بر عهده داری.

شرح واقعیت پرونده (query):
{query}

ماده‌ی مورد بررسی: {article_ref}
متن ماده:
{article_text}

شواهد پشتیبان (آرای مرتبط و واقعیت‌های استخراج‌شده):
{evidence_block}

قانون طلایی (خیلی مهم): چک‌لیست باید مختصِ **همین ماده** باشد، نه یک خلاصه‌ی کلی از واقعیت پرونده.
هر شرط باید مستقیماً از یک عنصر قانونی داخل متن همین ماده استخراج شده باشد (مثلاً یک قید، یک استثنا، یک رکن مادی یا معنوی که خودِ این ماده ذکر کرده).
اگر چک‌لیستی که می‌سازی می‌تواند بدون تغییر برای ماده‌ی دیگری هم استفاده شود (چون فقط واقعیت‌های عمومی پرونده مثل «درگیری رخ داده» یا «متهم شرکت داشته» را تکرار می‌کند)، اشتباه است — آن را دوباره بساز و عناصر خاص همین ماده را پیدا کن.

مثال درست (برای نشان دادن سطح دقت مورد انتظار):
ماده مربوط به دفاع مشروع → شرط باید چیزی مثل «تناسب دفاع با خطر احراز شده است» یا «امکان توسل به طرق قانونی دیگر وجود نداشته است» باشد — نه صرفاً «متهم برای دفاع از خود اقدام کرده است».
ماده مربوط به معاونت در جرم → شرط باید چیزی مثل «کمک یا تحریک متهم مقدم بر وقوع جرم اصلی بوده است» باشد — نه صرفاً «متهم در درگیری شرکت داشته است».

قانون طلایی دوم (خیلی مهم — قطبیت شرط‌ها): همیشه هر condition را طوری بنویس که satisfied=true به معنای «این عامل به نفع اعمال ماده است» باشد، نه برعکس.
- اگر متن ماده یک استثنا یا مانع دارد (مثل «...مگر اینکه X» یا «...به‌جز در صورت X»)، آن استثنا را مستقیماً به‌عنوان condition ننویس. آن را وارونه و مثبت بازنویسی کن: مثلاً به‌جای «متهم سابقه محکومیت دارد» (که وقتی false باشد گیج‌کننده است)، بنویس «استثنای سابقه‌ی محکومیت مصداق ندارد» و satisfied را بر همین اساس تعیین کن.
- اگر ماده یک تکلیف قانونی برای دادگاه/مرجع مقرر کرده و موضوع پرونده دقیقاً **نقض همان تکلیف** است (مثل زمینه‌های اعاده دادرسی که وقتی تکلیفی انجام نشده باشد راه باز می‌شود)، condition را از زاویه‌ی «آیا این زمینه برای اقدام حاضر [مثلاً اعاده دادرسی] فراهم است» بنویس، نه از زاویه‌ی «آیا آن تکلیف انجام شده است». مثال بد: «دادگاه مکلف به تعیین مجازات جایگزین است» با satisfied=false وقتی دادگاه این کار را نکرده — این وارونه است. مثال درست: «دادگاه مجازات جایگزین را علی‌رغم تکلیف قانونی تعیین نکرده است» با satisfied=true.
- قبل از نهایی کردن هر condition از خودت بپرس: «اگر satisfied=true باشد، آیا این واقعاً به معنای نزدیک‌تر شدن به اعمال ماده است؟» اگر جواب نه است، جمله را وارونه کن.

وظیفه‌ی تو:
۱. یک چک‌لیست تشخیصی از شرایط قانونی لازم برای اعمال **همین ماده** (طبق دو قانون طلایی بالا) بساز — فقط بر اساس متن ماده و شواهد نمایش داده‌شده، نه دانش خارجی.
۲. هر شرط را با توجه به شرح واقعیت و شواهد بالا ارزیابی کن و مشخص کن آیا برقرار است (satisfied) یا نه.
۳. برای هر شرط مشخص کن آیا یک شرط ضروری (necessary) برای اعمال ماده است یا صرفاً زمینه‌ای/توضیحی.

فقط یک آبجکت JSON با ساختار زیر برگردان، بدون هیچ متن اضافه. مقدار satisfied باید همیشه دقیقاً true یا false باشد (هیچ‌وقت "unknown" یا مقدار دیگر)؛ اگر مطمئن نیستی، بر اساس بهترین برداشتت از شواهد موجود true یا false انتخاب کن:
{{
  "checklist": [
    {{"condition": "...", "necessary": true, "satisfied": true}}
  ]
}}
"""


class VerificationError(Exception):
    """Base class for anything that stops verify_article from returning a
    usable checklist. Callers that don't need to distinguish the cause
    (e.g. AuditorService, which just skips the article) can catch this."""


class AuditorAPIError(VerificationError):
    """The LLM call itself failed: network error, invalid/missing API
    key, rate limit, quota, timeout, etc. This means verification did
    not run at all -- distinct from the model running and returning
    something unusable (see AuditorParseError)."""

    def __init__(self, provider: str, article_ref: str, cause: Exception):
        self.provider = provider
        self.article_ref = article_ref
        self.cause = cause
        super().__init__(
            f"{provider} API call failed for {article_ref}: "
            f"{type(cause).__name__}: {cause}"
        )


class AuditorParseError(VerificationError):
    """The LLM call succeeded but the response was not valid JSON, or
    didn't match the expected checklist schema. Includes a truncated
    copy of the raw response to help debug prompt/schema drift."""

    def __init__(self, provider: str, article_ref: str, raw_text: str | None, cause: Exception):
        self.provider = provider
        self.article_ref = article_ref
        self.raw_text = raw_text
        self.cause = cause
        preview = (raw_text or "")[:500]
        super().__init__(
            f"Could not parse {provider} response for {article_ref}: "
            f"{type(cause).__name__}: {cause}\n"
            f"raw response (truncated): {preview!r}"
        )


def _call_gemini(prompt: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=_GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )
    return response.text or ""




_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 5.0


def _post_chat_completion_with_retry(url: str, headers: dict, model: str, prompt: str) -> str:
    """
    Shared OpenAI-compatible chat-completions caller with 429 retry +
    exponential backoff, used by both Groq and OpenRouter. Free tiers on
    both can be exceeded mid-run even with a fixed delay between requests
    -- longer prompts (more evidence) burn the per-minute token budget
    faster than the per-minute request budget. Honors Retry-After when
    the provider sends one; otherwise backs off base * 2**attempt seconds.
    """
    last_error: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        logger.info(f"sending request to {url} (attempt {attempt + 1}/{_MAX_RETRIES})")
        response = requests.post(
            url,
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            },
            timeout=(10, 60),  # (connect_timeout, read_timeout) -- explicit split so a
                               # connection-level hang fails in 10s instead of possibly
                               # not honoring a bare int under some proxy setups
        )
        logger.info(f"got response from {url}: status={response.status_code}")

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            wait_seconds = (
                float(retry_after) if retry_after
                else _BACKOFF_BASE_SECONDS * (2 ** attempt)
            )
            logger.warning(
                f"{url} rate limit hit (attempt {attempt + 1}/{_MAX_RETRIES}), "
                f"waiting {wait_seconds:.1f}s"
            )
            last_error = requests.HTTPError(f"429 rate limited: {response.text[:200]}")
            time.sleep(wait_seconds)
            continue

        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        content = choice.get("message", {}).get("content")

        if not content:
            # Some free-tier models (observed with OpenRouter's gpt-oss-20b)
            # burn the whole output budget on reasoning tokens and return
            # content=None/"" with finish_reason="length" or similar. This
            # is not a network/HTTP failure, so it doesn't retry here --
            # it's surfaced to the caller as a clear error instead of
            # silently propagating None into json.loads().
            finish_reason = choice.get("finish_reason", "unknown")
            raise ValueError(
                f"empty completion content (finish_reason={finish_reason!r}); "
                f"the model likely exhausted its output budget on reasoning "
                f"tokens without producing a final answer"
            )

        return content

    raise last_error or RuntimeError(f"{url} call failed after retries")


_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_OPENROUTER_MODEL = "openai/gpt-oss-20b:free"
# google/gemma-4-26b-a4b-it:free
# openai/gpt-oss-20b:free
# z-ai/glm-5.2:free



def _call_groq(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    return _post_chat_completion_with_retry(_GROQ_URL, headers, _GROQ_MODEL, prompt)


def _call_openrouter(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    return _post_chat_completion_with_retry(_OPENROUTER_URL, headers, _OPENROUTER_MODEL, prompt)


_PROVIDERS = {
    "gemini": _call_gemini,
    "groq": _call_groq,
    "openrouter": _call_openrouter,
}


def verify_article(query: str, bundle: ArticleEvidenceBundle) -> list[ChecklistItemOut]:
    """
    Single combined LLM call per article: builds the diagnostic checklist
    for `bundle.article_ref` and evaluates every condition against `query`
    and `bundle.supporting_texts` in one pass. Checklist generation and
    item-wise verification are merged into one request to halve the
    number of LLM calls versus a two-step pipeline.

    Provider is chosen via settings.AUDITOR_LLM_PROVIDER ("groq",
    "openrouter", or "gemini"), defaulting to "groq". To switch providers,
    change only this ONE setting -- no code change needed:

        AUDITOR_LLM_PROVIDER = "openrouter"   # temporary, free-tier fallback
        AUDITOR_LLM_PROVIDER = "groq"         # previous default
        AUDITOR_LLM_PROVIDER = "gemini"       # production, once billing is set up

    Raises AuditorAPIError if the API call itself fails, or
    AuditorParseError if the call succeeds but the response can't be
    parsed into ChecklistItemOut records. Both subclass VerificationError.
    """
    provider_name = getattr(settings, "AUDITOR_LLM_PROVIDER", "groq")
    call = _PROVIDERS.get(provider_name)
    if call is None:
        raise ValueError(
            f"Unknown AUDITOR_LLM_PROVIDER={provider_name!r}, "
            f"expected one of {list(_PROVIDERS)}"
        )

    evidence_block = "\n".join(
        f"- {_truncate(t, _MAX_EVIDENCE_TEXT_CHARS)}" for t in bundle.supporting_texts
    )
    evidence_block = evidence_block or "(شواهد پشتیبان مستقیمی یافت نشد)"

    article_text = bundle.article_text or "(متن ماده در دسترس نیست)"
    article_text = _truncate(article_text, _MAX_ARTICLE_TEXT_CHARS)

    prompt = _PROMPT_TEMPLATE.format(
        query=query,
        article_ref=bundle.article_ref,
        article_text=article_text,
        evidence_block=evidence_block,
    )

    _MAX_PARSE_RETRIES = 2
    last_parse_error: Exception | None = None
    last_raw_text = ""

    for attempt in range(_MAX_PARSE_RETRIES):
        try:
            raw_text = call(prompt)
        except Exception as e:
            logger.error(f"{provider_name} API call failed for {bundle.article_ref}: {e}")
            raise AuditorAPIError(provider_name, bundle.article_ref, e) from e

        last_raw_text = raw_text
        try:
            raw = json.loads(raw_text)
            items = raw.get("checklist") if isinstance(raw, dict) else raw
            if not isinstance(items, list) or not items:
                raise ValueError(f"expected a non-empty list, got {items!r}")

            checklist: list[ChecklistItemOut] = []
            for item in items:
                try:
                    checklist.append(ChecklistItemOut(**item))
                except Exception as item_error:
                    # One malformed item (e.g. an extra/missing field) doesn't
                    # need to discard an otherwise-usable checklist -- skip
                    # just that item and keep going.
                    logger.warning(
                        f"Skipping malformed checklist item for {bundle.article_ref}: "
                        f"{item_error} -- item was {item!r}"
                    )
            if not checklist:
                raise ValueError("every checklist item failed to parse")
            return checklist
        except Exception as e:
            last_parse_error = e
            logger.warning(
                f"Parse attempt {attempt + 1}/{_MAX_PARSE_RETRIES} failed for "
                f"{bundle.article_ref}: {e} -- retrying call"
                if attempt + 1 < _MAX_PARSE_RETRIES else
                f"Auditor response parsing failed for {bundle.article_ref}: {e}"
            )

    raise AuditorParseError(provider_name, bundle.article_ref, last_raw_text, last_parse_error)