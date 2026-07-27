"""
features/configs.py — تعریف دسته‌های Feature و تنظیمات LLM برای لایه‌ی سوم
                       (Feature Extraction Graph)

چرا این فایل جدا از vocabulary_categorizer.py است؟
    چون این تنظیمات (نام دسته‌ها، توضیح‌شون، مدل LLM) در چند جای مختلف
    لازم می‌شن — هم در مرحله‌ی دسته‌بندیِ یک‌بارِ واژه‌نامه، هم بعداً در
    extractor.py (وقتی برای هر پرونده Feature استخراج می‌کنیم). دقیقاً
    مثل الگوی laws/configs.py که از parser.py و main_cases.py هر دو
    استفاده می‌شه.

چرا «Referenced Articles» اینجا نیست؟
    چون طبق تصمیم قبلی، آن Feature از قبل با regex در cases/parser.py
    استخراج می‌شه (cited_articles / text_cited_articles) و نباید دوباره
    با LLM بازتولید بشه. این فایل فقط دسته‌هایی را پوشش می‌دهد که واقعاً
    نیاز به LLM دارند.

چرا «Legal Facts» یک دسته‌ی جدا و بدون closed vocabulary است؟
    چون Factها (مثل «پرداخت انجام نشده») بی‌نهایت متنوع‌اند و نمی‌شه
    از قبل لیست بست‌شون؛ این‌ها باید آزادانه از متن استخراج بشن (با
    Evidence Span، طبق طرح اصلی). به همین دلیل در VOCAB_CATEGORIES
    نیست — چون قرار نیست از واژه‌نامه‌ی موجود دسته‌بندی بشه.

    نکته: پیشنهاد شد که Fact (وقتی مرتبط بود) به یک node ساختاریافته
    (Role/Action/Object) لینک اختیاری بخوره. این پیشنهاد عمداً رد شد —
    Fact باید همیشه متن آزاد و کامل بماند، بدون هیچ لینک یا وابستگی به
    closed vocabulary، تا اطلاعات پرونده هرس نشود.

چرا Normalization/Alias در همون pass دسته‌بندی انجام می‌شود، نه یک
pass دوم جدا؟
    چون یک pass دوم یعنی دوباره کل واژه‌نامه رو به LLM بدیم — هزینه‌ی
    دوبرابر برای چیزی که LLM می‌تونه همزمان با دسته‌بندی انجام بده.
    برای همین در vocabulary_categorizer.py، همراه با «دسته»، یک فیلد
    «ریشه» (canonical form) هم از LLM خواسته می‌شود: مثلاً «عقد بیع» و
    «قرارداد بیع» هر دو دسته=concept و ریشه="بیع" برمی‌گردانند.

چرا «principle» حذف شد؟
    چون در عمل (گزارش vocabulary_categorizer.py) از ۳۷۷ واژه‌ی خام
    فقط ۶ تا واقعاً «اصل حقوقی» معنادار بودند (۹۸٪ نویز — ادات ربط و
    قیدهایی مثل «با توجه به»، «مستنداً»، «هرچند»). آن ۶ واژه‌ی باقی‌مانده
    دستی به alias‌های concept منتقل شدند؛ این دسته دیگر در pipeline
    استخراج نمی‌شود.

چرا چند Provider LLM (نه فقط Gemini)؟
    چون روی هزاران فراخوانی (۲۴۲۱ پرونده)، حتی نرخ شکست کمِ یک provider
    (rate limit، قطعی موقت، کوتای تمام‌شده) یعنی ده‌ها پرونده گیر
    می‌کنن. به‌جای این‌که کل pipeline متوقف بشه، وقتی provider فعلی
    شکست خورد، خودکار می‌ریم سراغ provider بعدی در زنجیره.

چرا هنوز می‌شه دستی provider اول رو انتخاب کرد؟
    چون گاهی می‌خوای عمداً یک provider خاص (مثلاً DeepSeek R1 برای
    reasoning قوی‌تر، یا Cerebras برای سرعت) رو اول امتحان کنی، نه
    لزوماً همیشه Gemini. برای همین PROVIDER_ORDER پایین رو دستی
    می‌تونی عوض کنی — دقیقاً مثل قبل که فقط LLM_MODEL رو عوض می‌کردی،
    فقط حالا یک لیست ترتیب‌داره به‌جای یک مقدار تکی.
"""

from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class FeatureCategoryDef:
    key: str                # کلید داخلی، مثلاً "concept"
    label_fa: str            # نام فارسی برای نمایش/پرامپت
    description: str         # توضیح برای LLM، دقیقاً همون تعریفی که تصمیم گرفتیم
    examples: list[str]
    node_label: str          # لیبل نود در Neo4j، مثلاً "LegalConcept"
    relation_type: str       # نوع رابطه از Ruling، مثلاً "HAS_CONCEPT"
    from_closed_vocabulary: bool   # آیا این دسته باید از واژه‌نامه‌ی بسته انتخاب بشه؟


# دسته‌هایی که باید از data/legal-vocabulary/legal_vocabulary.json دسته‌بندی بشن
# (یعنی گام ۱ - vocabulary_categorizer.py روی این‌ها اجرا می‌شه)
VOCAB_CATEGORIES: dict[str, FeatureCategoryDef] = {
    "concept": FeatureCategoryDef(
        key="concept",
        label_fa="مفهوم حقوقی (Legal Concept)",
        description=(
            "یک مفهوم یا نهاد حقوقی انتزاعی که در حقوق مدنی/کیفری/تجاری "
            "تعریف‌شده است — نه یک شیء فیزیکی و نه یک نقش انسانی. شامل "
            "اصول کلی حقوقی هم می‌شود (مثل اصل برائت، اصل لزوم قرارداد). "
            "مثال‌ها: اهلیت، فسخ، خیار، عدم ایفای تعهد، اکراه، غرر، "
            "اصل برائت، اصل لزوم قرارداد."
        ),
        examples=["اهلیت", "فسخ", "خیار", "غرر", "اکراه", "اصل برائت", "اصل لزوم قرارداد"],
        node_label="LegalConcept",
        relation_type="HAS_CONCEPT",
        from_closed_vocabulary=True,
    ),
    "action": FeatureCategoryDef(
        key="action",
        label_fa="رفتار حقوقی (Legal Action)",
        description=(
            "یک فعل/رفتار حقوقی که یکی از طرفین دعوا انجام داده یا نداده "
            "است — معمولاً به‌صورت فعل یا مصدر. "
            "مثال‌ها: پرداخت کرد، پرداخت نکرد، فسخ کرد، مطالبه کرد، "
            "ابطال کرد، اقامه دعوا کرد، اعتراض کرد."
        ),
        examples=["پرداخت", "فسخ کرد", "مطالبه", "ابطال", "اقامه دعوا", "اعتراض"],
        node_label="LegalAction",
        relation_type="HAS_ACTION",
        from_closed_vocabulary=True,
    ),
    "role": FeatureCategoryDef(
        key="role",
        label_fa="نقش حقوقی (Legal Role)",
        description=(
            "عنوانی که یکی از طرفین دعوا یا یک شخص ثالث در پرونده بر عهده "
            "دارد. مثال‌ها: خواهان، خوانده، مستأجر، موجر، فروشنده، خریدار."
        ),
        examples=["خواهان", "خوانده", "مستأجر", "موجر", "فروشنده", "خریدار"],
        node_label="LegalRole",
        relation_type="HAS_ROLE",
        from_closed_vocabulary=True,
    ),
    "object": FeatureCategoryDef(
        key="object",
        label_fa="موضوع/شیء حقوقی (Legal Object)",
        description=(
            "یک شیء، سند، یا دارایی که موضوع دعوا یا معامله است. "
            "مثال‌ها: ملک، چک، قرارداد، سند، وجه نقد، وصیت‌نامه."
        ),
        examples=["ملک", "چک", "قرارداد", "سند", "وجه نقد", "وصیت‌نامه"],
        node_label="LegalObject",
        relation_type="HAS_OBJECT",
        from_closed_vocabulary=True,
    ),
}

# Legal Facts از واژه‌نامه دسته‌بندی نمی‌شه (آزاد از متن استخراج می‌شه)،
# ولی تعریفش برای استفاده‌ی یکسان در extractor.py و مستندسازی اینجا هست.
FACT_CATEGORY = FeatureCategoryDef(
    key="fact",
    label_fa="واقعه‌ی حقوقی (Legal Fact)",
    description=(
        "یک واقعه یا وضعیتِ رخ‌داده در پرونده که به‌صورت آزاد از متن "
        "استخراج می‌شود (نه از واژه‌نامه). مثال‌ها: پرداخت انجام نشده، "
        "قرارداد امضا شده، تحویل کالا انجام نشده، سند جعلی ارائه شده."
    ),
    examples=[
        "پرداخت انجام نشده",
        "قرارداد امضا شده",
        "تحویل کالا انجام نشده",
        "سند جعلی ارائه شده",
    ],
    node_label="LegalFact",
    relation_type="HAS_FACT",
    from_closed_vocabulary=False,
)

# دسته‌ای که یعنی «این واژه اصلاً یک Feature حقوقی مفید نیست» — چون
# legal_vocabulary.json یک دیکشنری عمومیه (شامل واژه‌های زبانی صرف مثل
# «ابعد: دورتر»)، نه فقط اصطلاحات فنی. بدون این گزینه، LLM مجبور می‌شه
# هر واژه را با زور در یکی از دسته‌ها جا بدهد و دقت پایین می‌آید.
SKIP_LABEL = "skip"

ALL_VOCAB_LABELS = list(VOCAB_CATEGORIES.keys()) + [SKIP_LABEL]


# ============================================================
# --- تنظیمات LLM با پشتیبانی چند Provider + Fallback خودکار ---
# ============================================================
#
# هر ۵ provider (Gemini، Groq، OpenRouter، GitHub Models، DeepSeek R1،
# Cerebras) نگه داشته شده‌اند — هیچ‌کدام حذف نشده. می‌تونی هم دستی
# انتخاب کنی provider اول کدوم باشه (PROVIDER_ORDER پایین)، هم بذاری
# خودکار fallback بین بقیه انجام بشه اگر اولی شکست خورد.

LLM_TEMPERATURE = 0.1

# --- نام مدل هرکدام از providerها ---
GEMINI_MODEL = "models/gemini-2.5-flash"       # https://ai.google.dev/gemini-api/docs/models
GROQ_MODEL = "llama-3.3-70b-versatile"          # https://console.groq.com/docs/models
OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct"  # https://openrouter.ai/models
GITHUB_MODEL = "gpt-4o-mini"                    # https://github.com/marketplace/models
DEEPSEEK_MODEL = "deepseek-reasoner"            # همون DeepSeek R1 — https://api-docs.deepseek.com
CEREBRAS_MODEL = "llama-3.3-70b"                # https://inference-docs.cerebras.ai/models

# چون بعضی فایل‌های قدیمی‌تر ممکنه مستقیم LLM_MODEL رو ایمپورت کنن،
# نگهش می‌داریم و برابر با provider پیش‌فرض (Gemini) می‌ذاریم — ولی
# در عمل هرکدوم از providerها با مدل خودشون صدا زده می‌شن، نه این
# متغیر.
LLM_MODEL = GEMINI_MODEL

# ترتیب اولویت providerها: این لیست رو دستی عوض کن تا provider اول
# (و بقیه به ترتیب fallback) رو خودت انتخاب کنی. اسم‌های مجاز:
# "gemini", "groq", "openrouter", "github", "deepseek", "cerebras"
#
# مثال: اگه می‌خوای DeepSeek R1 اول امتحان بشه (برای reasoning قوی‌تر)
# و بعدش Gemini fallback باشه:
#     PROVIDER_ORDER = ["deepseek", "gemini", "groq", "openrouter", "github", "cerebras"]
PROVIDER_ORDER = ["gemini", "groq", "openrouter", "github", "deepseek", "cerebras"]


class _MultiProviderChatClient:
    """
    بیرون دقیقاً شبیه یک کلاینت OpenAI-compatible است (چون
    extractor.py/vocabulary_categorizer.py با
    `_client.chat.completions.create(model=..., messages=..., temperature=...)`
    و `response.choices[0].message.content` کار می‌کنن)، ولی از داخل
    به‌ترتیب PROVIDER_ORDER چند provider رو امتحان می‌کنه تا یکی جواب
    بده. اگه provider فعلی خطا داد (rate limit، قطعی، کوتای تمام‌شده)،
    خودکار می‌ره سراغ provider بعدی در لیست — بدون این‌که کل extraction
    متوقف بشه.
    """

    class _Msg:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.message = _MultiProviderChatClient._Msg(content)

    class _Response:
        def __init__(self, content):
            self.choices = [_MultiProviderChatClient._Choice(content)]

    class _Completions:
        def __init__(self, providers: list[tuple[str, "callable"]]):
            self._providers = providers  # [(name, call_fn(prompt, temperature) -> str), ...]

        def create(self, model: str, messages: list[dict], temperature: float = 0.1, **_):
            prompt = messages[0]["content"]
            last_error = None
            for name, call_fn in self._providers:
                try:
                    text = call_fn(prompt, temperature)
                    return _MultiProviderChatClient._Response(text)
                except Exception as e:  # noqa: BLE001 — می‌خوایم هر خطایی رو fallback کنیم
                    last_error = e
                    print(f"  ⚠️ provider «{name}» شکست خورد ({e}) — رفتن سراغ provider بعدی...")
            raise RuntimeError(f"❌ همه‌ی providerها شکست خوردند. آخرین خطا: {last_error}")

    def __init__(self, providers: list[tuple[str, "callable"]]):
        self.chat = self
        self.completions = self._Completions(providers)


def _make_gemini_call():
    """
    فراخوانی Gemini مستقیم با SDK رسمی گوگل (google-genai). نیاز به
    GEMINI_API_KEY در .env دارد (بگیرش از
    https://aistudio.google.com/apikey).
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def call(prompt: str, temperature: float) -> str:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt],
            config=types.GenerateContentConfig(temperature=temperature),
        )
        return response.text

    return call


def _make_groq_call():
    """
    فراخوانی Groq (OpenAI-compatible). نیاز به GROQ_API_KEY در .env
    دارد (بگیرش از https://console.groq.com/keys).
    """
    from groq import Groq

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def call(prompt: str, temperature: float) -> str:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content

    return call


def _make_openrouter_call():
    """
    فراخوانی OpenRouter (OpenAI-compatible، proxy به چند مدل مختلف).
    نیاز به OPENROUTER_API_KEY در .env دارد
    (بگیرش از https://openrouter.ai/keys).
    """
    from openai import OpenAI

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))

    def call(prompt: str, temperature: float) -> str:
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content

    return call


def _make_github_call():
    """
    فراخوانی GitHub Models (OpenAI-compatible). نیاز به GITHUB_TOKEN
    در .env دارد (یک Personal Access Token از GitHub، بدون نیاز به
    scope خاصی برای Models). کاتالوگ کامل مدل‌ها:
    https://github.com/marketplace/models
    """
    from openai import OpenAI

    client = OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=os.getenv("GITHUB_TOKEN"),
    )

    def call(prompt: str, temperature: float) -> str:
        response = client.chat.completions.create(
            model=GITHUB_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content

    return call


def _make_deepseek_call():
    """
    فراخوانی DeepSeek R1 (OpenAI-compatible). نیاز به DEEPSEEK_API_KEY
    در .env دارد (بگیرش از https://platform.deepseek.com/api_keys).
    مدل "deepseek-reasoner" همون DeepSeek R1 است.
    """
    from openai import OpenAI

    client = OpenAI(base_url="https://api.deepseek.com", api_key=os.getenv("DEEPSEEK_API_KEY"))

    def call(prompt: str, temperature: float) -> str:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content

    return call


def _make_cerebras_call():
    """
    فراخوانی Cerebras Cloud (OpenAI-compatible، سرعت استنتاج خیلی بالا).
    نیاز به CEREBRAS_API_KEY در .env دارد
    (بگیرش از https://cloud.cerebras.ai/). کاتالوگ مدل‌ها:
    https://inference-docs.cerebras.ai/models
    """
    from openai import OpenAI

    client = OpenAI(base_url="https://api.cerebras.ai/v1", api_key=os.getenv("CEREBRAS_API_KEY"))

    def call(prompt: str, temperature: float) -> str:
        response = client.chat.completions.create(
            model=CEREBRAS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content

    return call


# نگاشت نام provider → تابع‌سازِ call. هر provider فقط وقتی به زنجیره
# اضافه می‌شه که API key مربوطه در .env تنظیم شده باشه.
_PROVIDER_FACTORIES = {
    "gemini": ("GEMINI_API_KEY", _make_gemini_call),
    "groq": ("GROQ_API_KEY", _make_groq_call),
    "openrouter": ("OPENROUTER_API_KEY", _make_openrouter_call),
    "github": ("GITHUB_TOKEN", _make_github_call),
    "deepseek": ("DEEPSEEK_API_KEY", _make_deepseek_call),
    "cerebras": ("CEREBRAS_API_KEY", _make_cerebras_call),
}


def get_llm_client():
    """
    کلاینت مشترک LLM برای همه‌ی ماژول‌های features/
    (vocabulary_categorizer.py, extractor.py). به‌ترتیب PROVIDER_ORDER
    امتحان می‌کند؛ اگر provider فعلی شکست بخورد (rate limit، قطعی،
    کوتای تمام‌شده)، خودکار به provider بعدی در لیست می‌رود.

    نیاز به متغیرهای محیطی در .env (هرکدام که می‌خوای فعال باشه):
        GEMINI_API_KEY      — https://aistudio.google.com/apikey
        GROQ_API_KEY        — https://console.groq.com/keys
        OPENROUTER_API_KEY  — https://openrouter.ai/keys
        GITHUB_TOKEN        — https://github.com/settings/tokens
        DEEPSEEK_API_KEY    — https://platform.deepseek.com/api_keys
        CEREBRAS_API_KEY    — https://cloud.cerebras.ai/

    اگر کلید یک provider تنظیم نشده باشد، آن provider به‌سادگی از
    زنجیره حذف می‌شود (نه این‌که خطا بدهد) — پس فقط گذاشتن یکی از
    کلیدها هم کار می‌کند، فقط fallback نخواهد داشت.

    ترتیب provider اول با PROVIDER_ORDER بالای همین فایل کنترل می‌شود؛
    برای انتخاب دستیِ provider اصلی، همان لیست را عوض کن.
    """
    providers = []
    for name in PROVIDER_ORDER:
        if name not in _PROVIDER_FACTORIES:
            print(f"  ⚠️ نام provider ناشناخته در PROVIDER_ORDER نادیده گرفته شد: «{name}»")
            continue
        env_key, factory = _PROVIDER_FACTORIES[name]
        if os.getenv(env_key):
            providers.append((name, factory()))

    if not providers:
        raise RuntimeError(
            "هیچ API key ای برای LLM تنظیم نشده. حداقل یکی از "
            "GEMINI_API_KEY / GROQ_API_KEY / OPENROUTER_API_KEY / "
            "GITHUB_TOKEN / DEEPSEEK_API_KEY / CEREBRAS_API_KEY را در .env بگذار."
        )

    print(f"🔌 LLM providerهای فعال (به‌ترتیب اولویت): {[name for name, _ in providers]}")
    return _MultiProviderChatClient(providers)


# --- تنظیمات batch برای vocabulary_categorizer.py (بدون تغییر) ---

# هر batch چند واژه بفرستیم. عدد کوچیک نگه داشته شده (نه ۲۴۱۶ تایی)
# چون: (۱) پرامپت طولانی‌تر یعنی احتمال خطای JSON بیشتر، و (۲) اگر یک
# batch خطا بده، فقط همون batch را دوباره می‌فرستیم نه کل واژه‌نامه را.
VOCAB_BATCH_SIZE = 60

# چند کاراکتر اول «معنی» برای هر واژه به LLM فرستاده بشه — کل تعریف
# لازم نیست (بعضی تعریف‌ها خیلی طولانی‌اند)، فقط برای رفع ابهام کافیه.
VOCAB_MEANING_CHAR_CAP = 180