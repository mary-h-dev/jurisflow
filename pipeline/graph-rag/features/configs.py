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
    با LLM بازتولید بشه. این فایل فقط ۵ دسته‌ای را پوشش می‌دهد که واقعاً
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
    «قرارداد بیع» هر دو دسته=concept و ریشه="بیع" برمی‌گردانند. خروجی
    نهایی per-category یک لیست از مفاهیم *ریشه* (نه همه‌ی ۲۴۱۶ واژه‌ی
    خام) است، به‌همراه نگاشت alias→ریشه برای ردیابی.

    هدف عددی ۴۰۰-۶۰۰ concept که پیشنهاد شد یک حدس منطقیه، نه یک قانون
    اجباری — کد آن را به‌زور تحمیل نمی‌کند؛ فقط بعد از اجرا در گزارش
    نهایی تعداد واقعی مفاهیم ریشه چاپ می‌شود تا خودت قضاوت کنی.
"""

from dataclasses import dataclass
import os


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
            "تعریف‌شده است — نه یک شیء فیزیکی و نه یک نقش انسانی. "
            "مثال‌ها: اهلیت، فسخ، خیار، عدم ایفای تعهد، اکراه، غرر."
        ),
        examples=["اهلیت", "فسخ", "خیار", "غرر", "اکراه"],
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
    "principle": FeatureCategoryDef(
        key="principle",
        label_fa="اصل حقوقی (Legal Principle)",
        description=(
            "یک اصل کلی و پایه‌ای حقوقی که مبنای استدلال قضایی قرار "
            "می‌گیرد — نه یک نهاد حقوقی مشخص (مثل «فسخ») بلکه یک قاعده‌ی "
            "بنیادین‌تر که در استدلال به آن استناد می‌شود. "
            "مثال‌ها: اصل برائت، اصل لزوم قرارداد، اصل نسبی بودن "
            "قراردادها، اصل حاکمیت اراده."
        ),
        examples=["اصل برائت", "اصل لزوم قرارداد", "اصل نسبی بودن قراردادها", "اصل حاکمیت اراده"],
        node_label="LegalPrinciple",
        relation_type="HAS_PRINCIPLE",
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
# هر واژه را با زور در یکی از ۴ دسته جا بدهد و دقت پایین می‌آید.
SKIP_LABEL = "skip"

ALL_VOCAB_LABELS = list(VOCAB_CATEGORIES.keys()) + [SKIP_LABEL]


# --- تنظیمات LLM ---
# چرا OpenRouter الان و نه Groq؟
#     به rate limit روی حساب رایگان Groq خوردیم. OpenRouter هم از همون
#     رابط OpenAI-compatible استفاده می‌کند، پس فقط base_url/api_key
#     عوض می‌شود، نه منطق فراخوانی (.chat.completions.create با همان
#     پارامترها کار می‌کند). بسته‌ی openai از قبل در pyproject.toml این
#     پروژه هست (>=2.44.0)، پس نیازی به نصب چیز جدیدی نیست.
#
# چرا client-creation یک‌جا اینجاست، نه در هر فایل (vocabulary_categorizer.py،
# extractor.py) جدا؟
#     چون سوییچِ provider باید فقط در یک نقطه انجام شود — وگرنه وقتی
#     provider دوباره عوض شد، احتمال دارد یکی از فایل‌ها فراموش شود و
#     پروژه با دو provider ناهماهنگ اجرا شود.
LLM_TEMPERATURE = 0.1

# LLM_MODEL = "llama-3.3-70b-versatile"  # Groq — قبلی، به‌خاطر rate limit کنار گذاشته شد (حذف نشده)
LLM_MODEL = "meta-llama/llama-3.3-70b-instruct"  # OpenRouter — لیست کامل مدل‌ها: https://openrouter.ai/models


def get_llm_client():
    """
    کلاینت مشترک LLM برای همه‌ی ماژول‌های features/ (vocabulary_categorizer.py, extractor.py).
    نیاز به متغیر محیطی OPENROUTER_API_KEY در .env دارد
    (بگیرش از https://openrouter.ai/keys).
    """
    # --- Groq (قبلی) — به‌خاطر rate limit روی حساب رایگان کنار گذاشته شد ---
    # from groq import Groq
    # return Groq(api_key=os.getenv("GROQ_API_KEY"))

    # --- OpenRouter (فعلی) ---
    from openai import OpenAI
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

# هر batch چند واژه بفرستیم. عدد کوچیک نگه داشته شده (نه ۲۴۱۶ تایی)
# چون: (۱) پرامپت طولانی‌تر یعنی احتمال خطای JSON بیشتر، و (۲) اگر یک
# batch خطا بده، فقط همون batch را دوباره می‌فرستیم نه کل واژه‌نامه را.
VOCAB_BATCH_SIZE = 20

# چند کاراکتر اول «معنی» برای هر واژه به LLM فرستاده بشه — کل تعریف
# لازم نیست (بعضی تعریف‌ها خیلی طولانی‌اند)، فقط برای رفع ابهام کافیه.
VOCAB_MEANING_CHAR_CAP = 180