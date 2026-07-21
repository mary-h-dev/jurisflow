"""
common/embedder.py — embedding با bge-m3 از طریق Ollama (محلی)

چرا Ollama و نه نسخه‌ی native (FlagEmbedding)؟
    نسخه‌ی native روی محیط با رم محدود (Codespace با ۹ گیگ) مدام OOM
    می‌شد. Ollama یک سرور جدا با مدیریت حافظه‌ی خودشه و پایدارتر اجراست
    — به قیمت دقت کمی پایین‌تر (چون مدلش کوانتیزه‌ست). این یک تصمیم
    آگاهانه برای عبور از گیرِ فعلیه، نه تصمیم نهایی — اگه بعداً منابع
    بیشتری در دسترس بود، می‌شه به native برگشت.

    ⚠️ نکته‌ی مهم برای بعد: embeddingهای این دو مدل با هم *سازگار نیستن*
    (فضای برداری متفاوت دارن). اگه یک روز مدل عوض شد، باید همه‌ی
    embeddingهای قبلی (چه Article/Note چه Ruling/RulingSection) از نو
    ساخته بشن — نمی‌شه قاطی‌شون کرد.

پیش‌نیاز: سرویس Ollama باید از قبل روشن باشه و مدل pull شده باشه:
    ollama pull bge-m3

نکته‌ی مهم درباره‌ی طول متن: برخلاف نسخه‌ی قبلیِ خودت، اینجا دیگه متن
رو با [:2000] قطع نمی‌کنیم — این دقیقاً همون باگی بود که دقت رو پایین
می‌آورد، مخصوصاً برای متن پرونده‌ها که خیلی طولانی‌تر از ۲۰۰۰ کاراکترن.
فقط یک سقف امنیتیِ خیلی سخاوتمندانه می‌ذاریم که صرفاً جلوی timeout روی
موارد کاملاً استثنایی رو بگیره، نه یک truncation واقعی.
"""


import time
import requests

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "bge-m3"

# سقف امنیتی پیش‌فرض (وقتی max_length مشخص نشده) — خیلی بزرگ‌تر از هر
# متن معمولی حقوقی، صرفاً برای جلوگیری از timeout روی موارد استثنایی
_DEFAULT_SAFETY_CHAR_CAP = 20000

# تخمین تقریبی نسبت کاراکتر به توکن برای فارسی — دقیق نیست، ولی برای
# تبدیل max_length (که واحدش توکنه) به یک سقف کاراکتری کافیه
_CHARS_PER_TOKEN_ESTIMATE = 4



def _resolve_char_cap(max_length: int | None) -> int:
    """
    Ollama برخلاف نسخه‌ی native، پارامتر max_length رو مستقیم قبول
    نمی‌کنه (کنترل context سمت سرور/Modelfile انجام می‌شه، نه per-request).
    برای همین، منطق tiering (512/2048/8192 توکن) رو با یک تخمین تقریبی
    به سقف کاراکتری تبدیل می‌کنیم — این‌جوری embed_all.py بدون تغییر کار
    می‌کنه و فرق نوع محتوا (ماده کوتاه در برابر بخش رأی طولانی) همچنان
    رعایت می‌شه، فقط این‌بار برای کاهش بار روی سرور Ollama، نه حافظه‌ی
    پایتون.
    """
    if max_length is None:
        return _DEFAULT_SAFETY_CHAR_CAP
    return max_length * _CHARS_PER_TOKEN_ESTIMATE


def embed_text(text: str, retries: int = 2, max_length: int | None = None) -> list[float]:
    char_cap = _resolve_char_cap(max_length)
    text = text[:char_cap]

    last_error = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={"model": MODEL_NAME, "prompt": text},
                timeout=120,
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < retries:
                print(f"  ⏳ خطا در embedding، تلاش دوباره ({attempt + 1}/{retries})...")
                time.sleep(3)

    raise last_error





def embed_batch(
    texts: list[str],
    batch_size: int = None,
    delay: float = 0.2,
    max_length: int | None = None,
) -> list[list[float]]:
    """
    Ollama batching واقعی سمت سرور نداره (هر درخواست یک متن)، پس یکی‌یکی
    صدا می‌زنیم. batch_size فقط برای سازگاری با فراخوانی‌های قبلی نگه
    داشته شده. max_length طبق _resolve_char_cap به سقف کاراکتری تبدیل
    و برای همه‌ی متن‌های این batch یکسان اعمال می‌شه.
    """
    embeddings = []
    for text in texts:
        embeddings.append(embed_text(text, max_length=max_length))
        if delay:
            time.sleep(delay)
    return embeddings