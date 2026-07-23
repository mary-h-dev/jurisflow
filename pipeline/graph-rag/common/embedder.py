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

نکته‌ی مهم درباره‌ی طول متن (نسخه‌ی دوم): نسخه‌ی اول فقط با یک سقف
کاراکتریِ بزرگ (max_length × ۴) کار می‌کرد و همه‌چیز رو در یک درخواست
می‌فرستاد — این برای بخش‌های خیلی طولانیِ رأی (چند هزار کلمه) باعث
می‌شد خودِ سرور Ollama با خطای 500 کرش کنه، چون context واقعیِ
سرویس‌دهیِ Ollama معمولاً کوچیک‌تر از ظرفیت نظریِ مدل (۸۱۹۲) پیکربندی
شده. راه‌حل: به‌جای truncate یا فرستادن یک‌جا، متن طولانی رو به
تکه‌های امن (chunk) تقسیم می‌کنیم، هرکدوم رو جدا embed می‌کنیم، و
میانگین بردارها رو برمی‌گردونیم — این‌جوری هیچ محتوایی گم نمی‌شه و
درخواست هم هیچ‌وقت از حد امن Ollama رد نمی‌شه.
"""

import time
import requests

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "bge-m3"

# سقف امنیتی پیش‌فرض (وقتی max_length مشخص نشده)
_DEFAULT_SAFETY_CHAR_CAP = 20000

# تخمین تقریبی نسبت کاراکتر به توکن برای فارسی
_CHARS_PER_TOKEN_ESTIMATE = 4

# حداکثر اندازه‌ی *هر تکه* که در یک درخواست تکی به Ollama فرستاده می‌شه.
# این عدد عمداً محافظه‌کارانه و مستقل از max_length کاربره — چون هدفش
# جلوگیری از کرش سرور Ollamaست، نه رعایت ظرفیت نظری مدل.
_SAFE_CHUNK_SIZE = 3000
_CHUNK_OVERLAP = 200


def _resolve_char_cap(max_length: int | None) -> int:
    """
    سقف کلیِ محتوایی که پردازش می‌شه (نه اندازه‌ی هر درخواست تکی — اون
    رو _SAFE_CHUNK_SIZE کنترل می‌کنه). اگه متن از این سقف بزرگ‌تر بود،
    قبل از chunk‌کردن، تا همین‌جا کوتاه می‌شه.
    """
    if max_length is None:
        return _DEFAULT_SAFETY_CHAR_CAP
    return max_length * _CHARS_PER_TOKEN_ESTIMATE


def _chunk_text(text: str) -> list[str]:
    """تقسیم متن به تکه‌های امن با هم‌پوشانی کوچیک (تا مرز جمله‌ها قطع نشه)"""
    if len(text) <= _SAFE_CHUNK_SIZE:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + _SAFE_CHUNK_SIZE
        chunks.append(text[start:end])
        start = end - _CHUNK_OVERLAP
    return chunks


def _average_vectors(vectors: list[list[float]]) -> list[float]:
    if len(vectors) == 1:
        return vectors[0]
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]


def _embed_single_request(text: str, retries: int = 2) -> list[float]:
    """یک درخواست تکی به Ollama — فرض بر اینه که text از قبل به اندازه‌ی امن chunk شده"""
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


def embed_text(text: str, retries: int = 2, max_length: int | None = None) -> list[float]:
    char_cap = _resolve_char_cap(max_length)
    text = text[:char_cap]

    chunks = _chunk_text(text)
    if len(chunks) > 1:
        print(f"  ✂️  متن طولانی به {len(chunks)} تکه تقسیم شد (برای جلوگیری از کرش Ollama)")

    vectors = [_embed_single_request(chunk, retries=retries) for chunk in chunks]
    return _average_vectors(vectors)


def embed_batch(
    texts: list[str],
    batch_size: int = None,
    delay: float = 0.2,
    max_length: int | None = None,
) -> list[list[float]]:
    """
    Ollama batching واقعی سمت سرور نداره (هر درخواست یک متن)، پس یکی‌یکی
    صدا می‌زنیم. batch_size فقط برای سازگاری با فراخوانی‌های قبلی نگه
    داشته شده.
    """
    embeddings = []
    for text in texts:
        embeddings.append(embed_text(text, max_length=max_length))
        if delay:
            time.sleep(delay)
    return embeddings