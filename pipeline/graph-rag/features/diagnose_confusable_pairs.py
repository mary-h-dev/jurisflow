"""
features/diagnose_confusable_pairs.py — تست ارزون قبل از commit کردن به
                                          resolver مبتنی‌بر embedding

چرا این اسکریپت لازم بود؟
    قبل از این‌که دوباره یک پایلوت کامل بزنیم و باز هم snap های غلط
    ببینیم، باید مستقیماً بپرسیم: «آیا bge-m3 اصلاً می‌تونه بین «توقیف»
    و «توقف» (یا «صدور حکم» و «دوره محکومیت») تمایز بذاره؟» اگه جواب
    نه باشه، resolver مبتنی‌بر embedding هم بی‌فایده‌ست و باید کلاً
    استراتژی عوض بشه (مثلاً فقط exact/alias match، بدون هیچ fuzzy).
    این تست فقط چندتا فراخوانی embed_text می‌خواد (چند ثانیه)، نه یک
    پایلوت ۵۰-تایی کامل (چند دقیقه + هزینه‌ی LLM).

نحوه‌ی اجرا:
    uv run -m features.diagnose_confusable_pairs
"""

import math

from common.embedder import embed_text

# جفت‌هایی که واقعاً توی خروجی‌های قبلی دیدیم به‌اشتباه به‌هم snap شدن،
# به‌علاوه‌ی چند جفتِ کنترل (که باید واقعاً شبیه باشن، تا مطمئن بشیم
# آستانه بی‌معنی/خیلی سخت‌گیرانه نیست)
CONFUSABLE_PAIRS = [
    # (استخراج‌شده‌ی مدل, واژه‌ی واژه‌نامه که غلط snap شد, آیا باید شبیه باشن؟)
    ("توقیف", "توقف", False),                    # مصادره‌ی مال  vs  متوقف‌شدن — نباید شبیه باشن
    ("صدور حکم بر رفع توقیف", "دوره محکومیت", False),  # کاملاً بی‌ربط
    ("بطلان عقد اجاره", "عقد اجاره", False),        # «بطلان» باید فرق داشته باشه، نه گم بشه
    ("انقضاء مدت اجاره", "عقد اجاره", False),
    ("اعلان بطلان", "اعلام اینکه", False),           # اصلاً یه واژه‌ی معتبر نیست
    # --- جفت‌های کنترل: این‌ها *باید* شبیه باشن ---
    ("خوانده ردیف اول", "خوانده", True),
    ("اقامه دعوا", "اقامه دعوی", True),
    ("توقیف خودرو", "توقیف", True),
]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def run():
    print("🔬 تست تمایزِ embedding روی جفت‌های واقعاً مشکل‌سازِ قبلی\n")
    print(f"{'عبارت استخراج‌شده':<28} {'واژه‌نامه':<20} {'باید شبیه؟':<10} {'شباهت واقعی':<12}")
    print("-" * 75)

    problems = []
    for raw, vocab_word, should_be_similar in CONFUSABLE_PAIRS:
        v1 = embed_text(raw)
        v2 = embed_text(vocab_word)
        score = _cosine(v1, v2)

        verdict = "✅" if (score > 0.75) == should_be_similar else "🔴"
        print(f"{raw:<28} {vocab_word:<20} {'بله' if should_be_similar else 'نه':<10} {score:.3f} {verdict}")

        if (score > 0.75) != should_be_similar:
            problems.append((raw, vocab_word, score, should_be_similar))

    print()
    if not problems:
        print("✅ embedding این جفت‌ها را درست تشخیص داد — resolver مبتنی‌بر "
              "embedding برای این واژه‌نامه احتمالاً قابل‌اعتماده.")
    else:
        print(f"🔴 {len(problems)} جفت اشتباه تشخیص داده شد:")
        for raw, vocab_word, score, expected in problems:
            print(f"  - «{raw}» vs «{vocab_word}»: شباهت={score:.3f} "
                  f"(انتظار می‌رفت {'شبیه' if expected else 'غیرشبیه'} باشن)")
        print("\n⚠️ اگه اکثر این‌ها 🔴 بودن، یعنی embedding هم برای این جفت‌های خاص "
              "قابل‌اعتماد نیست — باید resolver محدود بشه به فقط exact/alias "
              "match (بدون هیچ fuzzy semantic)، حتی اگه یعنی recall کمتر بشه.")


if __name__ == "__main__":
    run()