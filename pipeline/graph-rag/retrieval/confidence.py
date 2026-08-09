"""
retrieval/confidence.py — محاسبه‌ی امتیاز اطمینان retrieval (ورودی uncertainty estimator)

چرا این فایل از retriever.py جداست؟
    اجرای کوئری‌ها (retriever.py) و محاسبه‌ی امتیاز اطمینان دو مسئولیت
    کاملاً متفاوتن — تغییر فرمول امتیازدهی نباید نیاز به دست‌زدن به منطق
    جست‌وجو داشته باشه، و برعکس. این جداسازی همچنین باعث می‌شه بشه این
    ماژول رو مستقل و بدون اتصال به Neo4j تست کرد (کافیه چند
    RetrievedEvidence دستی بسازی).

چرا میانگین وزن‌دار ساده‌ی top-1 هر کانال کافی نبود:
    ۱. فقط بهترین نتیجه‌ی هر کانال رو می‌بینه، نه کیفیت کلی نتایج.
    ۲. وقتی یک کانال (مثلاً article) هیچ نتیجه‌ای نداره، روش قبلی وزنش
       رو بی‌صدا بین کانال‌های دیگه پخش می‌کرد (re-normalize) — یعنی
       عدد نهایی هیچ‌وقت نشون نمی‌داد که «هیچ ماده‌ی قانونی پیدا نشده»؛
       فقط یه عدد معمولی برمی‌گردوند که این نبود رو پنهان می‌کرد.

چرا فقط rrf_score هم کافی نبود:
    RRF بر پایه‌ی رتبه‌ست، نه مقدار امتیاز — یعنی «چندمین نتیجه بودی»،
    نه «چقدر مطمئنی». یک match با rank ۱ ولی شباهت معنایی ۰.۶۰، همون
    امتیاز rank ۱ با شباهت ۰.۹۹ رو می‌گیره. RRF برای ترکیب چند لیست
    نتایج (همون _fuse_with_rrf توی retriever.py) عالیه، ولی برای
    گزارش‌دهی سطح‌بالای «چقدر می‌شه به این جواب اعتماد کرد» گمراه‌کننده‌ست.

راه‌حل: به‌جای یک عدد تنها، یک بردار تشخیصی می‌سازیم — کیفیت هر کانال
    جدا قابل گزارشه (مثلاً «کانال article خالی بود») — و امتیاز نهایی از
    روی همین بردار محاسبه می‌شه، با جریمه‌ی صریح برای کانال‌های غایب،
    نه پنهان‌کاری با re-normalize کردن وزن‌ها.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # فقط برای type hint — هیچ وابستگی واقعی به retriever.py در زمان اجرا نیست
    from retrieval.retriever import RetrievalResult, RetrievedEvidence


# وزن هر کانال در امتیاز نهایی — مجموعشون ۱ است.
# چرا feature/ruling سنگین‌تر از article؟ چون سؤال‌های حقوقی معمولاً
# اول با «واقعیت پرونده» (feature) و «سابقه‌ی قضایی مشابه» (ruling)
# پاسخ داده می‌شن؛ ماده‌ی قانونی (article) معمولاً استناد تکمیلیه.
_CHANNEL_WEIGHTS = {
    "feature": 0.4,
    "ruling": 0.4,
    "article": 0.2,
}

# جریمه‌ی ثابت به‌ازای هر کانالی که هیچ نتیجه‌ای نداشته (missing channel).
# صریح و قابل‌توضیح بودنش مهم‌تر از دقیق‌بودنش است — عدد را می‌توان
# بعداً روی داده‌ی واقعی (labeled queries) کالیبره کرد.
_MISSING_CHANNEL_PENALTY = 0.15

# چند تا نتیجه‌ی برتر هر کانال برای محاسبه‌ی «کیفیت میانگین» در نظر گرفته بشه.
# فقط top-1 نویزپذیره؛ میانگین همه هم گمراه‌کننده‌ست (نتایج ضعیفِ ته لیست
# رو هم حساب می‌کنه). top-3 یک نقطه‌ی میانی معقوله.
_TOP_N_FOR_QUALITY = 3


@dataclass
class RetrievalUncertaintyVector:
    """
    تشخیص جزء‌به‌جزء کیفیت retrieval — هر بعد جدا قابل گزارش و بررسیه،
    برخلاف یک عدد تنها که دلیل پایین‌بودنش رو پنهان می‌کنه. این چیزیه که
    باید به uncertainty estimator و به گزارش نهایی (auditor) پاس داده بشه.
    """
    feature_quality: float                       # میانگین امتیاز top-N نتایج feature (۰ اگه خالی)
    ruling_quality: float                         # میانگین امتیاز top-N نتایج ruling (۰ اگه خالی)
    article_quality: float                        # میانگین امتیاز top-N نتایج article (۰ اگه خالی)
    missing_channels: list[str] = field(default_factory=list)

    @property
    def channel_qualities(self) -> dict[str, float]:
        return {
            "feature": self.feature_quality,
            "ruling": self.ruling_quality,
            "article": self.article_quality,
        }


def _channel_quality(results: list["RetrievedEvidence"], top_n: int = _TOP_N_FOR_QUALITY) -> float:
    """میانگین امتیاز N نتیجه‌ی برتر یک کانال؛ اگه کانال خالی بود، صفر."""
    if not results:
        return 0.0
    top = results[:top_n]
    return sum(ev.score for ev in top) / len(top)


def build_uncertainty_vector(result: "RetrievalResult") -> RetrievalUncertaintyVector:
    """از روی یک RetrievalResult (خروجی retriever.retrieve)، بردار تشخیصی کیفیت می‌سازه."""
    missing = []
    if not result.feature_results:
        missing.append("feature")
    if not result.ruling_results:
        missing.append("ruling")
    if not result.article_results:
        missing.append("article")

    return RetrievalUncertaintyVector(
        feature_quality=_channel_quality(result.feature_results),
        ruling_quality=_channel_quality(result.ruling_results),
        article_quality=_channel_quality(result.article_results),
        missing_channels=missing,
    )


def compute_retrieval_score(vec: RetrievalUncertaintyVector) -> float:
    """
    امتیاز نهایی بین ۰ تا ۱ — ورودی uncertainty estimator.
    میانگین وزن‌دار کیفیت کانال‌های فعال، منهای جریمه‌ی صریح به‌ازای هر
    کانال غایب. برخلاف re-normalize کردن، اینجا نبودِ یک کانال همیشه
    امتیاز نهایی رو پایین می‌آره — پنهان نمی‌شه.
    """
    weighted_sum = (
        _CHANNEL_WEIGHTS["feature"] * vec.feature_quality
        + _CHANNEL_WEIGHTS["ruling"] * vec.ruling_quality
        + _CHANNEL_WEIGHTS["article"] * vec.article_quality
    )
    penalty = _MISSING_CHANNEL_PENALTY * len(vec.missing_channels)
    return max(0.0, min(1.0, weighted_sum - penalty))


def score_result(result: "RetrievalResult") -> tuple[float, RetrievalUncertaintyVector]:
    """
    تابع راحت برای استفاده‌ی معمول: هم بردار تشخیصی هم امتیاز نهایی رو
    با یک فراخوانی برمی‌گردونه.

    نحوه‌ی استفاده:
        from retrieval.confidence import score_result
        score, vec = score_result(result)
        print(score, vec.missing_channels)
    """
    vec = build_uncertainty_vector(result)
    return compute_retrieval_score(vec), vec