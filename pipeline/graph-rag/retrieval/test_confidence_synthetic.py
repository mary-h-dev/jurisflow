"""
test_confidence_synthetic.py — تست مصنوعی monotonicity و dynamic range

چرا این تست لازمه؟
    خروجی واقعی روی سه سؤال تستی امتیازهای خیلی نزدیک به هم داد
    (۰.۸۳۵ / ۰.۸۳۵ / ۰.۸۴۴). این می‌تونه به دو دلیل باشه:
    ۱. واقعاً retrieval برای هر سه سؤال به یک اندازه خوب بوده (چون هر
       سه روی حوزه‌ای بودن که داده‌ی زیاد و باکیفیت داشتیم)
    ۲. فرمول compute_retrieval_score به‌اندازه‌ی کافی حساس نیست
       (dynamic range کم)

این فایل به‌جای اتکا به داده‌ی واقعی (که نیاز به دیتابیس و کوئری واقعی
داره)، چهار سناریوی دستی و کنترل‌شده می‌سازه — از «همه‌چیز عالی» تا
«همه‌چیز ضعیف» — و بررسی می‌کنه:
    ۱. آیا ترتیب امتیازها منطقیه؟ (A > B > C > D)
    ۲. آیا فاصله‌ی بین امتیازها معنادار و قابل‌تشخیصه، یا فرمول بیش‌ازحد
       محافظه‌کارانه است و همه‌چیز رو نزدیک هم نشون می‌ده؟

این یک اثبات کامل calibration نیست (اون نیاز به داده‌ی برچسب‌خورده‌ی
واقعی داره)، بلکه یک شواهد اولیه‌ست که فرمول رفتار منطقی داره.

اجرا:
    uv run python -m retrieval.test_confidence_synthetic
"""

from retrieval.retriever import RetrievedEvidence
from retrieval.confidence import build_uncertainty_vector, compute_retrieval_score


class _FakeResult:
    """جایگزین سبک RetrievalResult — فقط سه لیست که confidence.py بهشون نیاز داره"""
    def __init__(self, feature_results, ruling_results, article_results):
        self.feature_results = feature_results
        self.ruling_results = ruling_results
        self.article_results = article_results


def _make_evidence(source_type: str, score: float, n: int = 3) -> list[RetrievedEvidence]:
    """n تا RetrievedEvidence ساختگی با یک امتیاز مشخص (برای top-N quality)"""
    return [
        RetrievedEvidence(source_type=source_type, ruling_id="fake", text="متن نمونه", score=score)
        for _ in range(n)
    ]


scenarios = {
    "A — همه‌چیز عالی":      dict(feature=0.9, ruling=0.9, article=0.9),
    "B — article غایب":      dict(feature=0.9, ruling=0.9, article=None),
    "C — کیفیت متوسط/ضعیف":  dict(feature=0.9, ruling=0.3, article=0.2),
    "D — همه‌چیز ضعیف":       dict(feature=0.3, ruling=0.2, article=0.1),
}

print("سناریو".ljust(28), "امتیاز نهایی", "  Feature/Ruling/Article", "  Missing")
print("-" * 80)

results = []
for name, cfg in scenarios.items():
    feature = _make_evidence("feature", cfg["feature"]) if cfg["feature"] is not None else []
    ruling = _make_evidence("ruling", cfg["ruling"]) if cfg["ruling"] is not None else []
    article = _make_evidence("article", cfg["article"]) if cfg["article"] is not None else []

    fake_result = _FakeResult(feature, ruling, article)
    vec = build_uncertainty_vector(fake_result)
    score = compute_retrieval_score(vec)
    results.append((name, score))

    print(
        f"{name:<28} {score:.3f}          "
        f"{vec.feature_quality:.2f}/{vec.ruling_quality:.2f}/{vec.article_quality:.2f}"
        f"          {vec.missing_channels or '-'}"
    )

print("\n--- بررسی خودکار ---")

scores_only = [s for _, s in results]
is_monotonic = all(scores_only[i] > scores_only[i + 1] for i in range(len(scores_only) - 1))
print(f"ترتیب نزولی (A > B > C > D)؟  {'✅ بله' if is_monotonic else '❌ خیر — مشکل داره!'}")

spread = max(scores_only) - min(scores_only)
print(f"دامنه‌ی امتیازها (بیشترین - کمترین): {spread:.3f}")
if spread < 0.3:
    print("⚠️ دامنه کمتر از ۰.۳ است — فرمول ممکنه بیش‌ازحد محافظه‌کارانه باشه.")
else:
    print("✅ دامنه به‌اندازه‌ی کافی بزرگه — فرمول بین سناریوهای خوب و بد فرق می‌ذاره.")