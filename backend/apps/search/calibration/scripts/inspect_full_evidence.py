"""
Prints ALL evidence (feature/ruling/article) for a single query, with
full text, so you can inspect exactly what the feature channel is
returning and which rulings it's drawing from.

Run:
 DJANGO_SETTINGS_MODULE=config.settings.development python -c "import django; django.setup(); from apps.search.calibration.scripts.inspect_full_evidence import run; run()"
"""

from apps.search.services import search_service

# _QUERY = "در یک تصادف رانندگی که خودم مقصر بودم، همسر و فرزنم فوت کردند. آیا من به عنوان تنها وارث اون‌ها حق دریافت و مطالبه دیه فوتشون رو دارم؟"

_QUERY = "دادگاه منو به خاطر زنای غیرمحصنه به صد ضربه شلاق محکوم کرده و گفته هم اقرار کردم هم علم قاضی داره. حکم رو هم گفته علنی اجرا کنن. آیا دادگاه باید دقیق بگه دلیل اثبات جرم چیه و می‌تونه همزمان به اقرار و علم قاضی استناد کنه؟ اجرای علنی حد هم حتماً لازمه؟"

# _QUERY = "من و یکی دیگه به خاطر نگهداری مقدار زیادی تریاک به صورت زنجیره‌ای به اعدام محکوم شدیم. یکی از افراد اصلی پرونده هنوز دستگیر نشده. آیا برای اینکه جرم زنجیره‌ای حساب بشه حتماً باید حداقل سه نفر مشارکت داشته باشن؟"
_TOP_K = 30


def run():
    result = search_service.search(
        _QUERY,
        feature_top_k=_TOP_K,
        ruling_top_k=_TOP_K,
        article_top_k=_TOP_K,
        final_top_k=_TOP_K,
    )

    print("=" * 70)
    print(f"FEATURE EVIDENCE ({len(result.feature_results)})")
    print("=" * 70)
    for i, e in enumerate(result.feature_results, 1):
        print(f"\n{i}. [{e.feature_category}] value={e.feature_value!r}  score={e.score:.3f}  confidence={e.confidence}")
        print(f"   ruling_id: {e.ruling_id}")
        print(f"   text: {e.text}")

    print("\n" + "=" * 70)
    print(f"RULING EVIDENCE ({len(result.ruling_results)})")
    print("=" * 70)
    for i, e in enumerate(result.ruling_results, 1):
        print(f"\n{i}. ruling_id={e.ruling_id}  score={e.score:.3f}")
        print(f"   text: {e.text}")
        print(f"   cited_articles: {e.cited_articles}")

    print("\n" + "=" * 70)
    print(f"ARTICLE EVIDENCE ({len(result.article_results)})")
    print("=" * 70)
    for i, e in enumerate(result.article_results, 1):
        print(f"\n{i}. {e.law_name} - ماده {e.article_number}  score={e.score:.3f}")
        print(f"   text: {e.text[:300]}")

    # Unique ruling_ids touched by feature or ruling channel
    all_ruling_ids = set()
    for e in result.feature_results + result.ruling_results:
        if e.ruling_id:
            all_ruling_ids.add(e.ruling_id)

    print("\n" + "=" * 70)
    print(f"ALL RULING_IDS TOUCHED ({len(all_ruling_ids)})")
    print("=" * 70)
    print(sorted(all_ruling_ids))


if __name__ == "__main__":
    run()