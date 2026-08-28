"""
Same diagnostic as inspect_single_ruling.py, but takes the annotation
inline (new format with excluded_articles / gold_checklist_by_article)
instead of loading from the main annotation file — useful for testing a
newly-restructured sample before it's merged into case_grounded_100.json.

Shows exactly what the retrieval system (search_service.search) returns:
- which channels the router picked, and why (routing_confidence, intent)
- article channel hits (direct embedding) vs ruling citation hits
- for EACH gold article: was it found? at what rank? by which channel?
- for EACH excluded article: did retrieval also surface it? (false-positive risk)
- confidence/uncertainty vector for the whole query

Run:
DJANGO_SETTINGS_MODULE=config.settings.development python -c "import django; django.setup(); from apps.search.calibration.scripts.inspect_single_ruling_v2 import run; run()"
"""

from __future__ import annotations
from apps.search.services import search_service
from apps.search.calibration.metrics import normalize_refs

_PROBE_K = 30



# ---------------------------------------------------------------------------
# Inline annotation (ruling_id=10362) — paste any new-format sample here
# ---------------------------------------------------------------------------

_SAMPLE = {
    "ruling_id": "10493",
    "case_type": "حقوقی",
    "query": "در یک مبایعه‌نامه خریدار متعهد شده بود الباقی ثمن معامله را در تاریخ مشخصی بپردازد و شرط شده بود به ازای هر روز تأخیر مبلغ مشخصی وجه التزام بدهد. با توجه به عدم پرداخت به موقع پول، آیا فروشنده می‌تواند وجه التزام قراردادی مطالبه کند یا فقط خسارت تأخیر تأدیه تعلق می‌گیرد؟",
    "gold_articles": [
      "قانون آیین دادرسی مدنی - ماده ۵۲۲",
      "قانون مدنی - ماده ۲۳۲"
    ],
    "excluded_articles": [
      {
        "law_name": "قانون آیین دادرسی مدنی",
        "article_number": "۵۱۵",
        "reason": "تبصره ۲ این ماده به طور کلی اشاره به قابل مطالبه بودن خسارت تأخیر تأدیه در موارد قانونی دارد، اما اصل ضابطه و نحوه جبران خسارت تعهدات پولی در ماده ۵۲۲ مقرر شده است."
      },
      {
        "law_name": "قانون مدنی",
        "article_number": "۲۲۸",
        "reason": "این ماده مربوط به خسارت تأخیر در پرداخت وجه نقد در قانون مدنی است که طبق رای تجدیدنظر، ماده ۵۲۲ قانون آیین دادرسی مدنی بر آن حاکم دانسته شده و مبنای اصلی رای نیست."
      },
      {
        "law_name": "قانون مدنی",
        "article_number": "۲۳۰",
        "reason": "این ماده مربوط به اعتبار وجه التزام قراردادی است، اما دادگاه تجدیدنظر شمول آن را بر تعهدات پولی نفی کرده و شرط را نامشروع دانسته است."
      },
      {
        "law_name": "قانون مدنی",
        "article_number": "۲۹۳",
        "reason": "دادگاه بدوی بر اساس این ماده (تبدیل تعهد) حکم بر بطلان داده بود، اما دادگاه تجدیدنظر صریحاً استدلال مبنی بر تبدیل تعهد را رد کرد و مبنای رای را تغییر داد."
      },
      {
        "law_name": "قانون مدنی",
        "article_number": "۲۵۷",
        "reason": "مربوط به استدلال دادگاه بدوی بوده و در رای نهایی تجدیدنظر نقشی ندارد."
      },
      {
        "law_name": "قانون مجازات اسلامی",
        "article_number": "۵۹۵",
        "reason": "اگرچه دادگاه به عنوان تقویت استدلال غیرقانونی/ربوی بودن دریافت مازاد اشاره کرده، اما دعوی حقوقی مطالبه وجه التزام بر اساس بطلان شرط و حاکمیت ماده ۵۲۲ رد شده است."
      },
      {
        "law_name": "قانون آیین دادرسی مدنی",
        "article_number": "۲",
        "reason": "ماده عمومی و شکلی مربوط به عدم قابلیت استماع دعوی است."
      },
      {
        "law_name": "قانون آیین دادرسی مدنی",
        "article_number": "۳۵۳",
        "reason": "ماده شکلی تجدیدنظر در خصوص قرار بودن رای تلقی شده است."
      },
      {
        "law_name": "قانون آیین دادرسی مدنی",
        "article_number": "۳۵۸",
        "reason": "ماده شکلی تجدیدنظر جهت تایید رای بدوی با اصلاح نتیجه به قرار رد دعوی است."
      }
    ],
    "official_citation_check": [
      {
        "law_name": "قانون آیین دادرسی مدنی",
        "article_number": "۵۲۲",
        "status": "validated_as_gold",
        "reason": "در متن رای تجدیدنظر صریحاً ذکر شده و مبنای اصلی دادگاه برای حاکم بودن خسارت تأخیر تأدیه بر تعهدات پولی و عدم امکان مطالبه وجه التزام مازاد است."
      },
      {
        "law_name": "قانون مدنی",
        "article_number": "۲۳۲",
        "status": "validated_as_gold",
        "reason": "دادگاه تجدیدنظر شرط تعیین وجه التزام برای تعهد پولی را به استناد بند ۳ این ماده (شرط نامشروع/باطل) باطل دانسته است."
      }
    ],
    "gold_routing": {
      "feature_channel": true,
      "ruling_channel": false,
      "article_channel": true,
      "rationale": "article_channel لازم است زیرا مبنای قانونی عدم تعلق وجه التزام به تعهدات پولی، حاکمیت ماده ۵۲۲ قانون آیین دادرسی مدنی و باطل بودن شرط غیرمشروع طبق ماده ۲۳۲ قانون مدنی است؛ feature_channel لازم است زیرا باید موضوع تعهد (پولی بودن/پرداخت ثمن) و نوع شرط (وجه التزام روزانه) تطبیق داده شود؛ ruling_channel نیاز نیست زیرا رای بر اساس قواعد عام مادتین فوق صادر شده است."
    },
    "gold_checklist_by_article": {
      "قانون آیین دادرسی مدنی - ماده ۵۲۲": [
        {
          "condition": "موضوع تعهد، دین و از نوع وجه رایج (تعهد پولی) است",
          "necessary": true,
          "satisfied": true
        },
        {
          "condition": "خسارت درخواستی ناشی از تأخیر در پرداخت وجه نقد است",
          "necessary": true,
          "satisfied": true
        },
        {
          "condition": "جبران خسارت تأخیر در تعهدات پولی تابع تغییر شاخص سالانه بانک مرکزی است",
          "necessary": true,
          "satisfied": true
        }
      ],
      "قانون مدنی - ماده ۲۳۲": [
        {
          "condition": "شرط توافق شده در قرارداد با قوانین آمره و مقررات قانونی مغایرت دارد (شرط نامشروع)",
          "necessary": true,
          "satisfied": true
        },
        {
          "condition": "شرط باطل و غیرقابل استماع در محاکم است",
          "necessary": true,
          "satisfied": true
        }
      ]
    },
    "annotation_confidence": 0.95,
    "needs_human_review": false,
    "review_note": "نام قوانین باLAW_CONFIGS سیستم مچ شد. کلیه استنادها به قانون آیین دادرسی مدنی ساده‌سازی شدند تا در پایگاه داده دقیقاً تطبیق داده شوند."
  }


def _norm(ref: str) -> str:
    return next(iter(normalize_refs([ref])))


def _article_refs_from_evidences(evidences):
    refs, seen = [], set()
    for e in evidences:
        if e.source_type != "article" or not e.law_name or not e.article_number:
            continue
        ref = f"{e.law_name} - ماده {e.article_number}"
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return refs


def _cited_refs_from_rulings(evidences):
    refs, seen = [], set()
    for e in evidences:
        if e.source_type != "ruling":
            continue
        for ref in (e.cited_articles or []):
            if ref not in seen:
                seen.add(ref)
                refs.append(ref)
    return refs


def _find_rank(ref_norm: str, article_refs: list[str], ruling_refs: list[str]) -> str:
    article_norm = normalize_refs(article_refs)
    ruling_norm = normalize_refs(ruling_refs)

    article_list_norm = [_norm(r) for r in article_refs]
    ruling_list_norm = [_norm(r) for r in ruling_refs]

    parts = []
    if ref_norm in article_norm:
        parts.append(f"article_channel@rank{article_list_norm.index(ref_norm)+1}")
    if ref_norm in ruling_norm:
        parts.append(f"ruling_channel@rank{ruling_list_norm.index(ref_norm)+1}")
    return ", ".join(parts) if parts else "NOT FOUND"


def run():
    sample = _SAMPLE
    gold_norm = set(normalize_refs(sample["gold_articles"]))
    excluded_refs = [
        f"{a['law_name']} - ماده {a['article_number']}" for a in sample["excluded_articles"]
    ]
    excluded_norm = set(normalize_refs(excluded_refs))

    print("=" * 70)
    print(f"SINGLE RULING DEBUG (v2)  |  ruling_id={sample['ruling_id']}  |  probe_k={_PROBE_K}")
    print("=" * 70)
    print(f"\nQuery:\n  {sample['query']}\n")
    print(f"Gold articles ({len(sample['gold_articles'])}):")
    for a in sample["gold_articles"]:
        print(f"  - {a}")
    print(f"\nExcluded articles ({len(excluded_refs)}) — should NOT dominate retrieval:")
    for a, meta in zip(excluded_refs, sample["excluded_articles"]):
        print(f"  - {a}  ({meta['reason'][:60]}...)")
    print(f"\nGold routing: {sample['gold_routing']}")

    # -- run the real pipeline (router + all channels), large top_k --------
    result = search_service.search(
        sample["query"],
        feature_top_k=_PROBE_K,
        ruling_top_k=_PROBE_K,
        article_top_k=_PROBE_K,
        final_top_k=_PROBE_K,
    )

    print("\n" + "=" * 70)
    print("ACTUAL ROUTING DECISION")
    print("=" * 70)
    print(f"channels:            {result.routing.channels}")
    print(f"routing_confidence:  {result.routing.routing_confidence}")
    print(f"intent:              {result.routing.intent}")
    print(f"case_type_hint:      {result.routing.case_type_hint}")
    print(f"ambiguity_flag:      {result.routing.ambiguity_flag}")
    print(f"rewritten_query:     {result.routing.rewritten_query}")

    gold_channels = {k: v for k, v in sample["gold_routing"].items() if k != "rationale"}
    # actual_channels = {c: (c in result.routing.channels) for c in ["feature_channel", "ruling_channel", "article_channel"]}
    _CHANNEL_KEY_MAP = {
    "feature_channel": "feature",
    "ruling_channel": "ruling",
    "article_channel": "article",
    }
    actual_channels = {
        gold_key: (short_key in result.routing.channels)
        for gold_key, short_key in _CHANNEL_KEY_MAP.items()
    }
    print(f"\nGold vs actual channel selection:")
    for ch in ["feature_channel", "ruling_channel", "article_channel"]:
        gold_v = gold_channels.get(ch)
        actual_v = actual_channels.get(ch)
        match = "OK" if gold_v == actual_v else "MISMATCH"
        print(f"  {ch}: gold={gold_v}  actual={actual_v}  [{match}]")

    direct_article_refs = _article_refs_from_evidences(result.article_results)
    cited_via_rulings = _cited_refs_from_rulings(result.ruling_results)

    print("\n" + "=" * 70)
    print(f"ARTICLE CHANNEL — top {min(20, len(direct_article_refs))} of {len(direct_article_refs)}")
    print("=" * 70)
    for rank, ref in enumerate(direct_article_refs[:20], 1):
        norm = _norm(ref)
        tag = "GOLD" if norm in gold_norm else ("EXCLUDED" if norm in excluded_norm else "")
        print(f"  {rank:>2}. [{tag:>8}] {ref}")

    print("\n" + "=" * 70)
    print(f"RULING CHANNEL CITATIONS — top {min(20, len(cited_via_rulings))} of {len(cited_via_rulings)}")
    print("=" * 70)
    for rank, ref in enumerate(cited_via_rulings[:20], 1):
        norm = _norm(ref)
        tag = "GOLD" if norm in gold_norm else ("EXCLUDED" if norm in excluded_norm else "")
        print(f"  {rank:>2}. [{tag:>8}] {ref}")

    print("\n" + "=" * 70)
    print("PER-GOLD-ARTICLE TRACE")
    print("=" * 70)
    for gold_ref in sample["gold_articles"]:
        norm = _norm(gold_ref)
        found = _find_rank(norm, direct_article_refs, cited_via_rulings)
        print(f"  {gold_ref}")
        print(f"    -> {found}")

    print("\n" + "=" * 70)
    print("FALSE-POSITIVE CHECK (excluded articles that still surfaced)")
    print("=" * 70)
    any_fp = False
    for excl_ref in excluded_refs:
        norm = _norm(excl_ref)
        found = _find_rank(norm, direct_article_refs, cited_via_rulings)
        if found != "NOT FOUND":
            any_fp = True
            print(f"  ⚠ {excl_ref}")
            print(f"    -> {found}  (this was explicitly EXCLUDED in gold — may pollute Auditor input)")
    if not any_fp:
        print("  none of the excluded articles were retrieved — clean.")

    print("\n" + "=" * 70)
    print("CONFIDENCE / UNCERTAINTY VECTOR")
    print("=" * 70)
    conf = result.confidence
    print(f"score: {conf.score}   level: {conf.level}")
    print(f"feature_quality: {conf.vector.feature_quality:.3f}")
    print(f"ruling_quality:  {conf.vector.ruling_quality:.3f}")
    print(f"article_quality: {conf.vector.article_quality:.3f}")
    print(f"missing_channels: {conf.vector.missing_channels}")
    print(f"graph_support_quality: {conf.vector.graph_support_quality}")

    print()


if __name__ == "__main__":
    run()