"""
Isolates a retrieval regression from the Auditor layer entirely: prints
the router's decision and each channel's raw result count for a fixed
query, so a change in article_refs count can be traced to routing
(channels skipped) vs. per-channel top_k vs. citation attachment --
before ever touching the Auditor/LLM verification step.

Run:
DJANGO_SETTINGS_MODULE=config.settings.development python -c "import django; django.setup(); from apps.search.calibration.scripts.check_retrieval_counts import run; run()"
"""

from __future__ import annotations

from apps.search.services import search_service

_QUERY = "من به خاطر مزاحمت تلفنی به شش ماه حبس محکوم شدم. حداکثر مجازات قانونی این جرم همون شش ماهه. آیا دادگاه باید به جای حبس، مجازات جایگزین تعیین می‌کرد؟ با این وضع می‌تونم درخواست اعاده دادرسی بدم؟"


def run() -> None:
    result = search_service.search(_QUERY)

    print("=" * 70)
    print("ROUTING")
    print("=" * 70)
    print(f"  rewritten_query: {result.routing.rewritten_query!r}")
    print(f"  channels: {result.routing.channels}")
    print(f"  routing_confidence: {result.routing.routing_confidence}")
    print(f"  ambiguity_flag: {result.routing.ambiguity_flag}")

    print("\n" + "=" * 70)
    print("RAW CHANNEL COUNTS (before ranking/capping)")
    print("=" * 70)
    print(f"  feature_results: {len(result.feature_results)}")
    print(f"  ruling_results:  {len(result.ruling_results)}")
    print(f"  article_results: {len(result.article_results)}")

    citing_rulings = [e for e in result.ruling_results if e.cited_articles]
    total_citations = sum(len(e.cited_articles or []) for e in result.ruling_results)
    print(f"\n  rulings WITH cited_articles: {len(citing_rulings)} / {len(result.ruling_results)}")
    print(f"  total citation entries across all rulings: {total_citations}")
    if result.ruling_results and not citing_rulings:
        print("  ⚠ every retrieved ruling has an EMPTY cited_articles list -- "
              "this is why no citation-tier article_refs are appearing. "
              "Check the Cypher query building ruling Evidence objects "
              "(cited_articles collection) in apps/search/services.py.")

    print(f"\n  final article_refs (union, dedup): {len(result.article_refs)}")
    print(f"  (= article_results direct hits + citations from ruling_results)")


if __name__ == "__main__":
    run()