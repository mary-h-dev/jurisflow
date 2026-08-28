"""
Prints a full, real search_service.search() output for a query NOT from
annotation — so an AI building the Auditor can see actual evidence shape
and quality, not the clean annotation format.

Run:
    DJANGO_SETTINGS_MODULE=config.settings.development python -c "
    import django; django.setup()
    from apps.search.calibration.sample_search_output import run
    run()
    "
"""

from apps.search.services import search_service

_QUERY = "زوجه به دلیل نشوز آیا حق نفقه دارد؟"


def run():
    result = search_service.search(_QUERY)

    print("=" * 70)
    print("QUERY")
    print("=" * 70)
    print(result.query)

    print("\n" + "=" * 70)
    print("ROUTING")
    print("=" * 70)
    print(f"rewritten_query:     {result.routing.rewritten_query}")
    print(f"channels:            {result.routing.channels}")
    print(f"routing_confidence:  {result.routing.routing_confidence}")
    print(f"intent:              {result.routing.intent}")
    print(f"case_type_hint:      {result.routing.case_type_hint}")
    print(f"ambiguity_flag:      {result.routing.ambiguity_flag}")

    print("\n" + "=" * 70)
    print("CONFIDENCE")
    print("=" * 70)
    print(f"score: {result.confidence.score}")
    print(f"level: {result.confidence.level}")
    print(f"note:  {result.confidence.note}")
    print(f"feature_quality: {result.confidence.vector.feature_quality:.3f}")
    print(f"ruling_quality:  {result.confidence.vector.ruling_quality:.3f}")
    print(f"article_quality: {result.confidence.vector.article_quality:.3f}")
    print(f"missing_channels: {result.confidence.vector.missing_channels}")
    print(f"graph_support_quality: {result.confidence.vector.graph_support_quality}")

    print("\n" + "=" * 70)
    print(f"FEATURE EVIDENCES ({len(result.feature_results)})")
    print("=" * 70)
    for e in result.feature_results:
        print(f"  [{e.feature_category}] {e.feature_value!r}  score={e.score:.3f}")
        print(f"    text: {e.text[:150]}")
        print(f"    ruling_id: {e.ruling_id}  confidence: {e.confidence}")

    print("\n" + "=" * 70)
    print(f"RULING EVIDENCES ({len(result.ruling_results)})")
    print("=" * 70)
    for e in result.ruling_results:
        print(f"  ruling_id={e.ruling_id}  score={e.score:.3f}")
        print(f"    text: {e.text[:200]}")
        print(f"    cited_articles: {e.cited_articles}")

    print("\n" + "=" * 70)
    print(f"ARTICLE EVIDENCES ({len(result.article_results)})")
    print("=" * 70)
    for e in result.article_results:
        print(f"  {e.law_name} - ماده {e.article_number}  score={e.score:.3f}")
        print(f"    text: {e.text[:200]}")

    print("\n" + "=" * 70)
    print(f"FUSED EVIDENCES / final output ({len(result.evidences)})")
    print("=" * 70)
    for e in result.evidences:
        print(f"  [{e.source_type}] rrf_score={e.rrf_score:.4f}")
        print(f"    text: {e.text[:150]}")

    print("\n" + "=" * 70)
    print("SUMMARY FIELDS")
    print("=" * 70)
    print(f"ruling_ids:   {result.ruling_ids}")
    print(f"article_refs: {result.article_refs}")


if __name__ == "__main__":
    run()