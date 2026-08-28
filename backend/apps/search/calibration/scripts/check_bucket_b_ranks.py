"""
For Bucket B samples (gold article exists in graph but wasn't found
within production top_k), finds the TRUE rank using a much larger
probe_k (100), to distinguish "just needs bigger top_k" from "ranking
is fundamentally broken for this query".

Run:
    DJANGO_SETTINGS_MODULE=config.settings.development python -c "
    import django; django.setup()
    from apps.search.calibration.check_bucket_b_ranks import run
    run()
    "
"""

from __future__ import annotations

from apps.search.calibration.data import load_case_annotations, stratified_test_split
from apps.search.calibration.metrics import normalize_refs
from apps.search.services import search_service
from apps.search.calibration.leakage_guard import exclude_self_ruling

_ANNOTATION_PATH = "apps/search/calibration/data/annotations/case_grounded.json"
_PROBE_K = 100

_BUCKET_B_IDS = [
    "34894", "37607", "37587", "36157", "36284", "40963", "35599", "36311",
    "25154", "29030", "36525", "36572", "36913", "36199", "37581", "36900",
    "25094", "38529", "37656", "38695", "38404", "38347", "38338", "34703",
    "38363", "43144", "37673", "43816", "38790", "43602", "43586",
]


def run():
    samples = load_case_annotations(_ANNOTATION_PATH)

    found_within_100 = []
    not_found_within_100 = []

    for rid in _BUCKET_B_IDS:
        sample = next((s for s in samples if s.ruling_id == rid), None)
        if sample is None:
            print(f"{rid}: not found in annotation file, skipping")
            continue

        result = search_service.search(
            sample.query,
            feature_top_k=_PROBE_K,
            ruling_top_k=_PROBE_K,
            article_top_k=_PROBE_K,
            final_top_k=_PROBE_K,
        )
        result = exclude_self_ruling(result, sample.ruling_id)

        article_refs_ranked = [
            f"{e.law_name} - ماده {e.article_number}"
            for e in result.article_results
            if e.law_name and e.article_number
        ]
        cited_refs_ranked = []
        for e in result.ruling_results:
            for ref in (e.cited_articles or []):
                if ref not in cited_refs_ranked:
                    cited_refs_ranked.append(ref)

        article_norm = [next(iter(normalize_refs([r]))) for r in article_refs_ranked]
        cited_norm = [next(iter(normalize_refs([r]))) for r in cited_refs_ranked]

        best_rank = None
        best_source = None
        for gold_ref in sample.gold_articles:
            gold_norm = next(iter(normalize_refs([gold_ref])))
            if gold_norm in article_norm:
                r = article_norm.index(gold_norm) + 1
                if best_rank is None or r < best_rank:
                    best_rank, best_source = r, "article_channel"
            if gold_norm in cited_norm:
                r = cited_norm.index(gold_norm) + 1
                if best_rank is None or r < best_rank:
                    best_rank, best_source = r, "ruling_channel"

        if best_rank is not None:
            found_within_100.append((rid, best_rank, best_source))
            print(f"{rid}: FOUND at rank {best_rank} ({best_source})")
        else:
            not_found_within_100.append(rid)
            print(f"{rid}: NOT FOUND even within top-{_PROBE_K}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Found within top-{_PROBE_K}: {len(found_within_100)}")
    for rid, rank, source in sorted(found_within_100, key=lambda x: x[1]):
        print(f"  {rid}: rank={rank} ({source})")
    print(f"\nNOT found even within top-{_PROBE_K}: {len(not_found_within_100)}")
    print(f"  {not_found_within_100}")


if __name__ == "__main__":
    run()