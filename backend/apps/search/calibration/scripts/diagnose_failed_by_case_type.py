"""
Diagnostic for the FAILED samples (recall < 0.7) from final_calibration_report.py,
broken down by case_type (حقوقی / کیفری). Adapted from diagnose_zero_recall.py
(the earlier 34-zero-recall analysis) -- same graph-check logic, same
re-run-at-probe_k approach, extended to:
  (1) the recall<0.7 threshold instead of only recall==0
  (2) an explicit near-miss vs deep-miss split within "in graph, ranked too low"
  (3) a bucket x case_type summary table

Run:
DJANGO_SETTINGS_MODULE=config.settings.development python -c "import django; django.setup(); from apps.search.calibration.diagnose_failed_by_case_type import run; run()"
"""

from __future__ import annotations

from collections import defaultdict

from apps.search.calibration.data import load_case_annotations, stratified_test_split
from apps.search.calibration.calibrate import _collect_raw_outputs
from apps.search.calibration.cache_utils import load_or_collect
from apps.search.calibration.metrics import normalize_refs
from apps.search.services import search_service
from apps.search.calibration.leakage_guard import exclude_self_ruling
from core.neo4j import neo4j_client

_ANNOTATION_PATH = "apps/search/calibration/data/annotations/case_grounded.json"
_RECALL_THRESHOLD = 0.7
_PROD_TOP_K = 20     # feature/ruling/article/final_top_k in production
_PROBE_K = 40         # generous, to separate "not in graph" from "ranked too low"
_NEAR_MISS_RANK_CUTOFF = _PROD_TOP_K * 2  # rank <= 40 counts as "near miss"


def _check_article_in_graph(session, law_name: str, article_number: int) -> bool:
    """Bucket A check: does this article node exist at all, regardless of retrieval."""
    result = session.run(
        "MATCH (l:Law {name: $law_name})-[:CONTAINS]->(a:Article {article_number: $num}) "
        "RETURN count(a) AS c",
        law_name=law_name,
        num=article_number,
    )
    return result.single()["c"] > 0


def _parse_gold_ref(ref: str) -> tuple[str, int]:
    """'قانون مدنی - ماده 451' -> ('قانون مدنی', 451). Handles Persian digits."""
    persian_to_latin = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
    law_part, num_part = ref.rsplit(" - ماده ", 1)
    num_part = num_part.translate(persian_to_latin)
    return law_part.strip(), int(num_part)


def run():
    samples = load_case_annotations(_ANNOTATION_PATH)
    train_val, test = stratified_test_split(samples, test_fraction=0.2)
    all_samples = train_val + test

    # Cache-hit only (zero new search calls): reuse the exact same cached
    # results the final calibration report used, just to know which
    # ruling_ids failed. Do NOT call _collect_raw_outputs directly here --
    # that would silently re-run search() on all 140 samples again.
    train_val_raw = load_or_collect("train_val_140_no_routing", train_val, _collect_raw_outputs)
    test_raw = load_or_collect("test_140_no_routing", test, _collect_raw_outputs)
    raw = {**train_val_raw, **test_raw}

    failed_ids = [rid for rid, r in raw.items() if r.actual_recall < _RECALL_THRESHOLD]
    print(f"Total failed (recall < {_RECALL_THRESHOLD}): {len(failed_ids)}\n")
    print("(cache hit for the 140 recall values -- 0 new search calls for this part)\n")

    # bucket_name -> list of dicts (includes case_type for the summary table)
    buckets: dict[str, list[dict]] = defaultdict(list)

    with neo4j_client.session() as session:
        for rid in failed_ids:
            sample = next(s for s in all_samples if s.ruling_id == rid)
            raw_result = raw[rid]

            print("=" * 80)
            print(f"ruling_id: {rid}")
            print(f"case_type: {sample.case_type}")
            print(f"recall: {raw_result.actual_recall:.2f}")
            print(f"gold_articles: {sample.gold_articles}")
            print(f"fired_channels: {raw_result.queried_channels}")

            # Re-run full search (probe_k large) to inspect actual rank/score,
            # since RawSearchOutput only cached the uncertainty vector, not
            # full evidence lists.
            full_result = search_service.search(
                sample.query,
                feature_top_k=_PROBE_K,
                ruling_top_k=_PROBE_K,
                article_top_k=_PROBE_K,
                final_top_k=_PROBE_K,
            )
            full_result = exclude_self_ruling(full_result, sample.ruling_id)

            article_refs_ranked = [
                f"{e.law_name} - ماده {e.article_number}"
                for e in full_result.article_results
                if e.law_name and e.article_number
            ]
            cited_refs_ranked = []
            for e in full_result.ruling_results:
                for ref in (e.cited_articles or []):
                    if ref not in cited_refs_ranked:
                        cited_refs_ranked.append(ref)

            # per-sample worst-case bucket across its gold articles
            sample_bucket = None
            gold_detail = []

            for gold_ref in sample.gold_articles:
                law_name, article_number = _parse_gold_ref(gold_ref)
                in_graph = _check_article_in_graph(session, law_name, article_number)

                gold_norm = next(iter(normalize_refs([gold_ref])))
                article_norm = [next(iter(normalize_refs([r]))) for r in article_refs_ranked]
                cited_norm = [next(iter(normalize_refs([r]))) for r in cited_refs_ranked]

                article_rank = article_norm.index(gold_norm) + 1 if gold_norm in article_norm else None
                cited_rank = cited_norm.index(gold_norm) + 1 if gold_norm in cited_norm else None
                best_rank = min([r for r in (article_rank, cited_rank) if r is not None], default=None)

                print(f"  article: {gold_ref}")
                print(f"    in_graph: {in_graph}")
                print(f"    article_channel_rank (of {len(article_refs_ranked)}): {article_rank}")
                print(f"    ruling_citation_rank (of {len(cited_refs_ranked)}): {cited_rank}")

                gold_detail.append({
                    "ref": gold_ref, "in_graph": in_graph,
                    "article_rank": article_rank, "cited_rank": cited_rank,
                })

                # priority: A > B_deep_miss > B_near_miss (worst case wins)
                if not in_graph:
                    sample_bucket = "A_not_in_graph"
                elif best_rank is None:
                    if sample_bucket != "A_not_in_graph":
                        sample_bucket = "B_deep_miss"  # not found even at probe_k=40
                elif best_rank > _NEAR_MISS_RANK_CUTOFF:
                    if sample_bucket not in ("A_not_in_graph", "B_deep_miss"):
                        sample_bucket = "B_deep_miss"
                else:
                    # found within near-miss cutoff, but recall<0.7 at prod top_k=20
                    if sample_bucket is None:
                        sample_bucket = "B_near_miss"

            # Bucket C check: did the router even fire the right channel per gold_routing?
            gold_routing = getattr(sample, "gold_routing", None)
            if gold_routing:
                expected = {
                    "feature": gold_routing.get("feature_channel"),
                    "ruling": gold_routing.get("ruling_channel"),
                    "article": gold_routing.get("article_channel"),
                }
                actual = {c: (c in raw_result.queried_channels) for c in ["feature", "ruling", "article"]}
                mismatch = {c for c in expected if expected[c] != actual[c]}
                if mismatch:
                    print(f"  ROUTING MISMATCH: expected={expected} actual={actual}")
                    if sample_bucket is None:
                        sample_bucket = "C_routing_mismatch"

            if sample_bucket is None:
                sample_bucket = "unclassified"  # shouldn't normally happen; flag for manual look

            buckets[sample_bucket].append({
                "ruling_id": rid,
                "case_type": sample.case_type,
                "recall": raw_result.actual_recall,
                "gold_detail": gold_detail,
            })

            print()

    # ---- summary table: bucket x case_type ----
    print("\n" + "=" * 80)
    print("SUMMARY -- bucket x case_type")
    print("=" * 80)
    print(f"{'Bucket':25s} {'حقوقی':>8s} {'کیفری':>8s} {'Total':>8s}")
    bucket_order = ["A_not_in_graph", "B_deep_miss", "B_near_miss", "C_routing_mismatch", "unclassified"]
    for bucket_name in bucket_order:
        entries = buckets.get(bucket_name, [])
        civil = sum(1 for e in entries if e["case_type"] == "حقوقی")
        criminal = sum(1 for e in entries if e["case_type"] == "کیفری")
        print(f"{bucket_name:25s} {civil:8d} {criminal:8d} {len(entries):8d}")
        if entries:
            print(f"    ruling_ids: {[e['ruling_id'] for e in entries]}")


if __name__ == "__main__":
    run()