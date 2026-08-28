"""
Diagnostic for the 34 zero-recall samples: for each gold article, checks
(1) whether it exists in the graph at all, (2) if found by article_channel
or ruling_channel citations, what rank/score it got, (3) which channels
the router actually fired.

Run:
DJANGO_SETTINGS_MODULE=config.settings.development python -c "import django; django.setup(); from apps.search.calibration.diagnose_zero_recall import run; run()"
"""

from __future__ import annotations

from apps.search.calibration.data import load_case_annotations, stratified_test_split
from apps.search.calibration.calibrate import _collect_raw_outputs
from apps.search.calibration.metrics import normalize_refs
from apps.search.services import search_service
from apps.search.calibration.leakage_guard import exclude_self_ruling
from core.neo4j import neo4j_client

_ANNOTATION_PATH = "apps/search/calibration/data/annotations/case_grounded.json"
_PROBE_K = 40  # generous, to separate "not in graph" from "ranked too low"


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
    raw = _collect_raw_outputs(train_val)

    zero_ids = [rid for rid, r in raw.items() if r.actual_recall == 0.0]
    print(f"Total zero-recall: {len(zero_ids)}\n")

    bucket_a, bucket_b, bucket_c = [], [], []

    with neo4j_client.session() as session:
        for rid in zero_ids:
            sample = next(s for s in train_val if s.ruling_id == rid)
            raw_result = raw[rid]

            print("=" * 80)
            print(f"ruling_id: {rid}")
            print(f"case_type: {sample.case_type}")
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

            sample_bucket = None

            for gold_ref in sample.gold_articles:
                law_name, article_number = _parse_gold_ref(gold_ref)
                in_graph = _check_article_in_graph(session, law_name, article_number)

                gold_norm = next(iter(normalize_refs([gold_ref])))
                article_norm = [next(iter(normalize_refs([r]))) for r in article_refs_ranked]
                cited_norm = [next(iter(normalize_refs([r]))) for r in cited_refs_ranked]

                article_rank = article_norm.index(gold_norm) + 1 if gold_norm in article_norm else None
                cited_rank = cited_norm.index(gold_norm) + 1 if gold_norm in cited_norm else None

                print(f"  article: {gold_ref}")
                print(f"    in_graph: {in_graph}")
                print(f"    article_channel_rank (of {len(article_refs_ranked)}): {article_rank}")
                print(f"    ruling_citation_rank (of {len(cited_refs_ranked)}): {cited_rank}")

                if not in_graph:
                    sample_bucket = "A"
                elif article_rank is None and cited_rank is None:
                    if sample_bucket != "A":
                        sample_bucket = "B"
                else:
                    # found somewhere within probe_k=40, but recall_at_k said 0
                    # -> means production top_k (used in _collect_raw_outputs)
                    # is smaller than where it actually ranked
                    if sample_bucket not in ("A", "B"):
                        sample_bucket = "B_rank_too_low_for_prod_topk"

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
                        sample_bucket = "C"

            if sample_bucket == "A":
                bucket_a.append(rid)
            elif sample_bucket and sample_bucket.startswith("B"):
                bucket_b.append((rid, sample_bucket))
            elif sample_bucket == "C":
                bucket_c.append(rid)

            print()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Bucket A (not in graph):              {len(bucket_a)}  {bucket_a}")
    print(f"Bucket B (in graph, ranked too low):   {len(bucket_b)}  {bucket_b}")
    print(f"Bucket C (routing mismatch):            {len(bucket_c)}  {bucket_c}")


if __name__ == "__main__":
    run()

