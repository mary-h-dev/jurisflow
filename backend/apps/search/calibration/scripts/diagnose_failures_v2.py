"""
Same diagnostic as diagnose_zero_recall.py, generalized to any set of
failing ruling_ids (not just recall==0), and saves results to a
timestamped/labeled JSON file so previous diagnostic runs are never
overwritten.

Run:
    DJANGO_SETTINGS_MODULE=config.settings.development python -c "
    import django; django.setup()
    from apps.search.calibration.diagnose_failures_v2 import run
    run(output_label='failures_28_no_routing')
    "
"""

from __future__ import annotations

import json
from pathlib import Path

from apps.search.calibration.data import load_case_annotations, stratified_test_split
from apps.search.calibration.cache_utils import load_or_collect
from apps.search.calibration.calibrate import _collect_raw_outputs
from apps.search.calibration.metrics import normalize_refs
from apps.search.services import search_service
from apps.search.calibration.leakage_guard import exclude_self_ruling
from core.neo4j import neo4j_client

_ANNOTATION_PATH = "apps/search/calibration/data/annotations/case_grounded_100.json"
_RECALL_THRESHOLD = 0.7
_OUTPUT_DIR = Path(__file__).resolve().parent / "diagnostics"
_OUTPUT_DIR.mkdir(exist_ok=True)


def _check_article_in_graph(session, law_name: str, article_number: int) -> bool:
    result = session.run(
        "MATCH (l:Law {name: $law_name})-[:CONTAINS]->(a:Article {article_number: $num}) "
        "RETURN count(a) AS c",
        law_name=law_name, num=article_number,
    )
    return result.single()["c"] > 0


def _parse_gold_ref(ref: str) -> tuple[str, int]:
    persian_to_latin = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
    law_part, num_part = ref.rsplit(" - ماده ", 1)
    return law_part.strip(), int(num_part.translate(persian_to_latin))


def run(output_label: str, probe_k: int = 40):
    samples = load_case_annotations(_ANNOTATION_PATH)
    train_val, test = stratified_test_split(samples, test_fraction=0.2)
    train_val_raw = load_or_collect("train_val_140_no_routing", train_val, _collect_raw_outputs)
    test_raw = load_or_collect("test_140_no_routing", test, _collect_raw_outputs)
    all_raw = {**train_val_raw, **test_raw}
    all_samples = train_val + test

    failed_ids = [rid for rid, r in all_raw.items() if r.actual_recall < _RECALL_THRESHOLD]
    print(f"Diagnosing {len(failed_ids)} failing samples (recall < {_RECALL_THRESHOLD})...\n")

    results = {"bucket_a": [], "bucket_b": [], "bucket_other": [], "detail": []}

    with neo4j_client.session() as session:
        for rid in failed_ids:
            sample = next(s for s in all_samples if s.ruling_id == rid)

            full_result = search_service.search(
                sample.query,
                feature_top_k=probe_k, ruling_top_k=probe_k,
                article_top_k=probe_k, final_top_k=probe_k,
            )
            full_result = exclude_self_ruling(full_result, sample.ruling_id)

            article_refs_ranked = [
                f"{e.law_name} - ماده {e.article_number}"
                for e in full_result.article_results if e.law_name and e.article_number
            ]
            cited_refs_ranked = []
            for e in full_result.ruling_results:
                for ref in (e.cited_articles or []):
                    if ref not in cited_refs_ranked:
                        cited_refs_ranked.append(ref)

            gold_detail = []
            sample_bucket = None
            for gold_ref in sample.gold_articles:
                law_name, article_number = _parse_gold_ref(gold_ref)
                in_graph = _check_article_in_graph(session, law_name, article_number)

                gold_norm = next(iter(normalize_refs([gold_ref])))
                article_norm = [next(iter(normalize_refs([r]))) for r in article_refs_ranked]
                cited_norm = [next(iter(normalize_refs([r]))) for r in cited_refs_ranked]

                article_rank = article_norm.index(gold_norm) + 1 if gold_norm in article_norm else None
                cited_rank = cited_norm.index(gold_norm) + 1 if gold_norm in cited_norm else None

                gold_detail.append({
                    "article": gold_ref, "in_graph": in_graph,
                    "article_channel_rank": article_rank, "ruling_citation_rank": cited_rank,
                })

                if not in_graph:
                    sample_bucket = "A"
                elif article_rank is None and cited_rank is None and sample_bucket != "A":
                    sample_bucket = "B"

            entry = {
                "ruling_id": rid, "case_type": sample.case_type,
                "gold_articles": sample.gold_articles, "bucket": sample_bucket or "other",
                "gold_detail": gold_detail,
            }
            results["detail"].append(entry)
            if sample_bucket == "A":
                results["bucket_a"].append(rid)
            elif sample_bucket == "B":
                results["bucket_b"].append(rid)
            else:
                results["bucket_other"].append(rid)

            print(f"[{rid}] bucket={sample_bucket or 'other'}  gold={sample.gold_articles}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Bucket A (not in graph):      {len(results['bucket_a'])}  {results['bucket_a']}")
    print(f"Bucket B (in graph, low rank): {len(results['bucket_b'])}  {results['bucket_b']}")
    print(f"Bucket other:                  {len(results['bucket_other'])}  {results['bucket_other']}")

    output_path = _OUTPUT_DIR / f"{output_label}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    run(output_label="failures_default")