"""
Debug script: runs the 20-30 article-only annotated queries directly
through the article search channel (bypassing router/feature/ruling),
to isolate whether article retrieval itself is accurate.

Expects a JSON file at apps/search/calibration/data/article_only_queries.json
with objects like:
    {"query": "...", "gold_articles": ["قانون X - ماده Y", ...]}

Run:
    DJANGO_SETTINGS_MODULE=config.settings.development python -c "
    import django; django.setup()
    from apps.search.calibration.check_article_only_queries import run
    run()
    "
"""

import json

from apps.search.embedder import embed_text
from apps.search.services import _search_articles
from apps.search.calibration.metrics import normalize_refs, f1_score_refs
from core.neo4j import neo4j_client

_PATH = "apps/search/calibration/data/annotations/article_only_queries.json"


def run():
    with open(_PATH, encoding="utf-8") as f:
        samples = json.load(f)

    total_f1 = 0.0
    zero_count = 0

    with neo4j_client.session() as session:
        for sample in samples:
            query = sample["query"]
            gold = sample["gold_articles"]

            embedding = embed_text(query)
            results = _search_articles(session, embedding, top_k=5)

            predicted_refs = [
                f"{e.law_name} - ماده {e.article_number}"
                for e in results
                if e.law_name and e.article_number
            ]

            f1 = f1_score_refs(predicted_refs, gold)
            total_f1 += f1
            if f1 == 0.0:
                zero_count += 1

            status = "OK" if f1 > 0 else "MISS"
            print(f"[{status}] f1={f1:.2f}  query={query!r}")
            print(f"    predicted: {normalize_refs(predicted_refs)}")
            print(f"    gold:      {set(gold)}")
            print()

    print(f"\nAverage F1: {total_f1 / len(samples):.3f}")
    print(f"Zero-F1 count: {zero_count} / {len(samples)}")


if __name__ == "__main__":
    run()