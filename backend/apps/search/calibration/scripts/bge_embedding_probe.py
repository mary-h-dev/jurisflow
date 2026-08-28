"""
Same as gemini_embedding_probe.py but uses the local bge-m3 (Ollama)
model instead of Gemini -- so we can compare similarity scores under
the same model that actually powers the corpus vectors in Neo4j.

This is the FAIR comparison: corpus was embedded with bge-m3, so
sim(query_bge, gold_article_bge) reflects what the vector index
actually sees.

Run:
DJANGO_SETTINGS_MODULE=config.settings.development python -c "import django; django.setup(); from apps.search.calibration.bge_embedding_probe import run; run()"
"""

from __future__ import annotations


from apps.search.embedder import _embed_ollama as _embed

from apps.search.calibration.data import load_case_annotations, stratified_test_split
from apps.search.calibration.calibrate import _collect_raw_outputs
from apps.search.calibration.cache_utils import load_or_collect
from core.neo4j import neo4j_client

_ANNOTATION_PATH = "apps/search/calibration/data/annotations/case_grounded.json"
_RECALL_THRESHOLD = 0.7


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b)


def _get_article_text(session, law_name: str, article_number: int) -> str | None:
    result = session.run(
        "MATCH (l:Law {name: $law_name})-[:CONTAINS]->(a:Article {article_number: $num}) "
        "RETURN a.content AS text LIMIT 1",
        law_name=law_name,
        num=article_number,
    )
    rec = result.single()
    return rec["text"] if rec else None


def _parse_gold_ref(ref: str) -> tuple[str, int]:
    persian_to_latin = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
    law_part, num_part = ref.rsplit(" - ماده ", 1)
    num_part = num_part.translate(persian_to_latin)
    return law_part.strip(), int(num_part)


def run(sample_limit: int = 8):
    samples = load_case_annotations(_ANNOTATION_PATH)
    train_val, test = stratified_test_split(samples, test_fraction=0.2)

    train_val_raw = load_or_collect("train_val_140_no_routing", train_val, _collect_raw_outputs)
    test_raw = load_or_collect("test_140_no_routing", test, _collect_raw_outputs)
    raw = {**train_val_raw, **test_raw}

    failed_samples = [
        s for s in (train_val + test)
        if raw[s.ruling_id].actual_recall < _RECALL_THRESHOLD
    ]

    checked = 0
    with neo4j_client.session() as session:
        for sample in failed_samples:
            if checked >= sample_limit:
                break

            none_rank_articles = [{"ref": ref} for ref in sample.gold_articles]
            checked += 1

            print("=" * 70)
            print(f"ruling_id: {sample.ruling_id}  case_type: {sample.case_type}")
            print(f"query: {sample.query[:100]}")

            query_emb = _embed(sample.query)

            for g in none_rank_articles:
                law_name, article_number = _parse_gold_ref(g["ref"])
                article_text = _get_article_text(session, law_name, article_number)
                if not article_text:
                    print(f"  [{g['ref']}] -- article text not found, skipping")
                    continue
                gold_emb = _embed(article_text)
                sim_gold = _cosine(query_emb, gold_emb)
                print(f"  [{g['ref']}] BGE-M3 sim(query, gold_article) = {sim_gold:.4f}")

            print()


if __name__ == "__main__":
    run()