"""
Quick diagnostic (NOT a re-engineering): for samples where the current
article_channel completely missed the gold article (rank=None even at
probe_k=40), check whether embedding (query, gold_article_text) with
Gemini scores meaningfully higher similarity than you'd expect, as a
directional signal about whether the facts-vs-statute-text gap is a
matter of embedding MODEL quality vs. a structural mismatch that would
persist under any model.

IMPORTANT -- why this does NOT reuse search_service.search() directly:
the article corpus is presumably pre-embedded (with the DEV/Ollama model,
per the "[embed_text]" print lines seen in earlier probe output) and
stored in Neo4j. Swapping only the query's embedding function to Gemini
and reusing the existing vector search would compare a Gemini vector
against Ollama-embedded corpus vectors -- different embedding spaces,
mathematically meaningless similarity scores. So instead, this script
embeds BOTH the query and each candidate article's raw text with the
SAME model (Gemini) directly, side-stepping the stored/indexed corpus
vectors entirely. This tells you about relative similarity, not an
actual re-ranked position among the full corpus.

Timebox this to ~30-45 min. Two outcomes, both useful:
  - Gemini sim(query, gold) is clearly higher, relative to sim(query,
    wrongly-top-ranked-article) than you'd expect from the current
    model's behavior -> embedding MODEL quality is plausibly a factor.
    Follow-up (re-embedding the whole corpus with Gemini) is a bigger
    task, likely post-JURIX.
  - No meaningful difference -> supports the "structural gap between
    facts-phrased queries and formal statute text" explanation --
    report as a finding/limitation, not something to fix now.

Run:
DJANGO_SETTINGS_MODULE=config.settings.development python -c "import django; django.setup(); from apps.search.calibration.gemini_embedding_probe import run; run()"
"""

from __future__ import annotations

# TODO: point this at the real module path where embed_text / _embed_gemini
# actually live (the file you pasted -- confirm its location, e.g.
# apps/search/embeddings.py or similar).
from apps.search.embedder import _embed_gemini as _embed

from apps.search.calibration.data import load_case_annotations, stratified_test_split
from apps.search.calibration.calibrate import _collect_raw_outputs
from apps.search.calibration.cache_utils import load_or_collect
from core.neo4j import neo4j_client

_ANNOTATION_PATH = "apps/search/calibration/data/annotations/case_grounded.json"
_RECALL_THRESHOLD = 0.7
_PROBE_CACHE_KEY = "failed_bucket_probe_k40_v1"  # reuse the already-cached probe


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b)


def _get_article_text(session, law_name: str, article_number: int) -> str | None:
    # TODO: confirm the property name holding the article's full text
    # (guessing `text`; could be `content`, `body`, etc.)

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
    """sample_limit: how many None-rank cases to probe -- keep small,
    this is a timeboxed sanity check, not a full re-eval."""
    samples = load_case_annotations(_ANNOTATION_PATH)
    train_val, test = stratified_test_split(samples, test_fraction=0.2)
    all_samples = {s.ruling_id: s for s in (train_val + test)}

    train_val_raw = load_or_collect("train_val_140_no_routing", train_val, _collect_raw_outputs)
    test_raw = load_or_collect("test_140_no_routing", test, _collect_raw_outputs)
    raw = {**train_val_raw, **test_raw}

    failed_samples = [s for s in (train_val + test) if raw[s.ruling_id].actual_recall < _RECALL_THRESHOLD]

    checked = 0
    with neo4j_client.session() as session:
        for sample in failed_samples:
            if checked >= sample_limit:
                break

            # no probe cache needed -- just test all gold articles of this
            # failed sample directly (free: only uses the existing 140 cache)
            none_rank_articles = [{"ref": ref} for ref in sample.gold_articles]

            checked += 1
            print("=" * 70)
            print(f"ruling_id: {sample.ruling_id}  case_type: {sample.case_type}")
            print(f"query: {sample.query[:100]}")

            query_emb = _embed(sample.query)

            # For context: what did the CURRENT (Ollama) model rank #1 instead?
            # This is imperfect (top-1 text wasn't cached, only the ref string
            # in the earlier probe run) -- if you want this comparison, add the
            # top-ranked ref's text lookup here too via _get_article_text().
            for g in none_rank_articles:
                law_name, article_number = _parse_gold_ref(g["ref"])
                article_text = _get_article_text(session, law_name, article_number)
                if not article_text:
                    print(f"  [{g['ref']}] -- article text not found, skipping")
                    continue
                gold_emb = _embed(article_text)
                sim_gold = _cosine(query_emb, gold_emb)
                print(f"  [{g['ref']}] Gemini sim(query, gold_article) = {sim_gold:.4f}")

            print()


if __name__ == "__main__":
    run()