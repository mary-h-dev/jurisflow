"""
Isolates the source of non-determinism: calls embed_text() twice on the
EXACT same fixed string (bypassing the router/rewrite step entirely) and
checks whether the returned vectors are byte-identical. Then runs the
same raw embedding through the article_channel search twice and checks
whether the ordered result list is identical.

If the embeddings match but the search results don't -> the vector index
(HNSW, approximate) is the source of non-determinism, not the embedding
call or the router.

Run:
    DJANGO_SETTINGS_MODULE=config.settings.development python -c "
    import django; django.setup()
    from apps.search.calibration.check_determinism import run
    run()
    "
"""

from __future__ import annotations

from apps.search.embedder import embed_text
from core.neo4j import neo4j_client

_FIXED_QUERY = "زوجه به دلیل نشوز آیا حق دریافت نفقه دارد؟"
_TOP_K = 20

_ARTICLE_QUERY = """
CALL db.index.vector.queryNodes('article_embedding', $top_k, $embedding)
YIELD node AS article, score
MATCH (law:Law)-[:CONTAINS]->(article)
RETURN article.article_number AS article_number, law.name AS law_name, score
ORDER BY score DESC
LIMIT $top_k
"""


def run() -> None:
    print("=" * 70)
    print("STEP 1: embedding determinism (same text, two calls)")
    print("=" * 70)

    emb1 = embed_text(_FIXED_QUERY)
    emb2 = embed_text(_FIXED_QUERY)

    identical = emb1 == emb2
    print(f"  embedding length: {len(emb1)}")
    print(f"  identical vectors: {identical}")
    if not identical:
        diffs = sum(1 for a, b in zip(emb1, emb2) if a != b)
        print(f"  -> {diffs}/{len(emb1)} components differ. "
              f"The embedding CALL ITSELF is non-deterministic "
              f"(model/API-level, not the router).")
    else:
        print("  -> embedding call is deterministic for identical input.")

    print("\n" + "=" * 70)
    print("STEP 2: vector index (HNSW) determinism (same embedding, two searches)")
    print("=" * 70)

    with neo4j_client.session() as session:
        run1 = list(session.run(_ARTICLE_QUERY, embedding=emb1, top_k=_TOP_K))
        run2 = list(session.run(_ARTICLE_QUERY, embedding=emb1, top_k=_TOP_K))

    refs1 = [(r["law_name"], r["article_number"], round(r["score"], 6)) for r in run1]
    refs2 = [(r["law_name"], r["article_number"], round(r["score"], 6)) for r in run2]

    print(f"  run1 top-{_TOP_K}: {refs1}")
    print(f"  run2 top-{_TOP_K}: {refs2}")

    if refs1 == refs2:
        print("\n  -> HNSW search is deterministic for this identical embedding+top_k. "
              "Non-determinism must be coming from upstream (router rewrite still "
              "active somewhere, or a different embedding per call).")
    else:
        only_in_1 = set(refs1) - set(refs2)
        only_in_2 = set(refs2) - set(refs1)
        print(f"\n  -> HNSW search is NON-deterministic: same embedding, same top_k, "
              f"different result sets.")
        print(f"     only in run1: {only_in_1}")
        print(f"     only in run2: {only_in_2}")
        print("     This confirms the vector index itself (approximate search) is "
              "the source of run-to-run variance, not the router or embedding call.")


if __name__ == "__main__":
    run()