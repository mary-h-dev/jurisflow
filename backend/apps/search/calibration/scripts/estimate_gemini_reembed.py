"""
Estimate cost and time for re-embedding 4116 articles with Gemini.
Dry run first (no actual embedding), then optionally run real.

Run:
    DJANGO_SETTINGS_VERSION=config.settings.development python -c "
    import django; django.setup()
    from apps.search.calibration.estimate_gemini_reembed import run
    run(dry_run=True)
    "
"""
import time
from core.neo4j import neo4j_client

REQUESTS_PER_MINUTE = 1500
COST_PER_REQUEST = 0.0  # AI Studio = free


def run(dry_run: bool = True):
    with neo4j_client.session() as session:
        result = session.run("MATCH (a:Article) RETURN count(a) AS c")
        total = result.single()["c"]

    minutes = total / REQUESTS_PER_MINUTE
    seconds = minutes * 60

    print(f"Total articles     : {total}")
    print(f"Rate limit         : {REQUESTS_PER_MINUTE} req/min")
    print(f"Estimated time     : {minutes:.1f} min ({seconds:.0f} sec)")
    print(f"Estimated cost     : FREE (Google AI Studio)")
    print()

    if dry_run:
        print("✓ Dry run -- no embedding done.")
        print("  Run with dry_run=False to actually re-embed.")
        return

    # اگه dry_run=False بود، واقعی اجرا میکنه
    from apps.search.embedder import _embed_gemini

    with neo4j_client.session() as session:
        articles = session.run(
            "MATCH (a:Article) RETURN id(a) AS node_id, a.content AS content, "
            "a.article_number AS num LIMIT 10"  # ← اول 10 تا تست کن
        ).data()

    print(f"Embedding {len(articles)} articles (test batch)...")
    for i, art in enumerate(articles):
        if not art["content"]:
            print(f"  [{i}] SKIP — no content")
            continue
        emb = _embed_gemini(art["content"])
        print(f"  [{i}] article {art['num']} → vector dim={len(emb)} ✓")
        time.sleep(0.05)  # کمی throttle

    print("\nTest batch done. If all good, remove LIMIT 10 for full run.")