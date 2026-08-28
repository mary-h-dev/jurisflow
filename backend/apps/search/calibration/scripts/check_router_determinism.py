"""
Runs search_service.search() on the EXACT same query twice (the real
pipeline entry point, including router) and compares:
  1. routing.rewritten_query -- did the router produce different text
     each time for identical input?
  2. article_refs -- did the final result differ?

If (1) differs between runs -> the router's rewrite call is still the
source of non-determinism (check its temperature setting, or that it's
actually bypassed if you intended to remove it).
If (1) is identical but (2) still differs -> look elsewhere (e.g. an
LLM call inside routing beyond the rewrite itself, such as channel
selection).

Run:
    DJANGO_SETTINGS_MODULE=config.settings.development python -c "
    import django; django.setup()
    from apps.search.calibration.check_router_determinism import run
    run()
    "
"""

from __future__ import annotations

from apps.search.services import search_service

_FIXED_QUERY = "من به خاطر مزاحمت تلفنی به شش ماه حبس محکوم شدم. حداکثر مجازات قانونی این جرم همون شش ماهه. آیا دادگاه باید به جای حبس، مجازات جایگزین تعیین می‌کرد؟ با این وضع می‌تونم درخواست اعاده دادرسی بدم؟"


def run(n_runs: int = 4) -> None:
    print("=" * 70)
    print(f"Running search_service.search() {n_runs}x on the identical query")
    print("=" * 70)

    results = [search_service.search(_FIXED_QUERY) for _ in range(n_runs)]

    rewrites = [r.routing.rewritten_query for r in results]
    channel_sets = [r.routing.channels for r in results]
    ref_sets = [r.article_refs for r in results]

    for i, (rw, ch, refs) in enumerate(zip(rewrites, channel_sets, ref_sets), start=1):
        print(f"\nrun {i}: rewritten_query={rw!r}")
        print(f"         channels={ch}")
        print(f"         article_refs count={len(refs)}")

    rewrite_all_same = len(set(rewrites)) == 1
    channels_all_same = len({tuple(c) for c in channel_sets}) == 1
    refs_all_same = len({tuple(r) for r in ref_sets}) == 1

    print(f"\nrewritten_query identical across all {n_runs} runs: {rewrite_all_same}")
    print(f"channels identical across all {n_runs} runs: {channels_all_same}")
    print(f"article_refs identical across all {n_runs} runs: {refs_all_same}")

    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    if not rewrite_all_same:
        print("  The router's rewrite call is non-deterministic (varied across "
              f"{n_runs} identical calls) -- this is your source of variance. "
              "Check: is temperature set to 0? Is there retry/fallback logic "
              "that sometimes skips the rewrite and uses the raw query instead?")
    elif not refs_all_same:
        print("  rewritten_query was identical every time, but article_refs still "
              "varied. The rewrite is NOT the cause. Something else in routing "
              "(e.g. channel selection using an LLM, or a second embedding call "
              "with its own variance) must differ between calls.")
    else:
        print(f"  Fully deterministic across {n_runs} runs for this exact query. "
              "If non-determinism still shows up occasionally elsewhere, it's "
              "likely intermittent (API load / retry fallback) rather than "
              "structural -- worth logging routing.rewritten_query in production "
              "runs so you can catch it when it happens.")


if __name__ == "__main__":
    run()