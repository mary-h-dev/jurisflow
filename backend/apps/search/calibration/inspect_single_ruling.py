"""
Debug script: single hard ruling — probes article/ruling/feature channels
separately with large top_k to distinguish embedding failure vs cutoff.

Run:
    DJANGO_SETTINGS_MODULE=config.settings.development python -c "
    import django; django.setup()
    from apps.search.calibration.inspect_single_ruling import run
    run()
    "
"""

from __future__ import annotations
from apps.search.calibration.data import load_case_annotations
from apps.search.services import search_service
from apps.search.calibration.metrics import normalize_refs

_ANNOTATION_PATH = "apps/search/calibration/data/annotations/case_grounded_100.json"
_TARGET_RULING_ID = "34630"
_PROBE_K = 30


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _norm(ref: str) -> str:
    return next(iter(normalize_refs([ref])))



def _recall_breakdown_per_channel(
    article_refs: list[str],
    ruling_refs: list[str],
    gold: list[str],
) -> dict[int, dict]:
    """
    For each k, computes recall using the union of top-k from article_channel
    AND top-k from ruling_channel — matching how the real pipeline works
    (each channel has its own independent top_k, not a single shared cutoff
    over a merged list).
    """
    gold_norm = set(normalize_refs(gold))
    out = {}
    for k in [1, 3, 5, 10, 20, _PROBE_K]:
        article_top_k = set(normalize_refs(article_refs[:k]))
        ruling_top_k  = set(normalize_refs(ruling_refs[:k]))
        union = article_top_k | ruling_top_k
        hits = gold_norm & union
        out[k] = {
            "recall": len(hits) / len(gold_norm) if gold_norm else 0.0,
            "hits":   hits,
            "missed": gold_norm - hits,
        }
    return out

def _recall_breakdown(predicted: list[str], gold: list[str]) -> dict[int, dict]:
    gold_norm = set(normalize_refs(gold))
    out = {}
    for k in [1, 3, 5, 10, 20, _PROBE_K]:
        hits = gold_norm & set(normalize_refs(predicted[:k]))
        out[k] = {
            "recall": len(hits) / len(gold_norm) if gold_norm else 0.0,
            "hits":   hits,
            "missed": gold_norm - hits,
        }
    return out


def _print_ranked_list(label: str, refs: list[str], gold_norm: set[str]) -> None:
    print(f"\n  [{label}]  ({len(refs)} results)")
    for rank, ref in enumerate(refs, 1):
        norm = _norm(ref)
        hit  = "✓ GOLD" if norm in gold_norm else "      "
        print(f"    {rank:>2}. [{hit}] {ref}")


# ---------------------------------------------------------------------------
# article refs from each channel separately
# ---------------------------------------------------------------------------

def _article_refs_from_evidences(evidences):
    refs, seen = [], set()
    for e in evidences:
        if e.source_type != "article" or not e.law_name or not e.article_number:
            continue
        ref = f"{e.law_name} - ماده {e.article_number}"
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return refs


def _cited_refs_from_rulings(evidences):
    refs, seen = [], set()
    for e in evidences:
        if e.source_type != "ruling":
            continue
        for ref in (e.cited_articles or []):
            if ref not in seen:
                seen.add(ref)
                refs.append(ref)
    return refs


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run():
    samples = load_case_annotations(_ANNOTATION_PATH)
    target  = next((s for s in samples if s.ruling_id == _TARGET_RULING_ID), None)

    if target is None:
        print(f"[ERROR] ruling_id={_TARGET_RULING_ID} not found.")
        return

    gold_norm = set(normalize_refs(target.gold_articles))

    print("=" * 70)
    print(f"SINGLE RULING DEBUG  |  ruling_id={target.ruling_id}  |  probe_k={_PROBE_K}")
    print("=" * 70)
    print(f"\nQuery:\n  {target.query}\n")
    print(f"Gold ({len(target.gold_articles)} articles):")
    for a in target.gold_articles:
        print(f"  - {a}")

    # -- run search with large k on all three channels -----------------------
    result = search_service.search(
        target.query,
        feature_top_k = _PROBE_K,
        ruling_top_k  = _PROBE_K,
        article_top_k = _PROBE_K,
        final_top_k   = _PROBE_K,
    )

    # -- article channel: direct embedding hits ------------------------------
    direct_article_refs = _article_refs_from_evidences(result.article_results)
    # -- ruling channel: cited_articles from graph edges ---------------------
    cited_via_rulings   = _cited_refs_from_rulings(result.ruling_results)


    # -- combined (rank-merged, NOT simple concatenation): interleave by
    #    relative rank so an item strong in either channel keeps its
    #    effective position, instead of ruling_channel always starting
    #    after all 30 article_channel slots are exhausted --
    combined_refs, seen = [], set()
    max_len = max(len(direct_article_refs), len(cited_via_rulings))
    for i in range(max_len):
        if i < len(direct_article_refs):
            ref = direct_article_refs[i]
            if ref not in seen:
                seen.add(ref)
                combined_refs.append(ref)
        if i < len(cited_via_rulings):
            ref = cited_via_rulings[i]
            if ref not in seen:
                seen.add(ref)
                combined_refs.append(ref)


    print("\n" + "=" * 70)
    print("CHANNEL BREAKDOWN")
    print("=" * 70)

    _print_ranked_list("article_channel (embedding)", direct_article_refs, gold_norm)
    _print_ranked_list("ruling_channel  (cited_articles graph edges)", cited_via_rulings, gold_norm)
    _print_ranked_list("combined (article + ruling citations, pre-RRF)", combined_refs, gold_norm)

    # -- recall breakdown on COMBINED (not the RRF-fused final output) ------
    # NOTE: result.article_refs (the fused output) mixes feature/ruling/
    # article evidences via RRF and truncates to final_top_k across ALL
    # channels together — so an article present in the article channel's
    # own top-30 can still be absent from the fused output simply because
    # it lost the cross-channel ranking competition. That would misdiagnose
    # a "cutoff by fusion" issue as an "embedding failure". Using the
    # combined (unfused) list isolates the actual retrieval question.
    print("\n" + "=" * 70)
    print("RECALL BREAKDOWN  (article_channel + ruling citations, unfused)")
    print("=" * 70)
    # breakdown = _recall_breakdown(combined_refs, target.gold_articles)
    breakdown = _recall_breakdown_per_channel(direct_article_refs, cited_via_rulings, target.gold_articles)
    for k, info in breakdown.items():
        bar = "█" * int(info["recall"] * 20)
        print(f"  recall@{k:<3} = {info['recall']:.2f}  {bar}")

    missed_at_30 = breakdown[_PROBE_K]["missed"]

    # -- diagnosis -----------------------------------------------------------
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    recall_at_5  = breakdown[5]["recall"]
    recall_at_30 = breakdown[_PROBE_K]["recall"]

    print("\n  Per-article source tracing:")
    for gold_ref in target.gold_articles:
        norm       = _norm(gold_ref)
        in_article = norm in set(normalize_refs(direct_article_refs))
        in_ruling  = norm in set(normalize_refs(cited_via_rulings))
        sources    = []
        if in_article: sources.append("article_channel")
        if in_ruling:  sources.append("ruling_channel")
        status = ", ".join(sources) if sources else "NOT FOUND in top-30 of either channel"
        print(f"    {gold_ref}")
        print(f"      -> {status}")

    print()
    if recall_at_30 == 0.0:
        print("  EMBEDDING FAILURE")
        print("     هیچ gold article‌ای در top-30 هیچ channel‌ی نیست.")
        print("     -> مشکل از embedding، law_name mismatch، یا قانون لود نشده در گراف است.")
    elif recall_at_30 > recall_at_5:
        print("  TOP_K CUTOFF")
        print(f"     recall@5={recall_at_5:.2f}  ->  recall@30={recall_at_30:.2f}")
        print("     -> مواد در گراف هستند ولی cutoff پیش‌فرض (top_k=5) آن‌ها را حذف می‌کند.")
        print("     -> گزینه‌ها: top_k بالاتر، re-ranker، یا query decomposition.")
    else:
        print("  recall@5 ~= recall@30")
        print("     بالا بردن top_k کمکی نمی‌کند.")
        print("     -> مشکل از annotation، query mismatch، یا قانون لود نشده است.")

    if missed_at_30:
        print(f"\n  مواد گم‌شده حتی در top-{_PROBE_K} هر دو کانال:")
        for m in missed_at_30:
            print(f"    - {m}")
        print("  -> بررسی کن law_name در گراف دقیقاً با annotation match می‌کند یا اصلاً لود شده.")

    print()


if __name__ == "__main__":
    run()