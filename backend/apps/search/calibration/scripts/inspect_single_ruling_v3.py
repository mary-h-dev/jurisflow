"""
Post-ablation single-ruling diagnostic. Loads the sample directly from
the annotation file (no inline copy-paste -- avoids drift from the real
data), and reflects the current architecture:
- No channel gating: all three channels are always queried. The
  "Gold vs actual channel selection" comparison from v2 is REMOVED --
  gold_routing.channel booleans are no longer a meaningful comparison
  since gating was ablated away (see apps/search/calibration/ablations/).
- graph_support is always None now (disabled in services.py).
- routing.case_type_hint is shown as metadata only, not compared against
  anything (it doesn't filter retrieval).

Run:
DJANGO_SETTINGS_MODULE=config.settings.development python -c "import django; django.setup(); from apps.search.calibration.scripts.inspect_single_ruling_v3 import run; run(ruling_id='10493')"
"""

from __future__ import annotations

from apps.search.calibration.data import load_case_annotations
from apps.search.services import search_service
from apps.search.calibration.metrics import normalize_refs
from apps.search.calibration.leakage_guard import exclude_self_ruling

_ANNOTATION_PATH = "apps/search/calibration/data/annotations/case_grounded.json"
_PROBE_K = 30


def _norm(ref: str) -> str:
    return next(iter(normalize_refs([ref])))


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


def _find_rank(ref_norm: str, article_refs: list[str], ruling_refs: list[str]) -> str:
    article_list_norm = [_norm(r) for r in article_refs]
    ruling_list_norm = [_norm(r) for r in ruling_refs]

    parts = []
    if ref_norm in article_list_norm:
        parts.append(f"article_channel@rank{article_list_norm.index(ref_norm)+1}")
    if ref_norm in ruling_list_norm:
        parts.append(f"ruling_channel@rank{ruling_list_norm.index(ref_norm)+1}")
    return ", ".join(parts) if parts else "NOT FOUND"


def run(ruling_id: str):
    samples = load_case_annotations(_ANNOTATION_PATH)
    sample = next((s for s in samples if s.ruling_id == ruling_id), None)
    if sample is None:
        print(f"[ERROR] ruling_id={ruling_id} not found in {_ANNOTATION_PATH}")
        return

    gold_norm = set(normalize_refs(sample.gold_articles))
    excluded_refs = [
        f"{a['law_name']} - ماده {a['article_number']}"
        for a in getattr(sample, "excluded_articles", []) or []
    ]
    excluded_norm = set(normalize_refs(excluded_refs)) if excluded_refs else set()

    print("=" * 70)
    print(f"SINGLE RULING DEBUG (v3, post-ablation)  |  ruling_id={sample.ruling_id}  |  probe_k={_PROBE_K}")
    print("=" * 70)
    print(f"\nQuery:\n  {sample.query}\n")
    print(f"Gold articles ({len(sample.gold_articles)}):")
    for a in sample.gold_articles:
        print(f"  - {a}")
    if excluded_refs:
        print(f"\nExcluded articles ({len(excluded_refs)}) — should NOT dominate retrieval:")
        for a in excluded_refs:
            print(f"  - {a}")

    # -- run the real pipeline: router (rewrite only, no gating) + all channels --
    result = search_service.search(
        sample.query,
        feature_top_k=_PROBE_K,
        ruling_top_k=_PROBE_K,
        article_top_k=_PROBE_K,
        final_top_k=_PROBE_K,
    )
    result = exclude_self_ruling(result, sample.ruling_id)

    print("\n" + "=" * 70)
    print("ROUTER OUTPUT (metadata only -- does not filter retrieval anymore)")
    print("=" * 70)
    print(f"rewritten_query:     {result.routing.rewritten_query}")
    print(f"case_type_hint:      {result.routing.case_type_hint}")
    print(f"routing_confidence:  {result.routing.routing_confidence}")
    print(f"intent:              {result.routing.intent}")
    print(f"ambiguity_flag:      {result.routing.ambiguity_flag}")
    print(f"(channels field no longer gates anything -- all 3 channels always queried)")

    direct_article_refs = _article_refs_from_evidences(result.article_results)
    cited_via_rulings = _cited_refs_from_rulings(result.ruling_results)

    print("\n" + "=" * 70)
    print(f"ARTICLE CHANNEL — top {min(20, len(direct_article_refs))} of {len(direct_article_refs)}")
    print("=" * 70)
    for rank, ref in enumerate(direct_article_refs[:20], 1):
        norm = _norm(ref)
        tag = "GOLD" if norm in gold_norm else ("EXCLUDED" if norm in excluded_norm else "")
        print(f"  {rank:>2}. [{tag:>8}] {ref}")

    print("\n" + "=" * 70)
    print(f"RULING CHANNEL CITATIONS — top {min(20, len(cited_via_rulings))} of {len(cited_via_rulings)}")
    print("=" * 70)
    for rank, ref in enumerate(cited_via_rulings[:20], 1):
        norm = _norm(ref)
        tag = "GOLD" if norm in gold_norm else ("EXCLUDED" if norm in excluded_norm else "")
        print(f"  {rank:>2}. [{tag:>8}] {ref}")

    print("\n" + "=" * 70)
    print("PER-GOLD-ARTICLE TRACE")
    print("=" * 70)
    for gold_ref in sample.gold_articles:
        norm = _norm(gold_ref)
        found = _find_rank(norm, direct_article_refs, cited_via_rulings)
        print(f"  {gold_ref}")
        print(f"    -> {found}")

    if excluded_refs:
        print("\n" + "=" * 70)
        print("FALSE-POSITIVE CHECK (excluded articles that still surfaced)")
        print("=" * 70)
        any_fp = False
        for excl_ref in excluded_refs:
            norm = _norm(excl_ref)
            found = _find_rank(norm, direct_article_refs, cited_via_rulings)
            if found != "NOT FOUND":
                any_fp = True
                print(f"  WARNING: {excl_ref}")
                print(f"    -> {found}  (explicitly EXCLUDED in gold -- may pollute Auditor input)")
        if not any_fp:
            print("  none of the excluded articles were retrieved -- clean.")

    print("\n" + "=" * 70)
    print("CONFIDENCE / UNCERTAINTY VECTOR")
    print("=" * 70)
    conf = result.confidence
    print(f"score: {conf.score}   level: {conf.level}")
    print(f"feature_quality: {conf.vector.feature_quality:.3f}")
    print(f"ruling_quality:  {conf.vector.ruling_quality:.3f}")
    print(f"article_quality: {conf.vector.article_quality:.3f}")
    print(f"missing_channels: {conf.vector.missing_channels}")
    print(f"graph_support_quality: {conf.vector.graph_support_quality}  (always None -- disabled)")

    if hasattr(sample, "gold_checklist_by_article") and sample.gold_checklist_by_article:
        print("\n" + "=" * 70)
        print("GOLD CHECKLIST BY ARTICLE (for comparison with Auditor output)")
        print("=" * 70)
        for article, conditions in sample.gold_checklist_by_article.items():
            print(f"\n  {article}:")
            for c in conditions:
                print(f"    - [{'necessary' if c['necessary'] else 'optional'}] "
                      f"satisfied={c['satisfied']}: {c['condition'][:70]}")

    print()


if __name__ == "__main__":
    run(ruling_id="10493")