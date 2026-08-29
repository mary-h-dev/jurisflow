"""
End-to-end debug: query → search → audit → deliberation → uncertainty report.

AuditorOut is cached to JSON after the first run so subsequent runs
skip the expensive search+audit step (important when API calls are limited).

Cache location: /tmp/jurisflow_e2e_cache_{ruling_id}.json

Run:
DJANGO_SETTINGS_MODULE=config.settings.development python -c "import django; django.setup(); from apps.legal_agents.inspect_e2e import run; run()"

To force a fresh run (ignore cache):
    from apps.legal_agents.inspect_e2e import run; run(force_fresh=True)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Gold annotation — ruling 10362
# ---------------------------------------------------------------------------

_GOLD = {
    "ruling_id":             "10362",
    "case_type":             "کیفری",
    "query": (
        "در یک تصادف رانندگی که خودم مقصر بودم، همسر و فرزنم فوت کردند. "
        "آیا من به عنوان تنها وارث اون‌ها حق دریافت و مطالبه دیه فوتشون رو دارم؟"
    ),
    "gold_articles":         ["قانون مجازات اسلامی - ماده ۴۵۱"],
    "annotation_confidence": 0.95,
    "gold_checklist": {
        "قانون مجازات اسلامی - ماده ۴۵۱": [
            {"condition": "وقوع قتل یا صدمه بدنی ناشی از شبه‌عمد یا خطای محض محرز است",    "necessary": True, "satisfied": True},
            {"condition": "مطالبه‌کننده دیه، خود مرتکب و مقصر اصلی حادثه است",             "necessary": True, "satisfied": True},
            {"condition": "مرتکب به‌عنوان وارث مقتول، مطالبه دیه همان حادثه را نموده است", "necessary": True, "satisfied": True},
            {"condition": "مرتکب شبه‌عمد یا خطای محض از سهم دیه مقتول محروم است",          "necessary": True, "satisfied": True},
        ],
    },
}

_CACHE_DIR = Path("/tmp")


def _cache_path(ruling_id: str) -> Path:
    return _CACHE_DIR / f"jurisflow_e2e_cache_{ruling_id}.json"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _save_auditor_cache(ruling_id: str, auditor_out) -> None:
    path = _cache_path(ruling_id)
    path.write_text(auditor_out.model_dump_json(indent=2), encoding="utf-8")
    print(f"  [cache] saved → {path}")


def _load_auditor_cache(ruling_id: str):
    path = _cache_path(ruling_id)
    if not path.exists():
        return None

    from apps.auditor.schemas import AuditorOut
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        auditor_out = AuditorOut(**data)
        print(f"  [cache] loaded from {path} — skipping search+audit")
        return auditor_out
    except Exception as exc:
        print(f"  [cache] load failed ({exc}) — will re-run")
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _separator(title: str = "") -> None:
    print("\n" + "=" * 70)
    if title:
        print(title)
        print("=" * 70)


def _print_retrieval_vs_gold(article_refs: list[str], gold: list[str]) -> None:
    gold_set      = set(gold)
    retrieved_set = set(article_refs)
    hit           = gold_set & retrieved_set
    miss          = gold_set - retrieved_set

    print(f"\n  Retrieved articles : {len(article_refs)}")
    print(f"  Gold articles      : {len(gold)}")
    print(f"\n  Gold FOUND in retrieval ({len(hit)}):")
    for a in sorted(hit):
        print(f"    ✓  {a}")
    if miss:
        print(f"\n  Gold MISSED by retrieval ({len(miss)}):")
        for a in sorted(miss):
            print(f"    ✗  {a}  ← retrieval did not surface this")


def _print_auditor_vs_gold(auditor_out, gold_checklist: dict) -> None:
    applicable = [a for a in auditor_out.verified_articles if a.is_applicable]
    print(f"\n  Applicable after pruning : {len(applicable)}")
    for a in applicable:
        print(f"    ✓  {a.article_ref}  (conf={a.auditor_confidence:.2f})")

    print(f"\n  Pruned ({len(auditor_out.pruned_articles)}) :")
    for ref in auditor_out.pruned_articles:
        print(f"    ✗  {ref}")

    gold_refs = set(gold_checklist.keys())
    print(f"\n  Checklist vs gold:")
    for art in auditor_out.verified_articles:
        if art.article_ref not in gold_refs:
            continue
        gold_items = {g["condition"]: g["satisfied"] for g in gold_checklist[art.article_ref]}
        print(f"\n    [{art.article_ref}]  is_applicable={art.is_applicable}")
        for item in art.checklist:
            gold_val = gold_items.get(item.condition)
            if gold_val is None:
                match = "(no gold match)"
            else:
                match = "✓ match" if item.satisfied == gold_val else f"✗ MISMATCH (gold={gold_val})"
            print(f"      {match}  satisfied={item.satisfied}  — {item.condition[:70]}")


def _print_uncertainty_report(result, auditor_out, gold: dict) -> None:
    fusion = result.fusion
    if fusion is None:
        print("  FUSION FAILED")
        return

    gold_conf = gold["annotation_confidence"]

    print(f"\n  ── Confidence by layer ──────────────────────────────────")
    print(f"  Retrieval score  : {auditor_out.confidence.retrieval_score:.2f}")
    print(f"  Auditor score    : {auditor_out.confidence.auditor_score:.2f}")
    print(f"  Prune ratio      : {auditor_out.confidence.prune_ratio:.0%}")
    print(f"  Prior (combined) : {auditor_out.confidence.score:.2f}  ({auditor_out.confidence.level})")

    print(f"\n  ── Agent deliberation ───────────────────────────────────")
    print(f"  Verdict spread   : {fusion.verdict_spread}")
    print(f"  Consensus        : {fusion.consensus_level.value}")
    print(f"  Agent conf mean  : {fusion.agent_confidence_mean:.2f}  std={fusion.agent_confidence_std:.2f}")

    print(f"\n  ── Final output ─────────────────────────────────────────")
    print(f"  Final confidence : {fusion.final_confidence:.2f}  ({fusion.final_level})")
    print(f"  Uncertainty flag : {fusion.uncertainty_flag}")
    print(f"  Gold confidence  : {gold_conf:.2f}")
    delta = abs(fusion.final_confidence - gold_conf)
    quality = "✓ good" if delta < 0.15 else "⚠ review needed"
    print(f"  Delta vs gold    : {delta:.2f}  {quality}")

    if fusion.notes:
        print(f"\n  ── Uncertainty notes ────────────────────────────────────")
        for n in fusion.notes:
            print(f"    ⚠  {n}")

    gold_set   = set(gold["gold_articles"])
    agreed_set = set(fusion.agreed_articles)
    overlap    = gold_set & agreed_set
    print(f"\n  ── Article agreement ────────────────────────────────────")
    print(f"  Agreed by ≥2 agents : {fusion.agreed_articles}")
    print(f"  Overlap with gold   : {sorted(overlap)}")
    if gold_set - agreed_set:
        print(f"  Gold NOT agreed     : {sorted(gold_set - agreed_set)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(force_fresh: bool = False) -> None:
    from apps.legal_agents.services import deliberation_service

    query     = _GOLD["query"]
    ruling_id = _GOLD["ruling_id"]

    _separator(f"END-TO-END DEBUG  |  ruling={ruling_id}  |  {_GOLD['case_type']}")
    print(f"\nQuery:\n  {query}")
    print(f"\nGold articles   : {_GOLD['gold_articles']}")
    print(f"Annotation conf : {_GOLD['annotation_confidence']}")

    # ── Layer 1 + 2: Search + Audit (with cache) ──────────────────────────
    auditor_out = None if force_fresh else _load_auditor_cache(ruling_id)
    retrieval_elapsed = audit_elapsed = 0.0

    if auditor_out is None:
        from apps.search.calibration.leakage_guard import exclude_self_ruling
        from apps.search.services import search_service
        from apps.auditor.services import auditor_service

        _separator("LAYER 1 — RETRIEVAL")
        t0           = time.monotonic()
        search_raw   = search_service.search(query)
        search_result = exclude_self_ruling(search_raw, ruling_id)
        retrieval_elapsed = time.monotonic() - t0

        print(f"\n  Retrieval confidence : {search_result.confidence.score:.2f}")
        print(f"  Elapsed              : {retrieval_elapsed:.1f}s")
        _print_retrieval_vs_gold(search_result.article_refs, _GOLD["gold_articles"])

        _separator("LAYER 2 — AUDITOR")
        t1          = time.monotonic()
        auditor_out = auditor_service.audit(search_result)
        audit_elapsed = time.monotonic() - t1

        print(f"\n  Auditor confidence : {auditor_out.confidence.score:.2f}  ({auditor_out.confidence.level})")
        print(f"  Prune ratio        : {auditor_out.confidence.prune_ratio:.0%}")
        print(f"  Elapsed            : {audit_elapsed:.1f}s")
        _print_auditor_vs_gold(auditor_out, _GOLD["gold_checklist"])
        _save_auditor_cache(ruling_id, auditor_out)

    else:
        _separator("LAYER 1+2 — RETRIEVAL+AUDITOR  (from cache)")
        _print_auditor_vs_gold(auditor_out, _GOLD["gold_checklist"])

    # ── Layer 3: Deliberation ─────────────────────────────────────────────
    _separator("LAYER 3 — DELIBERATION AGENTS")
    t2     = time.monotonic()
    result = deliberation_service.run(auditor_out, session_id=f"e2e-{ruling_id}")
    agent_elapsed = time.monotonic() - t2

    print(f"\n  Elapsed : {agent_elapsed:.1f}s")
    for role, opinion in [
        ("defender",   result.defender_opinion),
        ("prosecutor", result.prosecutor_opinion),
        ("judge",      result.judge_opinion),
    ]:
        if opinion is None:
            print(f"  [{role:<10}]  FAILED")
            continue
        print(f"  [{role:<10}]  verdict={opinion.verdict.value:<16}  conf={opinion.confidence:.2f}")

    # ── Uncertainty report ────────────────────────────────────────────────
    _separator("UNCERTAINTY REPORT")
    _print_uncertainty_report(result, auditor_out, _GOLD)

    # ── Summary ───────────────────────────────────────────────────────────
    _separator("SUMMARY")
    total = retrieval_elapsed + audit_elapsed + agent_elapsed
    print(f"  Completed nodes : {result.completed_nodes}")
    print(f"  Total elapsed   : {total:.1f}s  "
          f"(retrieval={retrieval_elapsed:.1f}s  "
          f"audit={audit_elapsed:.1f}s  "
          f"agents={agent_elapsed:.1f}s)")
    if result.error:
        print(f"\n  ⚠  ERROR: {result.error}")

    if result.fusion:
        flag  = result.fusion.uncertainty_flag
        level = result.fusion.final_level
        print(f"\n  uncertainty_flag={flag}  final_level={level}")
        if flag:
            print("  → سیستم عدم قطعیت بالا تشخیص داد")
        else:
            print("  → سیستم با اطمینان نسبی نتیجه داد")

    print(f"\n  [cache] next run will load auditor from: {_cache_path(ruling_id)}")