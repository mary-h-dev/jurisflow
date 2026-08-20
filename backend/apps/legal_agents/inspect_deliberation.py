"""
Debug script: single deliberation run — feeds a hardcoded AuditorOut
into the three-agent layer and prints per-agent verdicts, fusion result,
and any API/parse errors in full detail.

Swap _MOCK_AUDITOR_OUT to test different scenarios (high prune ratio,
split consensus, missing articles, etc.).

Run:
    DJANGO_SETTINGS_MODULE=config.settings.development python -c "
    import django; django.setup();
    from apps.legal_agents.inspect_deliberation import run; run()"
"""

from __future__ import annotations

import time

from apps.legal_agents.schemas import (
    AuditedArticleOut,
    AuditorOut,
    ChecklistItemOut,
    CombinedConfidenceOut,
)
from apps.legal_agents.services import deliberation_service

# ---------------------------------------------------------------------------
# Hardcoded AuditorOut — swap this to test different cases
# ---------------------------------------------------------------------------

_MOCK_AUDITOR_OUT = AuditorOut(
    query="آیا مزاحمت تلفنی با توجه به سابقه کیفری متهم مشمول مجازات جایگزین حبس می‌شود؟",
    verified_articles=[
        AuditedArticleOut(
            article_ref        = "قانون مجازات اسلامی - ماده ۶۴۱",
            is_applicable      = True,
            auditor_confidence = 0.91,
            checklist          = [
                ChecklistItemOut(
                    condition = "حداکثر مجازات قانونی بزه مزاحمت تلفنی طبق این ماده شش ماه حبس است",
                    necessary = True,
                    satisfied = True,
                ),
            ],
        ),
        AuditedArticleOut(
            article_ref        = "قانون مجازات اسلامی - ماده ۶۶",
            is_applicable      = True,
            auditor_confidence = 0.87,
            checklist          = [
                ChecklistItemOut(
                    condition = "تکلیف دادگاه به تعیین مجازات جایگزین حبس به جای حبس طبق این ماده",
                    necessary = True,
                    satisfied = True,
                ),
            ],
        ),
        AuditedArticleOut(
            article_ref        = "قانون آیین دادرسی کیفری - ماده ۴۷۴",
            is_applicable      = True,
            auditor_confidence = 0.78,
            checklist          = [
                ChecklistItemOut(
                    condition = "انطباق مورد با بند چ این ماده برای تجویز اعاده دادرسی",
                    necessary = True,
                    satisfied = True,
                ),
                ChecklistItemOut(
                    condition = "ارائه دلیل یا مدرک جدید مؤثر",
                    necessary = False,
                    satisfied = False,
                ),
            ],
        ),
        AuditedArticleOut(
            article_ref        = "قانون مدنی - ماده ۱۰",
            is_applicable      = False,          # pruned — agents must NOT cite this
            auditor_confidence = 0.30,
            checklist          = [
                ChecklistItemOut(
                    condition = "وجود قرارداد خصوصی معتبر بین طرفین",
                    necessary = True,
                    satisfied = False,
                ),
            ],
        ),
    ],
    pruned_articles=["قانون مدنی - ماده ۱۰"],
    confidence=CombinedConfidenceOut(
        score           = 0.74,
        level           = "high",
        retrieval_score = 0.81,
        auditor_score   = 0.85,
        prune_ratio     = 0.25,
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_opinion(role: str, opinion) -> None:
    if opinion is None:
        print(f"  [{role.upper()}]  FAILED — no opinion returned")
        return
    print(f"\n  [{role.upper()}]")
    print(f"    verdict    : {opinion.verdict.value}")
    print(f"    confidence : {opinion.confidence:.2f}")
    print(f"    position   : {opinion.position}")
    print(f"    arguments  :")
    for i, arg in enumerate(opinion.arguments, 1):
        print(f"      {i}. {arg}")
    if opinion.cited_articles:
        print(f"    cited      : {', '.join(opinion.cited_articles)}")
    if opinion.risks:
        print(f"    risks      :")
        for r in opinion.risks:
            print(f"      - {r}")


def _print_fusion(fusion) -> None:
    if fusion is None:
        print("  FUSION FAILED — see errors above")
        return
    print(f"\n  consensus      : {fusion.consensus_level.value}")
    print(f"  majority verdict: {fusion.majority_verdict.value}")
    print(f"  verdict spread  : {fusion.verdict_spread}")
    print(f"  agent conf mean : {fusion.agent_confidence_mean:.2f}  "
          f"std={fusion.agent_confidence_std:.2f}")
    print(f"  prior conf      : {fusion.prior_confidence:.2f}")
    print(f"  final conf      : {fusion.final_confidence:.2f}  ({fusion.final_level})")
    print(f"  uncertainty flag: {fusion.uncertainty_flag}")
    if fusion.agreed_articles:
        print(f"  agreed articles : {fusion.agreed_articles}")
    if fusion.notes:
        print("  notes:")
        for n in fusion.notes:
            print(f"    ⚠ {n}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    print("=" * 70)
    print("DELIBERATION LAYER DEBUG")
    print("=" * 70)
    print(f"\nQuery:\n  {_MOCK_AUDITOR_OUT.query}\n")

    applicable = [a for a in _MOCK_AUDITOR_OUT.verified_articles if a.is_applicable]
    pruned     = _MOCK_AUDITOR_OUT.pruned_articles

    print(f"Applicable articles ({len(applicable)}):")
    for a in applicable:
        print(f"  ✓  {a.article_ref}  (auditor_conf={a.auditor_confidence:.2f})")
    print(f"\nPruned articles ({len(pruned)}):")
    for ref in pruned:
        print(f"  ✗  {ref}  ← agents must NOT cite this")

    print(f"\nAuditor confidence : {_MOCK_AUDITOR_OUT.confidence.score:.2f}"
          f"  ({_MOCK_AUDITOR_OUT.confidence.level})")
    print(f"Prune ratio        : {_MOCK_AUDITOR_OUT.confidence.prune_ratio:.0%}")

    print("\n" + "=" * 70)
    print("RUNNING AGENTS  (this will make LLM API calls — may take ~30s)")
    print("=" * 70)

    started = time.monotonic()
    result  = deliberation_service.run(_MOCK_AUDITOR_OUT, session_id="inspect-debug")
    elapsed = time.monotonic() - started

    print("\n" + "=" * 70)
    print("PER-AGENT OPINIONS")
    print("=" * 70)
    _print_opinion("defender",   result.defender_opinion)
    _print_opinion("prosecutor", result.prosecutor_opinion)
    _print_opinion("judge",      result.judge_opinion)

    print("\n" + "=" * 70)
    print("FUSION RESULT  (deterministic — no LLM)")
    print("=" * 70)
    _print_fusion(result.fusion)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  completed nodes : {result.completed_nodes}")
    print(f"  total elapsed   : {elapsed:.1f}s")
    if result.error:
        print(f"\n  ⚠ ERROR: {result.error}")

    # Citation safety check — warn if any agent cited a pruned article
    pruned_set = set(pruned)
    for role, opinion in [
        ("defender",   result.defender_opinion),
        ("prosecutor", result.prosecutor_opinion),
        ("judge",      result.judge_opinion),
    ]:
        if opinion is None:
            continue
        leaked = pruned_set & set(opinion.cited_articles)
        if leaked:
            print(f"\n  ⚠ CITATION LEAK [{role}]: cited pruned article(s): {leaked}")
            print("    -> check system prompt or article formatting in _base.py")


if __name__ == "__main__":
    run()