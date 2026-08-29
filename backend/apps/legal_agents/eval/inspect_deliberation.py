"""
Debug script: single deliberation run on a cached AuditorOut.

Uses real Auditor output (ruling 10362) — no search or audit API calls needed.
Only the three deliberation agents make LLM calls.

Run:
DJANGO_SETTINGS_MODULE=config.settings.development python -c "import django; django.setup(); from apps.legal_agents.inspect_deliberation import run; run()"
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
# Cached AuditorOut — ruling 10362
# ---------------------------------------------------------------------------

_AUDITOR_OUT = AuditorOut(
    query=(
        "در یک تصادف رانندگی که خودم مقصر بودم، همسر و فرزنم فوت کردند. "
        "آیا من به عنوان تنها وارث اون‌ها حق دریافت و مطالبه دیه فوتشون رو دارم؟"
    ),
    verified_articles=[
        AuditedArticleOut(
            article_ref="قانون مجازات اسلامی - ماده ۴۵۱",
            is_applicable=True, topically_relevant=True, auditor_confidence=1.0,
            checklist=[
                ChecklistItemOut(condition="قاتل از ورثه مقتول باشد", necessary=True, satisfied=True),
                ChecklistItemOut(condition="قتل به یکی از صور عمدی، شبه‌عمدی یا خطای محض واقع شده باشد", necessary=True, satisfied=True),
            ],
        ),
        AuditedArticleOut(
            article_ref="قانون مجازات اسلامی - ماده ۷۱۴",
            is_applicable=False, topically_relevant=False, auditor_confidence=0.0,
            checklist=[
                ChecklistItemOut(condition="وقوع صدمه بر صورت یا سایر اعضای بدن", necessary=True, satisfied=False),
                ChecklistItemOut(condition="ایجاد تغییر رنگ پوست در اثر صدمه واردشده", necessary=True, satisfied=False),
            ],
        ),
        AuditedArticleOut(
            article_ref="قانون مجازات اسلامی - ماده ۶۱۶",
            is_applicable=False, topically_relevant=True, auditor_confidence=0.667,
            checklist=[
                ChecklistItemOut(condition="وقوع قتل غیرعمد به واسطه بی‌احتیاطی یا بی‌مبالاتی", necessary=True, satisfied=True),
                ChecklistItemOut(condition="عدم شمول استثنای خطای محض بر حادثه مطروحه", necessary=True, satisfied=True),
                ChecklistItemOut(condition="وجود مطالبه دیه از ناحیه اولیای دم واجد صلاحیت قانونی", necessary=True, satisfied=False),
            ],
        ),
        AuditedArticleOut(
            article_ref="قانون مجازات اسلامی - ماده ۳۵۷",
            is_applicable=False, topically_relevant=True, auditor_confidence=0.5,
            checklist=[
                ChecklistItemOut(condition="جنایت ارتکابی از نوع جنایت عمدی باشد", necessary=True, satisfied=False),
                ChecklistItemOut(condition="مرتکب یا شریک جنایت، از ورثه مقتول باشد", necessary=True, satisfied=True),
            ],
        ),
        AuditedArticleOut(
            article_ref="قانون مجازات اسلامی - ماده ۳۵۲",
            is_applicable=False, topically_relevant=True, auditor_confidence=0.5,
            checklist=[
                ChecklistItemOut(condition="حق قصاص به دیه تبدیل شده یا مصالحه شده باشد", necessary=True, satisfied=False),
                ChecklistItemOut(condition="متقاضی ارث‌بری از دیه، همسر مقتول باشد", necessary=True, satisfied=True),
            ],
        ),
        AuditedArticleOut(
            article_ref="قانون مجازات اسلامی - ماده ۳۵۱",
            is_applicable=True, topically_relevant=True, auditor_confidence=1.0,
            checklist=[
                ChecklistItemOut(condition="شخص مطالبه‌کننده، از ورثه قانونی مقتول باشد", necessary=True, satisfied=True),
                ChecklistItemOut(condition="موضوع ادعا غیر از اعمال حق قصاص باشد", necessary=True, satisfied=True),
            ],
        ),
        AuditedArticleOut(
            article_ref="قانون مدنی - ماده ۹۴۶",
            is_applicable=True, topically_relevant=True, auditor_confidence=1.0,
            checklist=[
                ChecklistItemOut(condition="وجود رابطه زوجیت قانونی در زمان فوت", necessary=True, satisfied=True),
                ChecklistItemOut(condition="قرار گرفتن موضوع در زمره اموال مالی به‌جامانده از زوجه متوفی", necessary=True, satisfied=True),
            ],
        ),
        AuditedArticleOut(
            article_ref="قانون مدنی - ماده ۹۰۶",
            is_applicable=True, topically_relevant=True, auditor_confidence=1.0,
            checklist=[
                ChecklistItemOut(condition="عدم وجود اولاد یا اولادِ اولاد برای متوفی", necessary=True, satisfied=True),
                ChecklistItemOut(condition="زنده بودن حداقل یکی از ابوین متوفی", necessary=True, satisfied=True),
            ],
        ),
        AuditedArticleOut(
            article_ref="قانون مدنی - ماده ۸۸۰",
            is_applicable=False, topically_relevant=True, auditor_confidence=0.5,
            checklist=[
                ChecklistItemOut(condition="وقوع قتل مورث توسط وارث", necessary=True, satisfied=True),
                ChecklistItemOut(condition="عمدی بودن قتل", necessary=True, satisfied=False),
            ],
        ),
        AuditedArticleOut(
            article_ref="قانون مجازات اسلامی - ماده ۴۶۲",
            is_applicable=True, topically_relevant=True, auditor_confidence=1.0,
            checklist=[
                ChecklistItemOut(condition="ارتکاب جنایت", necessary=True, satisfied=True),
                ChecklistItemOut(condition="عمدی یا شبه‌عمدی بودن جنایت", necessary=True, satisfied=True),
            ],
        ),
        AuditedArticleOut(
            article_ref="قانون مجازات اسلامی - ماده ۴۶۸",
            is_applicable=False, topically_relevant=True, auditor_confidence=0.5,
            checklist=[
                ChecklistItemOut(condition="فرد از بستگان ذکور نسبی پدری باشد", necessary=True, satisfied=True),
                ChecklistItemOut(condition="فرد در زمان فوت توانایی ارث‌بردن داشته باشد", necessary=True, satisfied=False),
            ],
        ),
        AuditedArticleOut(
            article_ref="قانون مجازات اسلامی - ماده ۴۸۷",
            is_applicable=False, topically_relevant=True, auditor_confidence=0.5,
            checklist=[
                ChecklistItemOut(condition="وقوع قتل یا کشته شدن یک شخص", necessary=True, satisfied=True),
                ChecklistItemOut(condition="شناخته نشدن قاتل یا کشته شدن بر اثر ازدحام", necessary=True, satisfied=False),
            ],
        ),
        AuditedArticleOut(
            article_ref="قانون مدنی - ماده ۸۸۵",
            is_applicable=False, topically_relevant=True, auditor_confidence=0.5,
            checklist=[
                ChecklistItemOut(condition="مطالبه ارث توسط اولاد شخص ممنوع از ارث", necessary=True, satisfied=False),
                ChecklistItemOut(condition="عدم وجود وارث نزدیک‌تر", necessary=True, satisfied=True),
            ],
        ),
        AuditedArticleOut(
            article_ref="قانون مدنی - ماده ۸۶۶",
            is_applicable=False, topically_relevant=True, auditor_confidence=0.5,
            checklist=[
                ChecklistItemOut(condition="عدم وجود هرگونه وارث برای متوفی", necessary=True, satisfied=False),
                ChecklistItemOut(condition="وجود ترکه یا حقوق مالی به‌جامانده از متوفی", necessary=True, satisfied=True),
            ],
        ),
        AuditedArticleOut(
            article_ref="قانون مجازات اسلامی - ماده ۳۴۸",
            is_applicable=False, topically_relevant=True, auditor_confidence=0.0,
            checklist=[
                ChecklistItemOut(condition="موضوع حق یا ادعا «حق قصاص» باشد", necessary=True, satisfied=False),
                ChecklistItemOut(condition="شرایط انتقال و ارث‌بری حق قصاص محقق باشد", necessary=True, satisfied=False),
            ],
        ),
    ],
    pruned_articles=[
        "قانون مجازات اسلامی - ماده ۷۱۴",
        "قانون مدنی - ماده ۶۲۷",
        "قانون مدنی - ماده ۹۱۱",
        "قانون مدنی - ماده ۹۳۷",
        "قانون مجازات اسلامی - ماده ۴۴۹",
        "قانون آیین دادرسی مدنی - ماده ۳۴۸",
        "قانون آیین دادرسی مدنی - ماده ۳",
        "قانون آیین دادرسی مدنی - ماده ۵",
        "قانون آیین دادرسی مدنی - ماده ۳۵۸",
        "قانون مجازات اسلامی - ماده ۷۱۸",
    ],
    confidence=CombinedConfidenceOut(
        score=0.74,
        level="high",
        retrieval_score=0.81,
        auditor_score=0.85,
        prune_ratio=0.40,
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_articles_summary() -> None:
    applicable   = [a for a in _AUDITOR_OUT.verified_articles if a.is_applicable]
    topical_only = [a for a in _AUDITOR_OUT.verified_articles
                    if not a.is_applicable and a.topically_relevant]
    pruned       = [a for a in _AUDITOR_OUT.verified_articles
                    if not a.is_applicable and not a.topically_relevant]

    print(f"\nAPPLICABLE ({len(applicable)}):")
    for a in applicable:
        print(f"  ✓  {a.article_ref}  (conf={a.auditor_confidence:.2f})")

    print(f"\nNOT APPLICABLE but topically relevant ({len(topical_only)}):")
    for a in topical_only:
        failed = [c.condition for c in a.checklist if c.necessary and not c.satisfied]
        print(f"  ~  {a.article_ref}  (conf={a.auditor_confidence:.2f})")
        for f in failed:
            print(f"       failed: {f}")

    print(f"\nPRUNED — not sent to agents ({len(pruned)}):")
    for a in pruned:
        print(f"  ✗  {a.article_ref}")


def _print_opinion(role: str, opinion) -> None:
    if opinion is None:
        print(f"\n  [{role.upper()}]  FAILED — no opinion returned")
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
    print(f"\n  consensus       : {fusion.consensus_level.value}")
    print(f"  majority verdict: {fusion.majority_verdict.value}")
    print(f"  verdict spread  : {fusion.verdict_spread}")
    print(f"  agent conf      : mean={fusion.agent_confidence_mean:.2f}  std={fusion.agent_confidence_std:.2f}")
    print(f"  prior conf      : {fusion.prior_confidence:.2f}")
    print(f"  final conf      : {fusion.final_confidence:.2f}  ({fusion.final_level})")
    print(f"  uncertainty flag: {fusion.uncertainty_flag}")
    if fusion.agreed_articles:
        print(f"  agreed articles : {fusion.agreed_articles}")
    if fusion.notes:
        print("  notes:")
        for n in fusion.notes:
            print(f"    ⚠  {n}")


def _citation_leak_check(result) -> None:
    pruned_set = {a.article_ref for a in _AUDITOR_OUT.verified_articles
                  if not a.is_applicable and not a.topically_relevant}
    found_leak = False
    for role, opinion in [
        ("defender",   result.defender_opinion),
        ("prosecutor", result.prosecutor_opinion),
        ("judge",      result.judge_opinion),
    ]:
        if opinion is None:
            continue
        leaked = pruned_set & set(opinion.cited_articles)
        if leaked:
            found_leak = True
            print(f"  ⚠  CITATION LEAK [{role}]: cited pruned article(s): {leaked}")
    if not found_leak:
        print("  ✓  No citation leaks — agents respected pruned articles")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    print("=" * 70)
    print("DELIBERATION DEBUG  |  ruling=10362")
    print("=" * 70)
    print(f"\nQuery:\n  {_AUDITOR_OUT.query}\n")
    print(f"Auditor confidence: {_AUDITOR_OUT.confidence.score:.2f}"
          f"  ({_AUDITOR_OUT.confidence.level})"
          f"  prune_ratio={_AUDITOR_OUT.confidence.prune_ratio:.0%}")

    print("\n" + "=" * 70)
    print("ARTICLE SUMMARY")
    print("=" * 70)
    _print_articles_summary()

    print("\n" + "=" * 70)
    print("RUNNING AGENTS  (LLM calls — may take ~30s)")
    print("=" * 70)

    started = time.monotonic()
    result  = deliberation_service.run(_AUDITOR_OUT, session_id="inspect-10362")
    elapsed = time.monotonic() - started

    print("\n" + "=" * 70)
    print("PER-AGENT OPINIONS")
    print("=" * 70)
    _print_opinion("defender",   result.defender_opinion)
    _print_opinion("prosecutor", result.prosecutor_opinion)
    _print_opinion("judge",      result.judge_opinion)

    print("\n" + "=" * 70)
    print("FUSION  (deterministic)")
    print("=" * 70)
    _print_fusion(result.fusion)

    print("\n" + "=" * 70)
    print("CITATION LEAK CHECK")
    print("=" * 70)
    _citation_leak_check(result)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  completed nodes : {result.completed_nodes}")
    print(f"  total elapsed   : {elapsed:.1f}s")
    if result.error:
        print(f"\n  ⚠  ERROR: {result.error}")


if __name__ == "__main__":
    run()