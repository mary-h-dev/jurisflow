"""
Builds the user-facing CaseChecklist from a completed DeliberationOut.

No LLM involved — all labels are derived deterministically from the
structured fields already present in FusionOut and AgentOpinion.
"""

from __future__ import annotations

from apps.legal_agents.schemas import (
    ActionItem,
    AgentChecklistItem,
    ArticleChecklistItem,
    CaseChecklist,
    ConsensusLevel,
    DeliberationOut,
    Verdict,
)

# ---------------------------------------------------------------------------
# Label maps
# ---------------------------------------------------------------------------

_VERDICT_FA: dict[str, str] = {
    Verdict.STRONG_FOR:       "پرونده بسیار قوی",
    Verdict.MODERATE_FOR:     "پرونده نسبتاً قوی",
    Verdict.NEUTRAL:          "بلاتکلیف",
    Verdict.MODERATE_AGAINST: "پرونده نسبتاً ضعیف",
    Verdict.STRONG_AGAINST:   "پرونده بسیار ضعیف",
}

_VERDICT_EN: dict[str, str] = {
    Verdict.STRONG_FOR:       "Strong case for client",
    Verdict.MODERATE_FOR:     "Moderate case for client",
    Verdict.NEUTRAL:          "Inconclusive",
    Verdict.MODERATE_AGAINST: "Moderate case against client",
    Verdict.STRONG_AGAINST:   "Strong case against client",
}

_ROLE_FA: dict[str, str] = {
    "defender":   "وکیل مدافع",
    "prosecutor": "دادستان",
    "judge":      "قاضی",
}

_CONSENSUS_FA: dict[str, str] = {
    ConsensusLevel.FULL:     "توافق کامل",
    ConsensusLevel.MAJORITY: "اکثریت",
    ConsensusLevel.SPLIT:    "اختلاف نظر",
}

_CONSENSUS_EN: dict[str, str] = {
    ConsensusLevel.FULL:     "Full agreement",
    ConsensusLevel.MAJORITY: "Majority",
    ConsensusLevel.SPLIT:    "Split",
}


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_articles(deliberation: DeliberationOut) -> list[ArticleChecklistItem]:
    items: list[ArticleChecklistItem] = []

    for art in deliberation.applicable_articles:
        # applicable_articles may be dicts or AuditedArticleOut objects
        if isinstance(art, dict):
            checklist     = art.get("checklist", [])
            is_applicable = art.get("is_applicable", False)
            article_ref   = art.get("article_ref", "")
            satisfied = [c["condition"] for c in checklist if c.get("satisfied")]
            failed    = [c["condition"] for c in checklist
                         if not c.get("satisfied") and c.get("necessary")]
        else:
            checklist     = art.checklist
            is_applicable = art.is_applicable
            article_ref   = art.article_ref
            satisfied = [c.condition for c in checklist if c.satisfied]
            failed    = [c.condition for c in checklist
                         if not c.satisfied and c.necessary]

        if is_applicable:
            fa_label = "صدق می‌کند"
            en_label = "Applies"
            reason   = satisfied[0] if satisfied else ""
        else:
            fa_label = "صدق نمی‌کند"
            en_label = "Does not apply"
            reason   = failed[0] if failed else ""

        items.append(ArticleChecklistItem(
            article_ref   = article_ref,
            is_applicable = is_applicable,
            fa_label      = fa_label,
            en_label      = en_label,
            reason        = reason,
        ))

    return items


def _build_agents(deliberation: DeliberationOut) -> list[AgentChecklistItem]:
    items: list[AgentChecklistItem] = []

    for opinion in [
        deliberation.defender_opinion,
        deliberation.prosecutor_opinion,
        deliberation.judge_opinion,
    ]:
        if opinion is None:
            continue
        verdict_val = opinion.verdict if isinstance(opinion.verdict, str) else opinion.verdict.value
        items.append(AgentChecklistItem(
            role       = opinion.role,
            fa_role    = _ROLE_FA.get(opinion.role, opinion.role),
            verdict    = verdict_val,
            fa_verdict = _VERDICT_FA.get(verdict_val, verdict_val),
            confidence = opinion.confidence,
            position   = opinion.position,
        ))

    return items


def _build_actions(deliberation: DeliberationOut) -> list[ActionItem]:
    if deliberation.fusion is None:
        return []

    fusion   = deliberation.fusion
    verdict  = fusion.majority_verdict
    verdict_val = verdict if isinstance(verdict, str) else verdict.value
    uncertain = fusion.uncertainty_flag
    actions: list[ActionItem] = []
    priority = 1

    # Action 1 — consult a lawyer (always first)
    actions.append(ActionItem(
        priority  = priority,
        fa_action = "با یک وکیل متخصص مشورت کنید",
        en_action = "Consult a specialised lawyer",
    ))
    priority += 1

    # Action 2 — based on verdict direction
    if verdict_val in (Verdict.STRONG_FOR, Verdict.MODERATE_FOR):
        actions.append(ActionItem(
            priority  = priority,
            fa_action = "مدارک و شواهد مستدل خود را جمع‌آوری و آماده کنید",
            en_action = "Gather and organise supporting documents and evidence",
        ))
    elif verdict_val in (Verdict.STRONG_AGAINST, Verdict.MODERATE_AGAINST):
        actions.append(ActionItem(
            priority  = priority,
            fa_action = "نقاط ضعف پرونده را با وکیل بررسی و راه‌حل جایگزین پیدا کنید",
            en_action = "Review case weaknesses with your lawyer and explore alternatives",
        ))
    else:
        actions.append(ActionItem(
            priority  = priority,
            fa_action = "مدارک بیشتری برای روشن شدن وضعیت پرونده فراهم کنید",
            en_action = "Provide additional documents to clarify the case",
        ))
    priority += 1

    # Action 3 — study agreed articles
    if fusion.agreed_articles:
        refs = "، ".join(fusion.agreed_articles[:2])
        actions.append(ActionItem(
            priority  = priority,
            fa_action = f"مواد قانونی کلیدی را مطالعه کنید: {refs}",
            en_action = f"Study the key legal articles: {', '.join(fusion.agreed_articles[:2])}",
        ))
        priority += 1

    # Action 4 — uncertainty warning
    if uncertain:
        actions.append(ActionItem(
            priority  = priority,
            fa_action = "به دلیل عدم قطعیت بالا، نظر چند وکیل مختلف را بگیرید",
            en_action = "Due to high uncertainty, seek opinions from multiple lawyers",
        ))

    return actions


def _build_uncertainty(deliberation: DeliberationOut) -> tuple[str, str]:
    if deliberation.fusion is None or not deliberation.fusion.uncertainty_flag:
        return "", ""

    fusion = deliberation.fusion
    reasons_fa: list[str] = []
    reasons_en: list[str] = []

    consensus_val = fusion.consensus_level if isinstance(fusion.consensus_level, str) \
        else fusion.consensus_level.value

    if consensus_val == ConsensusLevel.SPLIT:
        reasons_fa.append("سه کارشناس اختلاف نظر دارند")
        reasons_en.append("the three experts disagree")

    if fusion.agent_confidence_std > 0.20:
        reasons_fa.append("اطمینان کارشناسان متفاوت است")
        reasons_en.append("expert confidence levels vary significantly")

    if deliberation.auditor_confidence:
        conf = deliberation.auditor_confidence
        prune_ratio = conf.get("prune_ratio", 0) if isinstance(conf, dict) \
            else conf.prune_ratio
        if prune_ratio > 0.5:
            reasons_fa.append("بخش زیادی از مواد بازیابی‌شده نامرتبط بود")
            reasons_en.append("a large portion of retrieved articles were irrelevant")

    fa = "عدم قطعیت بالا: " + "، ".join(reasons_fa) if reasons_fa else ""
    en = "High uncertainty: " + ", ".join(reasons_en) if reasons_en else ""
    return fa, en


def _overall_summary(verdict_val: str) -> tuple[str, str]:
    fa_map = {
        Verdict.STRONG_FOR:       "بر اساس تحلیل حقوقی، پرونده در وضعیت مطلوبی قرار دارد.",
        Verdict.MODERATE_FOR:     "بر اساس تحلیل حقوقی، پرونده نسبتاً مطلوب است.",
        Verdict.NEUTRAL:          "بر اساس تحلیل حقوقی، نتیجه پرونده قطعی نیست.",
        Verdict.MODERATE_AGAINST: "بر اساس تحلیل حقوقی، پرونده با چالش‌هایی مواجه است.",
        Verdict.STRONG_AGAINST:   "بر اساس تحلیل حقوقی، ادعا با موانع قانونی جدی روبروست.",
    }
    en_map = {
        Verdict.STRONG_FOR:       "Legal analysis indicates a strong case.",
        Verdict.MODERATE_FOR:     "Legal analysis indicates a moderately favourable case.",
        Verdict.NEUTRAL:          "Legal analysis is inconclusive.",
        Verdict.MODERATE_AGAINST: "Legal analysis indicates notable challenges.",
        Verdict.STRONG_AGAINST:   "Legal analysis indicates serious legal obstacles.",
    }
    return fa_map.get(verdict_val, ""), en_map.get(verdict_val, "")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_checklist(deliberation: DeliberationOut) -> CaseChecklist | None:
    """
    Build a CaseChecklist from a completed DeliberationOut.
    Returns None if fusion is not available.
    """
    if deliberation.fusion is None:
        return None

    fusion    = deliberation.fusion
    verdict   = fusion.majority_verdict
    verdict_val = verdict if isinstance(verdict, str) else verdict.value
    consensus = fusion.consensus_level
    consensus_val = consensus if isinstance(consensus, str) else consensus.value

    overall_fa, overall_en     = _overall_summary(verdict_val)
    uncertainty_fa, uncertainty_en = _build_uncertainty(deliberation)

    return CaseChecklist(
        overall_fa      = overall_fa,
        overall_en      = overall_en,
        articles        = _build_articles(deliberation),
        agents          = _build_agents(deliberation),
        consensus_fa    = _CONSENSUS_FA.get(consensus_val, consensus_val),
        consensus_en    = _CONSENSUS_EN.get(consensus_val, consensus_val),
        actions         = _build_actions(deliberation),
        uncertainty_flag = fusion.uncertainty_flag,
        uncertainty_fa  = uncertainty_fa,
        uncertainty_en  = uncertainty_en,
    )