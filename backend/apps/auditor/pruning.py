from __future__ import annotations

from .schemas import AuditedArticleOut, ChecklistItemOut

_APPLICABILITY_THRESHOLD = 0.6  


def decide(
    article_ref: str,
    checklist: list[ChecklistItemOut],
    topically_relevant: bool,
) -> AuditedArticleOut:
    """
    Applicability rule:
      - if any checklist item is flagged `necessary`, the article is
        applicable only if ALL necessary items are satisfied (matches
        the reference framework's hard verify-and-prune gate)
      - otherwise, fall back to a soft majority rule over all items

    `auditor_confidence` is the satisfaction ratio over the same item
    set used for the decision. It is independent of the retrieval-layer
    confidence.score produced by apps.search.

    `topically_relevant` is passed through unchanged from verify_article
    -- see AuditedArticleOut's docstring for why it must stay separate
    from is_applicable: an article can be topically relevant and still
    correctly come out is_applicable=False (a failed threshold IS often
    the answer), and that combination must not be treated as noise.
    """
    necessary_items = [i for i in checklist if i.necessary]
    decision_items = necessary_items or checklist

    if decision_items:
        satisfied_count = sum(1 for i in decision_items if i.satisfied)
        ratio = satisfied_count / len(decision_items)
    else:
        ratio = 0.0

    is_applicable = (ratio == 1.0) if necessary_items else (ratio >= _APPLICABILITY_THRESHOLD)

    return AuditedArticleOut(
        article_ref=article_ref,
        checklist=checklist,
        is_applicable=is_applicable,
        topically_relevant=topically_relevant,
        auditor_confidence=round(ratio, 3),
    )


def prune(audited_articles: list[AuditedArticleOut]) -> list[str]:
    """
    Returns article_refs where is_applicable=False, as a SUMMARY label
    only -- these articles are NOT removed from AuditorOut.verified_articles
    and must not be dropped downstream just because they appear here.
    "Not applicable" is frequently the correct legal answer itself (see
    AuditedArticleOut's docstring), so this list is for reporting/metrics
    (e.g. prune_ratio in confidence.py), not for filtering evidence.
    Only topically_relevant=False articles are safe to drop from a
    final answer's citations.
    """
    return [a.article_ref for a in audited_articles if not a.is_applicable]