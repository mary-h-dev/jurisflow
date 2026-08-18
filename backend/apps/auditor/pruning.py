from __future__ import annotations

from .schemas import AuditedArticleOut, ChecklistItemOut

_APPLICABILITY_THRESHOLD = 0.6  # used only when no item is flagged necessary


def decide(article_ref: str, checklist: list[ChecklistItemOut]) -> AuditedArticleOut:
    """
    Applicability rule:
      - if any checklist item is flagged `necessary`, the article is
        applicable only if ALL necessary items are satisfied (matches
        the reference framework's hard verify-and-prune gate)
      - otherwise, fall back to a soft majority rule over all items

    `auditor_confidence` is the satisfaction ratio over the same item
    set used for the decision. It is independent of the retrieval-layer
    confidence.score produced by apps.search.
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
        auditor_confidence=round(ratio, 3),
    )


def prune(audited_articles: list[AuditedArticleOut]) -> list[str]:
    """Returns article_refs judged not applicable, for removal from the
    evidence set passed downstream to the Adjudicator."""
    return [a.article_ref for a in audited_articles if not a.is_applicable]