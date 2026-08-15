from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

_PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def normalize_article_ref(ref: str) -> str:
    """Normalizes digits to Persian so refs compare equal regardless of
    which side (predicted or gold) used Latin digits."""
    return ref.translate(_PERSIAN_DIGITS)


def normalize_refs(refs: list[str]) -> set[str]:
    return {normalize_article_ref(r) for r in refs}


@dataclass
class PrecisionRecallF1:
    precision: float
    recall:    float
    f1:        float


def precision_recall_f1_refs(predicted: list[str], gold: list[str]) -> PrecisionRecallF1:
    """
    Computes precision/recall/f1 separately. Reporting all three matters:
    when top_k > len(gold) (common when gold has 1 article but top_k=5),
    F1 alone is structurally capped well below 1.0 even for perfect
    retrieval — recall isolates "did we find the right article(s)"
    from that top_k/gold-size mismatch.
    """
    pred_set = normalize_refs(predicted)
    gold_set = normalize_refs(gold)

    if not pred_set and not gold_set:
        return PrecisionRecallF1(precision=1.0, recall=1.0, f1=1.0)
    if not pred_set or not gold_set:
        return PrecisionRecallF1(precision=0.0, recall=0.0, f1=0.0)

    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    f1 = 0.0 if (precision + recall == 0) else 2 * precision * recall / (precision + recall)

    return PrecisionRecallF1(precision=precision, recall=recall, f1=f1)


def f1_score_refs(predicted: list[str], gold: list[str]) -> float:
    """Backward-compatible wrapper — returns only F1, for code that hasn't
    been updated to consume the full PrecisionRecallF1 result yet."""
    return precision_recall_f1_refs(predicted, gold).f1


def recall_at_k_refs(predicted: list[str], gold: list[str]) -> float:
    """
    Fraction of gold articles present anywhere in predicted (top_k).
    This is the metric that matters most when gold has fewer items than
    top_k — it directly answers "did we find the right article(s)"
    without being capped by the precision penalty of returning extra
    (correct, just unrequested) results.
    """
    pred_set = normalize_refs(predicted)
    gold_set = normalize_refs(gold)
    if not gold_set:
        return 1.0
    return len(pred_set & gold_set) / len(gold_set)


def spearman_correlation(confidence_scores: list[float], actual_scores: list[float]) -> float:
    if len(confidence_scores) < 2:
        return 0.0
    corr, _ = spearmanr(confidence_scores, actual_scores)
    return 0.0 if corr != corr else float(corr)  # NaN guard (e.g. constant input)


def auroc_high_quality(
    confidence_scores: list[float],
    actual_scores: list[float],
    threshold: float = 0.7,
) -> float | None:
    """
    Binarizes actual_scores (recall or F1) at `threshold` to define "high
    quality" ground truth, then measures how well confidence_scores
    rank-separate the two classes. Returns None if only one class is
    present (AUROC undefined) — this itself is diagnostic: it means the
    test set had no variation in outcome quality to evaluate against.
    """
    labels = [1 if s >= threshold else 0 for s in actual_scores]
    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, confidence_scores))