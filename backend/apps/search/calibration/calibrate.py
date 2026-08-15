from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

from apps.search import confidence as confidence_module
from apps.search.services import search_service
from .data import AnnotatedSample, stratified_k_folds
from .metrics import spearman_correlation, auroc_high_quality, recall_at_k_refs, f1_score_refs

logger = logging.getLogger(__name__)


@dataclass
class WeightConfig:
    feature_weight:             float
    ruling_weight:              float
    article_weight:             float
    graph_support_weight:       float
    missing_channel_penalty:    float
    routing_uncertainty_weight: float
    ambiguity_penalty:          float


# Search grid — coarse on purpose, this is a first calibration pass on the
# retrieval-confidence layer only (see note in run_calibration.py).
_GRID = {
    "feature_weight":             [0.3, 0.4, 0.5],
    "ruling_weight":              [0.3, 0.4, 0.5],
    "graph_support_weight":       [0.0, 0.1, 0.15, 0.2],
    "missing_channel_penalty":    [0.1, 0.15, 0.2],
    "routing_uncertainty_weight": [0.1, 0.15, 0.2],
    "ambiguity_penalty":          [0.05, 0.1, 0.15],
}


def _generate_candidates() -> list[WeightConfig]:
    candidates = []
    for fw, rw, gsw, mcp, ruw, ap in itertools.product(
        _GRID["feature_weight"],
        _GRID["ruling_weight"],
        _GRID["graph_support_weight"],
        _GRID["missing_channel_penalty"],
        _GRID["routing_uncertainty_weight"],
        _GRID["ambiguity_penalty"],
    ):
        if fw + rw > 1.0:
            continue
        article_weight = round(1.0 - fw - rw, 3)
        candidates.append(WeightConfig(
            feature_weight=fw,
            ruling_weight=rw,
            article_weight=article_weight,
            graph_support_weight=gsw,
            missing_channel_penalty=mcp,
            routing_uncertainty_weight=ruw,
            ambiguity_penalty=ap,
        ))
    return candidates


def _apply_config(config: WeightConfig) -> None:
    """Monkey-patches the module-level constants confidence.py reads.
    Safe here because calibration runs single-threaded, offline, outside
    the web process."""
    confidence_module._CHANNEL_WEIGHTS = {
        "feature": config.feature_weight,
        "ruling":  config.ruling_weight,
        "article": config.article_weight,
    }
    confidence_module._GRAPH_SUPPORT_WEIGHT = config.graph_support_weight
    confidence_module._MISSING_CHANNEL_PENALTY = config.missing_channel_penalty
    confidence_module._ROUTING_UNCERTAINTY_WEIGHT = config.routing_uncertainty_weight
    confidence_module._AMBIGUITY_PENALTY = config.ambiguity_penalty


@dataclass
class RawSearchOutput:
    vector:              "confidence_module.UncertaintyVector"
    queried_channels:    list[str]
    routing_confidence:  float
    ambiguity_flag:      bool
    actual_recall:       float   # recall_at_k — primary metric (see debug session)
    actual_f1:           float   # kept alongside for comparison/analysis


def _collect_raw_outputs(samples: list[AnnotatedSample]) -> dict[str, RawSearchOutput]:
    """Runs search() exactly once per sample (expensive: LLM + embedding +
    Neo4j calls), caching the raw uncertainty vector so grid search over
    weight configs is pure in-memory recomputation afterward."""
    raw: dict[str, RawSearchOutput] = {}
    for sample in samples:
        try:
            result = search_service.search(sample.query)
        except Exception as e:
            logger.error(f"Search failed for ruling_id={sample.ruling_id}: {e}")
            continue

        recall = recall_at_k_refs(result.article_refs, sample.gold_articles)
        f1 = f1_score_refs(result.article_refs, sample.gold_articles)

        raw[sample.ruling_id] = RawSearchOutput(
            vector=result.confidence.vector,
            queried_channels=result.routing.channels,
            routing_confidence=result.routing.routing_confidence,
            ambiguity_flag=result.routing.ambiguity_flag,
            actual_recall=recall,
            actual_f1=f1,
        )
    return raw


def _score_config(
    config: WeightConfig,
    cached_raw_results: dict[str, RawSearchOutput],
    samples: list[AnnotatedSample],
    metric: str = "recall",
) -> float:
    """Recomputes confidence.score for each cached raw search output under
    a given weight config, without re-running search(). Returns Spearman
    correlation against the chosen metric ('recall' or 'f1')."""
    _apply_config(config)

    confidences, targets = [], []
    for sample in samples:
        raw = cached_raw_results.get(sample.ruling_id)
        if raw is None:
            continue
        conf_result = confidence_module.compute_confidence(
            raw.vector,
            queried_channels=raw.queried_channels,
            routing_confidence=raw.routing_confidence,
            ambiguity_flag=raw.ambiguity_flag,
        )
        confidences.append(conf_result.score)
        targets.append(raw.actual_recall if metric == "recall" else raw.actual_f1)

    if len(confidences) < 2:
        return -1.0
    return spearman_correlation(confidences, targets)


def calibrate(
    train_val_samples: list[AnnotatedSample],
    k: int = 5,
    metric: str = "recall",
) -> tuple[WeightConfig, dict[str, RawSearchOutput]]:
    """
    Stratified k-fold CV over train_val_samples. For each candidate weight
    config, averages validation-fold Spearman across folds; returns the
    config with the best average.

    NOTE: this calibrates confidence.py (retrieval-level confidence) only.
    It is a sanity-check step for this layer, not the final calibration
    of overall_uncertainty — that must be re-run once checklist/agent/
    citation signals exist, per the earlier project decision.

    metric: 'recall' (default) or 'f1'. Recall is preferred as the primary
    optimization target because F1 is structurally capped when top_k >
    len(gold_articles), which is common in this dataset (see debug session
    where F1 was capped at 0.33 despite correct retrieval).
    """
    logger.info("Collecting raw search outputs for all train_val samples (one search pass each)...")
    raw_outputs = _collect_raw_outputs(train_val_samples)

    folds = stratified_k_folds(train_val_samples, k=k)
    candidates = _generate_candidates()
    logger.info(f"Evaluating {len(candidates)} weight configs over {k} folds (metric={metric})...")

    best_config, best_avg_score = None, float("-inf")

    for config in candidates:
        fold_scores = []
        for train_fold, val_fold in folds:
            score = _score_config(config, raw_outputs, val_fold, metric=metric)
            fold_scores.append(score)

        avg_score = sum(fold_scores) / len(fold_scores)
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            best_config = config

    logger.info(f"Best config (avg Spearman={best_avg_score:.3f}, metric={metric}): {best_config}")
    return best_config, raw_outputs