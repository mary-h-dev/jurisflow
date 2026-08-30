"""
Conformal Prediction baseline for JurisFlow's article-retrieval component.

Method: LofreeCP-style split conformal prediction, adapted from
"API Is Enough: Conformal Prediction for LLMs Without Logit-Access"
(Su et al., arXiv:2403.01216), Section 3.3. We use the raw article_channel
retrieval score as the (logit-free) nonconformity signal instead of
sampling-frequency, since we query the retriever once rather than
resampling an LLM — so the paper's NE/SS fine-grained terms (which
require repeated sampling) do not apply here; this baseline is the
frequency/logit-free case by construction.

Two coverage variants are reported, because a query in this dataset can
have more than one gold article (unlike the paper's single-answer QA/MCQ
setting):
  - "all": the prediction set must contain EVERY gold article for that
    query. Aggregation for calibration: max(nonconformity) over golds.
  - "any": the prediction set must contain AT LEAST ONE gold article.
    Aggregation for calibration: min(nonconformity) over golds.
Both preserve the nesting property (paper Eq. 1) and therefore the
split-CP coverage guarantee (paper Proposition 3.2), just evaluated
against a different definition of "covered".

Reuses the project's existing calibration infrastructure instead of
duplicating it:
  - .data.load_case_annotations / .data.stratified_test_split
  - .cache_utils.load_or_collect
The 28-sample CP test set is NOT a separate file. It is regenerated
deterministically via
    stratified_test_split(load_case_annotations(case_grounded.json),
                           test_fraction=0.2, seed=42)
i.e. the exact same split calibrate_regression.py already uses, so
there is no risk of it drifting from "the test set" used elsewhere.
The 25-sample calibration set is a separate, untouched annotation file
in the same AnnotatedSample schema (ruling_id, case_type, query,
gold_articles, gold_routing, gold_checklist_by_article,
excluded_articles).

CACHING CAVEAT: the existing `test_140_no_routing` / `test_140_final` pkl
caches (built by apps.search.calibration.calibrate._collect_raw_outputs)
only store the AGGREGATE confidence.vector per sample — the raw
per-article article_channel candidate list is computed for recall/F1 and
then discarded, never cached. So those caches can't be reused as-is for
CP: `_collect_cp_raw_outputs` below does one extra search() pass over the
same 28 (and 25) samples, additionally keeping `result.article_results`
(the RAW, pre-RRF-fusion article_channel output — NOT `result.article_refs`,
which is fused with ruling-channel citations per services.py). Store its
output under a NEW cache key (e.g. "test_140_cp_candidates"), since the
pickled object shape differs from the existing cache.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

CoverMode = Literal["all", "any"]

# Must match services._LATIN_TO_PERSIAN exactly -- article_id below is
# built the same way services._build_article_refs builds article_refs,
# so it lines up 1:1 with the gold_articles annotation strings
# (e.g. "قانون مدنی - ماده ۱۱۲۳").
_LATIN_TO_PERSIAN = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


# ---------------------------------------------------------------------------
# 1. Candidate schema (this module's own, independent of your project's
#    internal raw-output class — kept intentionally minimal)
# ---------------------------------------------------------------------------

@dataclass
class RetrievedArticle:
    rank: int
    article_id: str
    raw_score: float
    normalized_score: float | None = None


@dataclass
class QueryCandidates:
    ruling_id: str
    retrieved: list[RetrievedArticle]


# ---------------------------------------------------------------------------
# 2. Raw-output collection: mirrors calibrate._collect_raw_outputs exactly
#    (same search() call, same leakage guard, same recall/F1 formulas),
#    but additionally keeps the raw article_channel candidate list, which
#    the existing function discards after computing recall/F1.
# ---------------------------------------------------------------------------

@dataclass
class CPRawOutput:
    """CP-specific counterpart of calibrate.RawSearchOutput. Keeps
    article_channel_candidates (needed for conformal prediction) in
    addition to recall/F1 (kept only so these runs stay comparable to
    the other calibration reports, TOP_K=20 must match article_channel's
    default retrieval size)."""
    article_channel_candidates: list[RetrievedArticle]
    actual_recall: float
    actual_f1: float


def _collect_cp_raw_outputs(samples: list) -> dict[str, CPRawOutput]:
    """
    Mirrors calibrate._collect_raw_outputs exactly (same search() call,
    same exclude_self_ruling leakage guard) with one addition: also keeps
    `result.article_results`, the RAW article_channel output BEFORE
    RRF fusion with the feature/ruling channels -- this is what a
    logit-free / fusion-free conformal-prediction baseline needs as its
    candidate pool. (`result.article_refs` is NOT used for the candidate
    pool: per services._build_article_refs it merges article_results with
    ruling-channel citations, i.e. it is already fused.)

    Recall/F1 are still computed on the FUSED `result.article_refs`, same
    as calibrate.py and calibrate_regression.py, so those two numbers
    stay directly comparable across all calibration scripts -- only the
    CP candidate pool itself is channel-raw.

    article_id is built exactly like services._build_article_refs does:
    "{law_name} - ماده {persian_digit_article_number}", so it matches the
    gold_articles annotation strings 1:1 (e.g. "قانون مدنی - ماده ۱۱۲۳").
    Evidence.article_number from Neo4j is a Latin-digit int; gold_articles
    uses Persian digits -- _LATIN_TO_PERSIAN bridges that.
    """
    from apps.search.services import search_service
    from ..metrics import recall_at_k_refs, f1_score_refs
    from ..leakage_guard import exclude_self_ruling

    raw: dict[str, CPRawOutput] = {}
    for sample in samples:
        try:
            result = search_service.search(sample.query)
        except Exception as e:
            logger.error(f"Search failed for ruling_id={sample.ruling_id}: {e}")
            continue

        result = exclude_self_ruling(result, sample.ruling_id)

        recall = recall_at_k_refs(result.article_refs, sample.gold_articles)
        f1 = f1_score_refs(result.article_refs, sample.gold_articles)

        candidates: list[RetrievedArticle] = []
        for rank, ev in enumerate(result.article_results, start=1):
            if not ev.law_name or not ev.article_number:
                continue  # same skip rule as services._build_article_refs
            article_id = f"{ev.law_name} - ماده {str(ev.article_number).translate(_LATIN_TO_PERSIAN)}"
            candidates.append(RetrievedArticle(rank=rank, article_id=article_id, raw_score=ev.score))

        raw[sample.ruling_id] = CPRawOutput(
            article_channel_candidates=candidates,
            actual_recall=recall,
            actual_f1=f1,
        )
    return raw


def _normalize_scores(candidates: list[RetrievedArticle]) -> None:
    """Min-max normalize raw_score within this query's own top-k pool -> [0, 1]."""
    if not candidates:
        return
    scores = [c.raw_score for c in candidates]
    lo, hi = min(scores), max(scores)
    span = hi - lo
    for c in candidates:
        c.normalized_score = 1.0 if span == 0 else (c.raw_score - lo) / span


def build_query_candidates(cp_raw_outputs: dict[str, CPRawOutput]) -> dict[str, QueryCandidates]:
    """cp_raw_outputs: {ruling_id: CPRawOutput}, e.g. what
    load_or_collect(cache_key, samples, _collect_cp_raw_outputs) returns."""
    out: dict[str, QueryCandidates] = {}
    for ruling_id, raw in cp_raw_outputs.items():
        candidates = list(raw.article_channel_candidates)
        _normalize_scores(candidates)
        out[ruling_id] = QueryCandidates(ruling_id=ruling_id, retrieved=candidates)
    return out


# ---------------------------------------------------------------------------
# 3. Nonconformity
# ---------------------------------------------------------------------------

def nonconformity(article: RetrievedArticle) -> float:
    """N(article) = 1 - normalized_score(article). Lower = more confident."""
    assert article.normalized_score is not None, "call _normalize_scores first"
    return 1.0 - article.normalized_score


def gold_nonconformity_scores(
    gold_ids: set[str], candidates: list[RetrievedArticle]
) -> list[float]:
    """N for each gold article; math.inf if a gold isn't in the top-k pool
    (mirrors the paper's rule in Sec. 3.3: true label absent from the
    response pool -> nonconformity score = infinity)."""
    pool = {c.article_id: c for c in candidates}
    return [nonconformity(pool[gid]) if gid in pool else math.inf for gid in gold_ids]


def per_query_calibration_score(
    gold_ids: set[str], candidates: list[RetrievedArticle], mode: CoverMode
) -> float:
    """
    Aggregate per-gold nonconformity scores into ONE score per calibration
    query (so the standard split-CP quantile step still applies and the
    coverage guarantee holds for the chosen mode):
      mode="all": max(N) over golds  -> threshold >= this covers ALL golds
      mode="any": min(N) over golds  -> threshold >= this covers AT LEAST ONE
    """
    scores = gold_nonconformity_scores(gold_ids, candidates)
    if not scores:
        return math.inf
    return max(scores) if mode == "all" else min(scores)


# ---------------------------------------------------------------------------
# 4. Calibration (paper Sec. 2, Step 3)
# ---------------------------------------------------------------------------

def compute_quantile(scores: list[float], alpha: float) -> float:
    """
    q_hat = the ceil((n+1)(1-alpha))-th smallest value among scores.
    If ceil((n+1)(1-alpha)) > n, the target coverage 1-alpha is NOT
    achievable with this calibration-set size -> return math.inf.

    NOTE: q_hat can ALSO come out to math.inf even when the target IS
    formula-achievable (k <= n), if enough calibration scores are
    themselves math.inf (i.e. the gold article wasn't in the top-k
    retrieval pool at all for many calibration samples). That is a
    DIFFERENT failure mode -- a retrieval-recall bottleneck, not a
    calibration-set-size problem -- see `is_alpha_achievable` below,
    which callers should check separately to tell the two apart when
    reporting results.
    """
    n = len(scores)
    if n == 0:
        raise ValueError("Empty calibration set.")
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return math.inf
    return sorted(scores)[k - 1]


def is_alpha_achievable(n_calib: int, alpha: float) -> bool:
    """True iff ceil((n_calib+1)(1-alpha)) <= n_calib, i.e. the split-CP
    quantile formula itself has a valid (non-inf-by-construction) answer
    for this calibration-set size. Does NOT guarantee q_hat != inf --
    that can still happen if too many calibration nonconformity scores
    are inf due to low retrieval recall (see compute_quantile docstring)."""
    return math.ceil((n_calib + 1) * (1 - alpha)) <= n_calib


def calibrate(
    calib_annotations: dict,       # {ruling_id: AnnotatedSample}
    calib_candidates: dict[str, QueryCandidates],
    alpha: float,
    mode: CoverMode,
) -> float:
    scores = []
    for ruling_id, ann in calib_annotations.items():
        qc = calib_candidates.get(ruling_id)
        if qc is None:
            raise KeyError(f"No retrieval candidates found for calibration sample {ruling_id}")
        gold_ids = set(ann.gold_articles)
        scores.append(per_query_calibration_score(gold_ids, qc.retrieved, mode))
    return compute_quantile(scores, alpha)


# ---------------------------------------------------------------------------
# 5. Prediction sets (paper Sec. 2, Step 4)
# ---------------------------------------------------------------------------

def prediction_set(candidates: list[RetrievedArticle], q_hat: float) -> list[RetrievedArticle]:
    return [c for c in candidates if nonconformity(c) <= q_hat]


# ---------------------------------------------------------------------------
# 6. Evaluation: ECR, SSC, APSS
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    mode: CoverMode
    alpha: float
    q_hat: float
    ecr: float                     # %
    apss: float
    ssc_by_size: dict[int, float]  # {set_size: coverage %} for reliable bins
    ssc_worst: float               # % — this is "SSC" as reported in the paper's tables
    n_test: int


def is_covered(gold_ids: set[str], pred_ids: set[str], mode: CoverMode) -> bool:
    if not gold_ids:
        return True
    return gold_ids.issubset(pred_ids) if mode == "all" else bool(gold_ids & pred_ids)


def evaluate(
    test_annotations: dict,        # {ruling_id: AnnotatedSample}
    test_candidates: dict[str, QueryCandidates],
    q_hat: float,
    mode: CoverMode,
    alpha: float,
    min_bin_frac: float = 0.10,    # ignore size-bins with <10% of test samples
) -> EvalResult:
    n = len(test_annotations)
    covered_flags: list[bool] = []
    sizes: list[int] = []
    size_to_covered: dict[int, list[bool]] = {}

    for ruling_id, ann in test_annotations.items():
        qc = test_candidates.get(ruling_id)
        if qc is None:
            raise KeyError(f"No retrieval candidates found for test sample {ruling_id}")
        gold_ids = set(ann.gold_articles)
        pred = prediction_set(qc.retrieved, q_hat)
        pred_ids = {c.article_id for c in pred}

        covered = is_covered(gold_ids, pred_ids, mode)
        covered_flags.append(covered)
        sizes.append(len(pred))
        size_to_covered.setdefault(len(pred), []).append(covered)

    ecr = sum(covered_flags) / n * 100
    apss = sum(sizes) / n

    ssc_by_size = {
        size: sum(flags) / len(flags) * 100
        for size, flags in size_to_covered.items()
        if size > 0 and len(flags) / n >= min_bin_frac
    }
    ssc_worst = min(ssc_by_size.values()) if ssc_by_size else float("nan")

    return EvalResult(mode=mode, alpha=alpha, q_hat=q_hat, ecr=ecr, apss=apss,
                       ssc_by_size=ssc_by_size, ssc_worst=ssc_worst, n_test=n)


# ---------------------------------------------------------------------------
# 7. AUROC & ECE — properties of the scoring function itself, independent
#    of the conformal threshold / coverage mode.
# ---------------------------------------------------------------------------

def flatten_candidates(
    annotations: dict, candidates: dict[str, QueryCandidates]
) -> tuple[list[float], list[int]]:
    """score = normalized_score; label = 1 if candidate is gold else 0,
    pooled over every (query, candidate) pair."""
    scores: list[float] = []
    labels: list[int] = []
    for ruling_id, ann in annotations.items():
        qc = candidates.get(ruling_id)
        if qc is None:
            continue
        gold_ids = set(ann.gold_articles)
        for c in qc.retrieved:
            scores.append(c.normalized_score)
            labels.append(1 if c.article_id in gold_ids else 0)
    return scores, labels


def compute_auroc(scores: list[float], labels: list[int]) -> float:
    from sklearn.metrics import roc_auc_score
    if len(set(labels)) < 2:
        return float("nan")
    return roc_auc_score(labels, scores)


def compute_ece(scores: list[float], labels: list[int], n_bins: int = 10) -> float:
    """Standard Expected Calibration Error (%): bin normalized_score into
    n_bins equal-width bins, compare mean confidence vs empirical accuracy
    per bin, weighted by bin size."""
    import numpy as np
    scores_arr = np.asarray(scores)
    labels_arr = np.asarray(labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(scores_arr)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (scores_arr >= lo) & (scores_arr <= hi if i == n_bins - 1 else scores_arr < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(scores_arr[mask].mean() - labels_arr[mask].mean())
    return ece * 100


# ---------------------------------------------------------------------------
# 8. Runner — wires the project's own loaders/cache into the module above
# ---------------------------------------------------------------------------

# Paths — adjust if these differ from the actual locations.
_FULL_ANNOTATION_PATH = "apps/search/calibration/data/annotations/case_grounded.json"
_CALIB_ANNOTATION_PATH = "apps/search/calibration/data/annotations/conformal_calibration_25.json"
_CALIB_CACHE_KEY = "conformal_calibration_25_cp_candidates"
_TEST_CACHE_KEY = "test_140_cp_candidates"  # NEW key: see module docstring re: caching caveat
_ALPHAS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]  # widened to probe where
                                                              # (if anywhere) prediction
                                                              # sets shrink below top-k


def run_conformal_baseline(
    calibration_annotation_path: str,
    full_annotation_path: str,
    calib_cache_key: str,
    test_cache_key: str,
    alphas: list[float],
) -> dict:
    from ..data import load_case_annotations, stratified_test_split
    from ..cache_utils import load_or_collect

    calib_samples = load_case_annotations(calibration_annotation_path)
    full_samples = load_case_annotations(full_annotation_path)
    _, test_samples = stratified_test_split(full_samples, test_fraction=0.2, seed=42)

    calib_raw = load_or_collect(calib_cache_key, calib_samples, _collect_cp_raw_outputs)
    test_raw = load_or_collect(test_cache_key, test_samples, _collect_cp_raw_outputs)

    calib_candidates = build_query_candidates(calib_raw)
    test_candidates = build_query_candidates(test_raw)

    calib_ann = {s.ruling_id: s for s in calib_samples}
    test_ann = {s.ruling_id: s for s in test_samples}

    results: dict = {"all": {}, "any": {}}
    for mode in ("all", "any"):
        for alpha in alphas:
            q_hat = calibrate(calib_ann, calib_candidates, alpha, mode)
            results[mode][alpha] = evaluate(test_ann, test_candidates, q_hat, mode, alpha)

    test_scores, test_labels = flatten_candidates(test_ann, test_candidates)
    results["_scoring_function"] = {
        "auroc": compute_auroc(test_scores, test_labels),
        "ece": compute_ece(test_scores, test_labels),
    }
    results["_n_calib"] = len(calib_ann)
    return results


def _print_results(results: dict) -> None:
    n_calib = results["_n_calib"]
    for mode in ("all", "any"):
        print(f"\n=== coverage mode = {mode} ===")
        for alpha, res in results[mode].items():
            if res.q_hat != math.inf:
                q_str = f"{res.q_hat:.4f}"
            elif not is_alpha_achievable(n_calib, alpha):
                # Formula itself has no valid answer at this n_calib -- need
                # more calibration samples to target this alpha at all.
                q_str = f"inf (alpha not achievable at n_calib={n_calib})"
            else:
                # Formula IS answerable, but resolves to inf anyway because
                # too many calibration golds are missing from the top-k
                # retrieval pool entirely -- a RECALL bottleneck, not a
                # calibration-size problem. Prediction set = full pool.
                q_str = "inf (retrieval-recall bottleneck, not calib size)"
            print(f"alpha={alpha:.2f}  q_hat={q_str:<48}  "
                  f"ECR={res.ecr:5.1f}%  APSS={res.apss:5.2f}  SSC={res.ssc_worst:5.1f}%")

    sf = results["_scoring_function"]
    print("\n=== scoring function (mode/alpha independent, test pool) ===")
    print(f"AUROC={sf['auroc']:.4f}   ECE={sf['ece']:.2f}%")


def run():
    """Entry point matching the sibling scripts' convention (see
    calibrate_regression.run()). Invoke with:

        DJANGO_SETTINGS_MODULE=config.settings.development python -c \\
            "import django; django.setup(); \\
             from apps.search.calibration.baseline.calibrate_conformal import run; run()"
    """
    results = run_conformal_baseline(
        calibration_annotation_path=_CALIB_ANNOTATION_PATH,
        full_annotation_path=_FULL_ANNOTATION_PATH,
        calib_cache_key=_CALIB_CACHE_KEY,
        test_cache_key=_TEST_CACHE_KEY,
        alphas=_ALPHAS,
    )
    _print_results(results)


if __name__ == "__main__":
    import django
    django.setup()
    run()