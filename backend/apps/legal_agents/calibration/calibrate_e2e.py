"""
End-to-end calibration: runs the full pipeline (retrieval → auditor →
deliberation agents) on every split and saves final_confidence per sample.

Outputs feed two downstream scripts:
  - calibrate_fusion.py  (uses train_val to fit fusion weights)
  - conformal prediction (uses calibration_25 + test for ECR/APSS/SSC)

Splits:
  - "calibration"  25 samples  conformal_calibration_25.json
  - "train_val"   112 samples  case_grounded.json (stratified 80%)
  - "test"         28 samples  case_grounded.json (stratified 20%)

Coverage modes (both stored per sample):
  - "any": prediction set covers AT LEAST ONE gold article
  - "all": prediction set covers ALL gold articles

Per-sample caching: each sample is cached individually (pickle) so a
partial run can be resumed without re-spending API budget.

Run (dry-run first):
    DJANGO_SETTINGS_MODULE=config.settings.development python -c "
    import django; django.setup()
    from apps.legal_agents.calibration.calibrate_e2e import dry_run; dry_run()"

Run (one split at a time — recommended order: calibration → test → train_val):
    DJANGO_SETTINGS_MODULE=config.settings.development python -c "
    import django; django.setup()
    from apps.legal_agents.calibration.calibrate_e2e import run; run('calibration')"

DJANGO_SETTINGS_MODULE=config.settings.development python -c "from apps.legal_agents.calibration.calibrate_e2e import run; run('calibration')"

DJANGO_SETTINGS_MODULE=config.settings.development python -c "from apps.legal_agents.calibration.calibrate_e2e import run; run('test')"


DJANGO_SETTINGS_MODULE=config.settings.development python -c "from apps.legal_agents.calibration.calibrate_e2e import run; run('train_val')"


"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.metrics import roc_auc_score

from apps.search.calibration.data import load_case_annotations, stratified_test_split
from apps.search.calibration.leakage_guard import exclude_self_ruling

logger = logging.getLogger(__name__)

_FULL_ANNOTATION_PATH  = "apps/search/calibration/data/annotations/case_grounded.json"
_CALIB_ANNOTATION_PATH = "apps/search/calibration/data/annotations/conformal_calibration_25.json"

_DATA_DIR  = Path(__file__).parent / "data"
_CACHE_DIR = _DATA_DIR / "e2e_cache"

Split = Literal["calibration", "train_val", "test"]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(ruling_id: str, split: str) -> str:
    payload = json.dumps({"ruling_id": ruling_id, "split": split}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_path(ruling_id: str, split: str) -> Path:
    return _CACHE_DIR / f"{_cache_key(ruling_id, split)}.pkl"


def _load_cache(ruling_id: str, split: str) -> dict | None:
    path = _cache_path(ruling_id, split)
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def _save_cache(ruling_id: str, split: str, data: dict) -> None:
    with _cache_path(ruling_id, split).open("wb") as f:
        pickle.dump(data, f)


# ---------------------------------------------------------------------------
# Correctness — both modes
# ---------------------------------------------------------------------------

def _correctness_any(gold_articles: list[str], fusion) -> int:
    """1 if AT LEAST ONE gold article is in agreed_articles (cited by >=2 agents)."""
    if fusion is None:
        return 0
    return int(bool(set(gold_articles) & set(fusion.agreed_articles)))


def _correctness_all(gold_articles: list[str], fusion) -> int:
    """1 if ALL gold articles are in agreed_articles (cited by >=2 agents)."""
    if fusion is None:
        return 0
    return int(set(gold_articles).issubset(set(fusion.agreed_articles)))


def _article_prf(gold_articles: list[str], fusion) -> tuple[float, float, float]:
    """Precision/Recall/F1 at article level (predicted = all_cited_articles)."""
    if fusion is None:
        return 0.0, 0.0, 0.0
    gold      = set(gold_articles)
    predicted = set(fusion.all_cited_articles)
    if not predicted:
        return 0.0, 0.0, 0.0
    tp   = len(gold & predicted)
    prec = tp / len(predicted)
    rec  = tp / len(gold) if gold else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return round(prec, 4), round(rec, 4), round(f1, 4)


# ---------------------------------------------------------------------------
# ECE
# ---------------------------------------------------------------------------

def _ece(confidences: list[float], labels: list[int], n_bins: int = 5) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    confs = np.array(confidences)
    labs  = np.array(labels, dtype=float)
    bins  = np.linspace(0.0, 1.0, n_bins + 1)
    ece   = 0.0
    n     = len(confs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(labs[mask].mean() - confs[mask].mean())
    return round(float(ece), 4)


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def _aggregate(rows: list[dict], split: str) -> dict:
    valid = [r for r in rows if "final_confidence" in r and "error" not in r]
    if not valid:
        return {"split": split, "n": 0}

    confs       = [r["final_confidence"] for r in valid]
    correct_any = [r["correct_any"] for r in valid]
    correct_all = [r["correct_all"] for r in valid]
    precs       = [r["precision"] for r in valid]
    recs        = [r["recall"] for r in valid]
    f1s         = [r["f1"] for r in valid]
    consensuses = [r["consensus"] for r in valid]

    def safe_auroc(labels: list[int]) -> float | None:
        return (round(roc_auc_score(labels, confs), 4)
                if len(set(labels)) > 1 else None)

    return {
        "split":   split,
        "n":       len(valid),
        "n_errors": len(rows) - len(valid),

        # Coverage any
        "auroc_any":    safe_auroc(correct_any),
        "ece_any":      _ece(confs, correct_any),
        "accuracy_any": round(sum(correct_any) / len(correct_any), 4),

        # Coverage all
        "auroc_all":    safe_auroc(correct_all),
        "ece_all":      _ece(confs, correct_all),
        "accuracy_all": round(sum(correct_all) / len(correct_all), 4),

        # Article-level
        "mean_precision": round(float(np.mean(precs)), 4),
        "mean_recall":    round(float(np.mean(recs)), 4),
        "mean_f1":        round(float(np.mean(f1s)), 4),

        # Consensus
        "consensus_rate":       round(
            sum(1 for c in consensuses if c in ("full", "majority")) / len(consensuses), 4),
        "full_consensus_rate":  round(
            sum(1 for c in consensuses if c == "full") / len(consensuses), 4),
        "split_rate":           round(
            sum(1 for c in consensuses if c == "split") / len(consensuses), 4),

        # Confidence distribution
        "mean_confidence":      round(float(np.mean(confs)), 4),
        "std_confidence":       round(float(np.std(confs)), 4),
        "uncertainty_flag_rate": round(
            sum(1 for r in valid if r.get("uncertainty_flag")) / len(valid), 4),

        # Gold coverage by auditor (before agents)
        "gold_in_applicable_rate": round(
            sum(r.get("gold_in_applicable", 0) for r in valid) / len(valid), 4),
    }


def _print_metrics(metrics: dict) -> None:
    print(f"\n{'='*60}")
    print(f"E2E [{metrics['split'].upper()}] — {metrics['n']} samples"
          f"  (errors: {metrics.get('n_errors', 0)})")
    print(f"{'='*60}")
    print(f"  ── Coverage (any: >=1 gold in agreed_articles) ──────")
    print(f"  AUROC      : {metrics.get('auroc_any')}")
    print(f"  ECE        : {metrics.get('ece_any')}")
    print(f"  Accuracy   : {metrics.get('accuracy_any')}")
    print(f"\n  ── Coverage (all: ALL golds in agreed_articles) ─────")
    print(f"  AUROC      : {metrics.get('auroc_all')}")
    print(f"  ECE        : {metrics.get('ece_all')}")
    print(f"  Accuracy   : {metrics.get('accuracy_all')}")
    print(f"\n  ── Article-level ────────────────────────────────────")
    print(f"  Precision  : {metrics.get('mean_precision')}")
    print(f"  Recall     : {metrics.get('mean_recall')}")
    print(f"  F1         : {metrics.get('mean_f1')}")
    print(f"\n  ── Consensus ────────────────────────────────────────")
    print(f"  Rate (full+majority) : {metrics.get('consensus_rate')}")
    print(f"  Full consensus       : {metrics.get('full_consensus_rate')}")
    print(f"  Split rate           : {metrics.get('split_rate')}")
    print(f"  Uncertainty flags    : {metrics.get('uncertainty_flag_rate')}")
    print(f"\n  ── Confidence distribution ──────────────────────────")
    print(f"  Mean +/- std : {metrics.get('mean_confidence')} +/- {metrics.get('std_confidence')}")
    print(f"\n  ── Auditor quality ──────────────────────────────────")
    print(f"  Gold kept by auditor : {metrics.get('gold_in_applicable_rate')}")


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def dry_run() -> None:
    samples         = load_case_annotations(_FULL_ANNOTATION_PATH)
    train_val, test = stratified_test_split(samples, test_fraction=0.2)
    calib           = load_case_annotations(_CALIB_ANNOTATION_PATH)

    print("=" * 60)
    print("DRY RUN — no API calls made")
    print("=" * 60)
    for label, subset in [
        ("calibration", calib),
        ("test",        test),
        ("train_val",   train_val),
    ]:
        cached = sum(1 for s in subset if _cache_path(s.ruling_id, label).exists())
        remain = len(subset) - cached
        calls  = remain * 8
        cost   = calls * 0.002
        print(f"  {label:<12}: {len(subset):>3} samples  "
              f"cached={cached:>3}  remaining={remain:>3}  "
              f"~{calls:>4} calls  ~${cost:.2f}")
    print("\nRecommended order: calibration → test → train_val")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(split: Split = "test") -> None:
    from apps.search.services import search_service
    from apps.auditor.services import auditor_service
    from apps.legal_agents.services import deliberation_service

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _DATA_DIR / f"e2e_{split}.json"

    # Load samples for this split
    if split == "calibration":
        samples = load_case_annotations(_CALIB_ANNOTATION_PATH)
    else:
        all_samples     = load_case_annotations(_FULL_ANNOTATION_PATH)
        train_val, test = stratified_test_split(all_samples, test_fraction=0.2)
        samples         = train_val if split == "train_val" else test

    cached_count = sum(1 for s in samples if _cache_path(s.ruling_id, split).exists())
    remaining    = len(samples) - cached_count

    print(f"\n{'='*60}")
    print(f"calibrate_e2e [{split}]")
    print(f"{'='*60}")
    print(f"  Total    : {len(samples)} samples")
    print(f"  Cached   : {cached_count}  (no API cost)")
    print(f"  Remaining: {remaining}  (~{remaining * 8} API calls  ~${remaining * 8 * 0.002:.2f})")
    print(f"  Output   : {out_path}\n")

    rows:       list[dict] = []
    cache_hits: int        = 0
    t_start                = time.monotonic()

    for i, sample in enumerate(samples, 1):
        # Progress summary every 10 samples
        if i > 1 and (i - 1) % 10 == 0:
            elapsed = time.monotonic() - t_start
            done    = i - 1
            rate    = done / elapsed if elapsed > 0 else 0
            eta     = (len(samples) - done) / rate if rate > 0 else 0
            print(f"\n  ── {done}/{len(samples)} done  "
                  f"elapsed={elapsed:.0f}s  ETA≈{eta:.0f}s ──\n")

        print(f"[{i:>3}/{len(samples)}] ruling={sample.ruling_id} ...",
              end=" ", flush=True)

        # Cache check
        cached = _load_cache(sample.ruling_id, split)
        if cached is not None:
            cache_hits += 1
            rows.append(cached)
            print(f"[cache]  conf={cached.get('final_confidence', '?'):.2f}  "
                  f"any={cached.get('correct_any', '?')}  "
                  f"all={cached.get('correct_all', '?')}")
            continue

        t0  = time.monotonic()
        row: dict = {
            "ruling_id":     sample.ruling_id,
            "case_type":     getattr(sample, "case_type", ""),
            "gold_articles": sample.gold_articles,
        }

        try:
            # Layer 1 — Retrieval
            print("retrieval...", end=" ", flush=True)
            search_raw    = search_service.search(sample.query)
            search_result = exclude_self_ruling(search_raw, sample.ruling_id)
            row["retrieval_confidence"]   = round(search_result.confidence.score, 4)
            row["retrieval_article_refs"] = search_result.article_refs
            row["retrieval_n_articles"]   = len(search_result.article_refs)
            print(f"conf={row['retrieval_confidence']:.2f} "
                  f"n={row['retrieval_n_articles']}",
                  end="  ", flush=True)

            # Layer 2 — Auditor
            print("auditor...", end=" ", flush=True)
            auditor_out = auditor_service.audit(search_result)
            applicable  = [a for a in auditor_out.verified_articles if a.is_applicable]
            gold_in_app = int(bool(
                set(sample.gold_articles) & {a.article_ref for a in applicable}
            ))
            row["auditor_confidence"]  = round(auditor_out.confidence.score, 4)
            row["auditor_level"]       = auditor_out.confidence.level
            row["prune_ratio"]         = round(auditor_out.confidence.prune_ratio, 4)
            row["n_applicable"]        = len(applicable)
            row["n_pruned"]            = len(auditor_out.pruned_articles)
            row["applicable_articles"] = [a.article_ref for a in applicable]
            row["gold_in_applicable"]  = gold_in_app
            print(f"conf={row['auditor_confidence']:.2f} "
                  f"applicable={row['n_applicable']} "
                  f"gold_kept={gold_in_app}",
                  end="  ", flush=True)

            # Layer 3 — Deliberation agents
            print("agents...", end=" ", flush=True)
            result = deliberation_service.run(
                auditor_out,
                session_id=f"e2e-{split}-{sample.ruling_id}",
            )
            fusion = result.fusion
            conf   = fusion.final_confidence if fusion else 0.5

            c_any        = _correctness_any(sample.gold_articles, fusion)
            c_all        = _correctness_all(sample.gold_articles, fusion)
            prec, rec, f1 = _article_prf(sample.gold_articles, fusion)

            row.update({
                "final_confidence":  round(conf, 4),
                "final_level":       fusion.final_level if fusion else None,
                "correct_any":       c_any,
                "correct_all":       c_all,
                "majority_verdict":  fusion.majority_verdict.value if fusion else None,
                "consensus":         fusion.consensus_level.value if fusion else None,
                "verdict_spread":    fusion.verdict_spread if fusion else {},
                "agreed_articles":   fusion.agreed_articles if fusion else [],
                "all_cited":         fusion.all_cited_articles if fusion else [],
                "uncertainty_flag":  fusion.uncertainty_flag if fusion else None,
                "agent_conf_mean":   fusion.agent_confidence_mean if fusion else None,
                "agent_conf_std":    fusion.agent_confidence_std if fusion else None,
                "prior_confidence":  fusion.prior_confidence if fusion else None,
                "precision":         prec,
                "recall":            rec,
                "f1":                f1,
                "elapsed_s":         round(time.monotonic() - t0, 1),
                "agent_error":       result.error,
            })

            _save_cache(sample.ruling_id, split, row)

            status = "✓" if c_any else "✗"
            print(f"{status}  conf={conf:.2f}  "
                  f"any={c_any}  all={c_all}  "
                  f"consensus={row['consensus']}  "
                  f"({row['elapsed_s']}s)")

        except Exception as exc:
            elapsed = time.monotonic() - t0
            print(f"ERROR ({elapsed:.1f}s): {exc}")
            logger.error("[e2e] ruling=%s error=%s", sample.ruling_id, exc, exc_info=True)
            row.update({
                "final_confidence": 0.5,
                "correct_any":      0,
                "correct_all":      0,
                "precision":        0.0,
                "recall":           0.0,
                "f1":               0.0,
                "elapsed_s":        round(elapsed, 1),
                "error":            str(exc),
            })

        rows.append(row)

    # Aggregate & save
    metrics = _aggregate(rows, split)
    _print_metrics(metrics)

    total_elapsed = time.monotonic() - t_start
    print(f"\n  Total elapsed : {total_elapsed:.0f}s  "
          f"(cache hits: {cache_hits}/{len(samples)})")

    output = {"split": split, "metrics": metrics, "per_sample": rows}
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  Saved → {out_path}")
