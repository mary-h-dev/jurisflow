"""
Simple pickle-based cache for raw search outputs, to avoid re-hitting
Groq/embedding APIs every time calibration weights or thresholds are
tuned. Collection (search_service.search per sample) is the expensive,
API-costing step; everything downstream (weight fitting, threshold
choice, metric computation) is pure in-memory math and should never
re-trigger it.

Usage:
    from apps.search.calibration.cache_utils import load_or_collect

    raw_outputs = load_or_collect(
        cache_key="raw_outputs_150",
        samples=train_val,
        collect_fn=_collect_raw_outputs,
    )
"""

from __future__ import annotations

import pickle
from pathlib import Path

_CACHE_DIR = Path(__file__).resolve().parent / "cache"
_CACHE_DIR.mkdir(exist_ok=True)


def load_or_collect(cache_key: str, samples: list, collect_fn, force_refresh: bool = False):
    """
    Loads cached raw outputs if present, otherwise runs collect_fn(samples)
    (which hits Groq + embedding APIs) and caches the result.

    Set force_refresh=True only when you've changed retrieval code itself
    (e.g. router prompt, feature matcher, PMI formula) — a weight/threshold
    change alone does NOT require this.
    """
    cache_path = _CACHE_DIR / f"{cache_key}.pkl"

    if cache_path.exists() and not force_refresh:
        print(f"[cache] Loading cached raw outputs from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"[cache] No cache found (or force_refresh=True) — collecting fresh "
          f"outputs for {len(samples)} samples (this WILL call Groq/embedding APIs)...")
    raw_outputs = collect_fn(samples)

    with open(cache_path, "wb") as f:
        pickle.dump(raw_outputs, f)
    print(f"[cache] Saved to {cache_path}")

    return raw_outputs