from __future__ import annotations

import logging
from functools import lru_cache

from .router import RoutingResult, route_query

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1024)
def route_query_cached(query: str) -> RoutingResult:
    """
    Caches route_query() by exact query text, within this process. The
    router's rewrite LLM call is not fully deterministic even at
    temperature=0 (provider-side batching/floating-point variance), which
    makes repeated eval runs on the same annotation set non-reproducible
    -- two runs of the same 100-sample eval can silently score
    differently just from rewrite drift, with no code change in between.

    This does not fix the LLM's non-determinism; it pins ONE rewrite per
    query for the lifetime of the process, so a given eval run (or a
    single debug script invocation that calls search() more than once on
    the same query) is internally consistent. A fresh process still gets
    a fresh rewrite -- this is not a persisted/cross-run cache.

    Use this in place of route_query() in services.py once confirmed
    working, or import it directly in eval/calibration scripts that need
    repeatable results without touching the production search path yet.
    """
    result = route_query(query)
    logger.debug(f"route_query_cached: cache miss, computed fresh rewrite for query")
    return result