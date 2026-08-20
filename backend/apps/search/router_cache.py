from __future__ import annotations

import dataclasses
import json
import logging
from functools import lru_cache
from pathlib import Path
from threading import Lock

from .router import RoutingResult, route_query

logger = logging.getLogger(__name__)

_CACHE_PATH = Path(__file__).resolve().parent / "_router_cache.json"
_file_lock = Lock()


def _load_disk_cache() -> dict[str, dict]:
    if not _CACHE_PATH.exists():
        return {}
    try:
        return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"router disk cache unreadable ({e}), starting fresh")
        return {}


_disk_cache: dict[str, dict] = _load_disk_cache()


def _save_disk_cache() -> None:
    try:
        _CACHE_PATH.write_text(
            json.dumps(_disk_cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as e:
        logger.warning(f"could not persist router disk cache: {e}")


@lru_cache(maxsize=1024)
def route_query_cached(query: str) -> RoutingResult:
    """
    Caches route_query() by exact query text, backed by a JSON file
    (_router_cache.json next to this module) in addition to the
    in-memory lru_cache. The router's rewrite LLM call is not fully
    deterministic even at temperature=0 (provider-side batching/
    floating-point variance), which makes eval non-reproducible not
    just across repeated calls within one process, but ACROSS SEPARATE
    script invocations -- every `python -c "..."` debug run starts a
    fresh process with an empty in-memory cache, so a fresh rewrite (and
    therefore a possibly different retrieval set) was happening every
    single run despite the earlier in-memory-only cache.

    First hit for a query computes and persists to disk; every
    subsequent call, in this process or a new one, reads the same
    rewrite back. To force a fresh rewrite for a query (e.g. after a
    prompt change in router.py), delete its entry from
    _router_cache.json or delete the file entirely.
    """
    with _file_lock:
        cached = _disk_cache.get(query)
    if cached is not None:
        logger.debug("route_query_cached: disk cache hit")
        return RoutingResult(**cached)

    result = route_query(query)
    logger.debug("route_query_cached: cache miss, computed fresh rewrite for query")

    with _file_lock:
        _disk_cache[query] = dataclasses.asdict(result)
        _save_disk_cache()

    return result