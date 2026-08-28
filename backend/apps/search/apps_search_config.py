"""
Runtime configuration for the search app's router LLM call.
Only defines this app's ModelConfig -- the client itself lives in
core/llm_client.py, shared with apps.auditor and apps.legal_agents.
"""

from __future__ import annotations

from core.llm_client import ModelConfig, default_extra_headers

_GEMINI_FLASH = "google/gemini-3.6-flash"

# temperature=0.0: routing decisions (channel selection, rewrite) should
# be as stable as possible run-to-run -- see router_cache.py for why
# this alone isn't a full determinism guarantee and the disk cache
# exists on top of it.
# max_tokens=2048: safety margin against reasoning-token overhead
# eating the visible-output budget (observed truncation at 1024).
ROUTER_MODEL = ModelConfig(
    model=_GEMINI_FLASH,
    temperature=0.0,
    max_tokens=2048,
    extra_headers=default_extra_headers(),
)