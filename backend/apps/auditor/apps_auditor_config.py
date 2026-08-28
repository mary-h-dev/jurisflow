"""
Runtime configuration for the Auditor's LLM calls.
Only defines this app's ModelConfig -- the client itself lives in
core/llm_client.py, shared with apps.search and apps.legal_agents.
"""

from __future__ import annotations

from core.llm_client import ModelConfig, default_extra_headers

_GEMINI_FLASH = "google/gemini-3.6-flash"

# max_tokens=4096: 1024 was observed to truncate mid-JSON on real runs
# (reasoning-token overhead eating the visible-output budget). A
# multi-item checklist plus that overhead needs meaningfully more room.
AUDITOR_MODEL = ModelConfig(
    model=_GEMINI_FLASH,
    temperature=0.0,
    max_tokens=4096,
    extra_headers=default_extra_headers(),
)