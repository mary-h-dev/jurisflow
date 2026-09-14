"""
Runtime configuration for the legal-agents deliberation LLM calls.
Only defines this app's ModelConfig instances -- the client itself lives in
core/llm_client.py, shared with apps.auditor and apps.search.
"""

from __future__ import annotations

from core.llm_client import ModelConfig, default_extra_headers

_GEMINI_FLASH = "google/gemini-3.6-flash"

# Defender and prosecutor -- slight creativity headroom for argumentation
DELIBERATION_MODEL = ModelConfig(
    model         = _GEMINI_FLASH,
    temperature   = 0.2,
    max_tokens    = 4096,
    extra_headers = default_extra_headers(),
)

# Judge -- lower temperature, more deterministic ruling
JUDGE_MODEL = ModelConfig(
    model         = _GEMINI_FLASH,
    temperature   = 0.1,
    max_tokens    = 4096,
    extra_headers = default_extra_headers(),
)