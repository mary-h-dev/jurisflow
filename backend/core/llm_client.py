"""
Shared LLM client for every app that calls an LLM through OpenRouter:
apps.search's router, apps.auditor's verifier, and apps.legal_agents'
deliberation agents.

One client, one place to configure retries/base_url/timeouts -- each
app still defines its OWN ModelConfig instances (different model,
temperature, max_tokens per role), just importing ModelConfig and
get_client from here instead of keeping three near-identical copies.

Django settings required:
    OPENROUTER_API_KEY
Optional:
    OPENROUTER_BASE_URL    (default: https://openrouter.ai/api/v1)
    APP_SITE_URL, APP_NAME (attribution headers for OpenRouter leaderboard)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

from django.conf import settings
from openai import OpenAI


@dataclass(frozen=True)
class ModelConfig:
    model:         str
    temperature:   float
    max_tokens:    int = 1024
    extra_headers: dict[str, str] = field(default_factory=dict)


def default_extra_headers() -> dict[str, str]:
    """Optional OpenRouter leaderboard attribution headers, shared by
    every app's ModelConfig so they don't each rebuild this dict."""
    return {
        k: v for k, v in {
            "HTTP-Referer": getattr(settings, "APP_SITE_URL", ""),
            "X-Title":      getattr(settings, "APP_NAME", ""),
        }.items() if v
    }


class LLMClient:
    """OpenAI-compatible client -- pointed at OpenRouter."""

    def __init__(self, api_key: str, base_url: str) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, model: str, temperature: float, messages: list[dict], max_tokens: int = 1024) -> str:
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()


_client: LLMClient | None = None
_lock = Lock()


def get_client() -> LLMClient:
    """
    Lazily builds the single shared client on first use, reading
    settings.OPENROUTER_API_KEY directly. No AppConfig.ready() hook
    required in any app -- that per-app init pattern was the actual
    source of the "file not found" / import-order errors hit earlier
    (three separate apps.py files, each racing to construct an
    equivalent client from the same settings). A double-checked lock
    keeps this safe if two threads call get_client() concurrently
    before the first client exists.
    """
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = LLMClient(
                    api_key=settings.OPENROUTER_API_KEY,
                    base_url=getattr(
                        settings, "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
                    ),
                )
    return _client