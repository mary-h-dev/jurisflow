from __future__ import annotations

import logging

import requests
from django.conf import settings

logger = logging.getLogger(__name__)


def embed_text(text: str) -> list[float]:
    """
    Dev  -> Ollama (local, no rate limit)
    Prod -> Gemini (gemini-embedding-001)
    """
    if settings.DEBUG:
        return _embed_ollama(text)
    return _embed_gemini(text)


def _embed_ollama(text: str) -> list[float]:
    print(f"[embed_text] len={len(text)} chars | preview: {text[:150]!r}")
    try:
        response = requests.post(
            f"{settings.OLLAMA_URL}/api/embeddings",
            json={"model": settings.OLLAMA_MODEL, "prompt": text[:2000]},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("embedding") or data.get("embeddings", [[]])[0]
    except Exception as e:
        logger.error(f"Ollama embedding failed: {e}")
        raise



def _embed_gemini(text: str) -> list[float]:
    try:
        from google import genai
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        result = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=text[:2000],
        )
        return result.embeddings[0].values
    except Exception as e:
        logger.error(f"Gemini embedding failed: {e}")
        raise