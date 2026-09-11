"""
common/embedder.py — embedding with bge-m3 via Ollama (local)

Why Ollama and not the native version (FlagEmbedding)?
    The native version kept OOM-crashing on a memory-constrained
    environment (a Codespace with 9GB RAM). Ollama runs as a separate
    server with its own memory management and runs more stably — at the
    cost of slightly lower accuracy (since its model is quantized). This
    is a deliberate decision to get past the current constraint, not a
    final one — if more resources become available later, it's possible
    to go back to the native version.

    ⚠️ Important note for later: embeddings from these two models are
    *not compatible* with each other (different vector spaces). If the
    model is ever switched, all previous embeddings (Article/Note as well
    as Ruling/RulingSection) must be regenerated from scratch — they can't
    be mixed.

Prerequisite: the Ollama service must already be running and the model
must be pulled:
    ollama pull bge-m3

Important note about text length (second version): the first version only
worked with a single large character cap (max_length × 4) and sent
everything in one request — this caused the Ollama server itself to crash
with a 500 error on very long ruling sections (several thousand words),
because Ollama's actual serving context is usually configured smaller
than the model's theoretical capacity (8192). Solution: instead of
truncating or sending everything at once, long text is split into safe
chunks, each is embedded separately, and the vectors are averaged — this
way no content is lost and the request never exceeds Ollama's safe limit.
"""

import time
import requests

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "bge-m3"

# Default safety cap (when max_length isn't specified)
_DEFAULT_SAFETY_CHAR_CAP = 20000

# Rough character-to-token ratio estimate for Persian
_CHARS_PER_TOKEN_ESTIMATE = 4

# Maximum size of *each chunk* sent to Ollama in a single request.
# This number is deliberately conservative and independent of the
# caller's max_length — its purpose is to prevent the Ollama server from
# crashing, not to respect the model's theoretical capacity.
_SAFE_CHUNK_SIZE = 3000
_CHUNK_OVERLAP = 200


def _resolve_char_cap(max_length: int | None) -> int:
    """
    Overall content cap for what gets processed (not the size of each
    individual request — that's controlled by _SAFE_CHUNK_SIZE). If the
    text exceeds this cap, it's truncated to this length before chunking.
    """
    if max_length is None:
        return _DEFAULT_SAFETY_CHAR_CAP
    return max_length * _CHARS_PER_TOKEN_ESTIMATE


def _chunk_text(text: str) -> list[str]:
    """Split text into safe chunks with a small overlap (so sentence boundaries aren't cut)"""
    if len(text) <= _SAFE_CHUNK_SIZE:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + _SAFE_CHUNK_SIZE
        chunks.append(text[start:end])
        start = end - _CHUNK_OVERLAP
    return chunks


def _average_vectors(vectors: list[list[float]]) -> list[float]:
    if len(vectors) == 1:
        return vectors[0]
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]


def _embed_single_request(text: str, retries: int = 2) -> list[float]:
    """A single request to Ollama — assumes text has already been chunked to a safe size"""
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={"model": MODEL_NAME, "prompt": text},
                timeout=120,
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < retries:
                print(f"  ⏳ Embedding error, retrying ({attempt + 1}/{retries})...")
                time.sleep(3)

    raise last_error


def embed_text(text: str, retries: int = 2, max_length: int | None = None) -> list[float]:
    char_cap = _resolve_char_cap(max_length)
    text = text[:char_cap]

    chunks = _chunk_text(text)
    if len(chunks) > 1:
        print(f"  ✂️  Long text split into {len(chunks)} chunks (to prevent Ollama from crashing)")

    vectors = [_embed_single_request(chunk, retries=retries) for chunk in chunks]
    return _average_vectors(vectors)


def embed_batch(
    texts: list[str],
    batch_size: int = None,
    delay: float = 0.2,
    max_length: int | None = None,
) -> list[list[float]]:
    """
    Ollama has no real server-side batching (one text per request), so we
    call it one by one. batch_size is only kept for compatibility with
    previous call sites.
    """
    embeddings = []
    for text in texts:
        embeddings.append(embed_text(text, max_length=max_length))
        if delay:
            time.sleep(delay)
    return embeddings