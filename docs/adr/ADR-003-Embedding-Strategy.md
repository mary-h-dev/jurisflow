# ADR-003 — Embedding Strategy

**Date:** 2026-07  
**Status:** Accepted (dev) / Planned (prod)

---

## Context

Embedding quality directly determines retrieval accuracy.
Persian legal text presents specific challenges:

- Classical Persian vocabulary in older laws (pre-1979)
- Legal terminology not present in general corpora
- Mixed Arabic-Persian script
- Article numbers embedded in text

We needed an embedding model that handles Persian well,
fits our infrastructure constraints, and can scale to production.

---

## Decision

Use a **two-tier embedding strategy**:

| Environment | Model | Dimensions | Rate Limit |
|-------------|-------|------------|------------|
| Development | Ollama `bge-m3` | 1024 | None (local) |
| Production  | Gemini `gemini-embedding-001` | 3072 | 1000 req/day (free) |

Switching is controlled by `settings.DEBUG`:

```python
def _embed(self, text: str) -> list[float]:
    if settings.DEBUG:
        return self._embed_ollama(text)
    return self._embed_gemini(text)
```

---

## Alternatives Considered

| Model | Decision | Reason |
|-------|----------|--------|
| `mxbai-embed-large` | Rejected | English-only; poor Persian performance |
| `paraphrase-multilingual-mpnet-base-v2` | Rejected | 768 dims; limited Persian training |
| `text-embedding-3-small` (OpenAI) | Rejected | No free tier; cost concerns |
| `gemini-embedding-001` | Accepted (prod) | Best Persian performance; free tier available |
| `bge-m3` | Accepted (dev) | Local, no rate limit, acceptable Persian support |

---

## Migration Protocol

Switching embedding models requires a full re-indexing:

```cypher
-- Step 1: Drop old index
DROP INDEX article_embedding IF EXISTS

-- Step 2: Clear embeddings
MATCH (a:Article) REMOVE a.embedding
```

```bash
# Step 3: Re-embed with new model
uv run python -m embeddings.pipeline
```

**This must be done atomically** — mixed-dimension embeddings in the same
index will cause `vector dimensionality mismatch` errors.

---

## Embedding Pipeline Rate Limits

| Model | Limit | Workaround |
|-------|-------|------------|
| Gemini free tier | 1000 req/day | Resume-safe pipeline (skips already embedded) |
| Ollama bge-m3 | None | `time.sleep(0)` — full speed |

The pipeline uses `WHERE a.embedding IS NULL` to skip already-processed articles,
making it safe to interrupt and resume at any point.

---

## Consequences

**Positive:**
- Zero cost during development
- Production model (Gemini) significantly outperforms dev model for Persian
- Resume-safe pipeline prevents duplicate work

**Negative:**
- Different dimensions between dev and prod require full re-indexing on switch
- Gemini 1000 req/day free limit means initial embedding takes 2+ days for large corpora
- Model names are not stable (e.g., `gemini-1.5-flash` → `gemini-2.5-flash`)
