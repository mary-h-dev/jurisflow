# ADR-004 — Data Pipeline: Embedding (ETL)

**Date:** 2026-07
**Status:** Accepted

---

## Context

Persian legal text needs to be embedded for vector search across the Rule
Graph (Article, Note), the Fact Graph (Ruling title/summary,
RulingSection), and the Feature Graph (LegalConcept, LegalAction,
LegalRole, LegalObject, LegalFact). The initial approach used the native
`bge-m3` model (FlagEmbedding) directly in-process.

## Decision

Use **Ollama-hosted `bge-m3`**, running locally, called via HTTP
(`http://localhost:11434/api/embeddings`) through a single entry point:
`common.embedder.embed_text()`.

## Alternatives Considered

| Option | Reason Rejected |
|--------|----------------|
| Native `bge-m3` (FlagEmbedding, in-process) | Repeated OOM crashes on a memory-constrained environment (Codespace, 9GB RAM) |

## Long-Text Handling

Sending long `RulingSection` text in a single request crashed the
Ollama server (500 error) — actual serving context is smaller than the
model's theoretical 8192-token capacity. Fix: long text is split into
3000-character chunks (200-character overlap), each embedded
separately, and the vectors are averaged — no content dropped, no
request exceeds Ollama's safe limit.

## Storage Design

Each embeddable unit gets its own dedicated field: `Article.embedding`,
`Note.embedding`, `Ruling.title_embedding`, `Ruling.summary_embedding`,
`RulingSection.embedding`, `LegalConcept.embedding`,
`LegalAction.embedding`, `LegalRole.embedding`, `LegalObject.embedding`,
`LegalFact.embedding`. Vector indexes use 1024 dimensions with cosine
similarity, matching bge-m3's native output size.

## Feature Layer: Copy vs. Re-Embed

`LegalConcept`/`LegalAction`/`LegalRole`/`LegalObject` are **not**
re-embedded when attached to the graph — their vectors are copied
directly from `vocab_embeddings_cache.json` (the same cache built once
during vocabulary categorization/gap-audit) via
`EmbeddingStore.attach_vocab_embeddings()`. Re-embedding the same
~400–600 canonical terms on every attach would be redundant, and risks
silently drifting from the cache used elsewhere (e.g.
`vocab_gap_audit.py`'s similarity checks).

`LegalFact` is the exception: it stays free text by design (ADR-003),
so it has no cache entry — it is embedded fresh, per fact, on every run.

Run-order dependency: `embed_features.py` must run *after*
`main_features.py load`, since it attaches embeddings to Feature nodes
that must already exist in the graph.

## Consequences

**Positive:**
- Stable within available memory (no more OOM crashes)
- No cost, no external API rate limits
- Resumable by design — indexing queries filter on `embedding IS NULL`
- Feature-layer embeddings stay consistent with the vocabulary cache
  (no duplicate/drifted vectors for the same canonical term)

**Negative:**
- The Ollama-served model is quantized, trading some accuracy for stability
- Embeddings from Ollama `bge-m3` are not compatible with any other
  model's vector space — switching models later requires regenerating
  every embedding across all three graphs from scratch, with no partial
  migration path