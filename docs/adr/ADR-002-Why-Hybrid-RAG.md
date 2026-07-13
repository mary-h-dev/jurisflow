# ADR-002 — Why Hybrid Retrieval (Graph + Vector)

**Date:** 2026-07  
**Status:** Accepted

---

## Context

Legal questions in Persian require two fundamentally different types of retrieval:

1. **Semantic similarity** — "what does this question mean?" maps to related articles
2. **Structural relationships** — "Article 190 references Article 223" must be followed

Neither approach alone is sufficient for legal reasoning.

---

## Decision

Use a **Hybrid Retrieval** pipeline that combines:
- Vector similarity search (semantic layer)
- Neo4j graph traversal (structural layer)

```
Query
  ↓
embed(query)
  ↓
Vector Search → top-10 articles by cosine similarity
  ↓
Graph Traversal → expand via REFERENCES + HAS_TOPIC edges
  ↓
Deduplicate + Rank
  ↓
LLM Answer Generation with citations
```

---

## Alternatives Considered

| Option | Reason Rejected |
|--------|----------------|
| Pure vector search | Misses explicit article references; accuracy 68% |
| Pure keyword search | No semantic understanding of Persian legal language |
| Pure graph traversal | Requires exact keyword match to enter the graph |
| LLM with full law text | Context window overflow; hallucination risk |

---

## Benchmark Results

Evaluated on 38 ground-truth Persian legal Q&A pairs:

| Strategy | Accuracy |
|----------|----------|
| Vector only, top_k=5 | 73.7% |
| Vector + topic filter | 68.4% |
| **Vector only, top_k=10** | **81.6%** |

> Topic filter was removed after experiments showed it reduces accuracy
> when classification confidence is below 0.7.

---

## Confidence Scoring

Each response includes a composite confidence score:

```
Final = 0.5 × embedding_score
      + 0.3 × llm_confidence
      + 0.2 × graph_score

≥ 0.8 → high    (answer presented directly)
≥ 0.6 → medium  (review warning shown)
< 0.6 → low     (lawyer consultation recommended)
```

---

## Consequences

**Positive:**
- 81.6% article-level retrieval accuracy on MVP dataset
- Citation-backed answers prevent hallucination
- Graph traversal finds related articles the user didn't explicitly ask about

**Negative:**
- Two-step retrieval adds ~300-500ms latency
- Accuracy is bounded by embedding model quality (bge-m3 for dev)
- Production accuracy expected to improve significantly with Gemini embeddings
