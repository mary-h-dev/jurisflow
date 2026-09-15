# ADR-003 — Data Pipeline: Features (ETL)

**Date:** 2026-07

**Status:** Accepted

**Component:** `pipeline/graph-rag/features`

---

## Context

Feature extraction (concepts, actions, roles, objects) must map ruling text to a **closed legal vocabulary** so graph nodes are unified and queryable across rulings.

Directly providing the full vocabulary in the LLM prompt failed due to:

1. **Context Bottleneck & Attention Loss:** A massive prompt degraded retrieval accuracy (*Lost in the Middle*) for terms in the center of the list.
2. **Latency & Throughput Constraints:** Large token footprints per request caused unnecessary input-token overhead, introduced a processing latency bottleneck, and reduced pipeline throughput.
3. **Failure of Vector Pre-filtering (`vocab_retrieval.py`):** Pre-filtering terms via embedding similarity against long ruling paragraphs caused short, high-frequency terms (e.g., "خواهان") to score poorly and be silently dropped before reaching the LLM.

---

## Decision

Split the extraction process into **two independent stages**:

```text
[ Ruling Section Text ]
          │
          ▼ 1. Free Extraction (extractor.py)
[ Raw Phrases + Verbatim Evidence Quote ]
          │
          ▼ 2. Deterministic Resolution (vocab_resolver.py)
[ Unified Neo4j Graph Nodes ]

```

1. **Free Extraction (`extractor.py`):** The LLM extracts terms directly from ruling text without seeing any vocabulary or paraphrasing. Each extracted value **must** include a verbatim evidence quote.
2. **Post-Hoc Resolution (`vocab_resolver.py`):** Extracted phrases are deterministically matched against the closed vocabulary *after* the LLM call using exact match, alias mapping, and token containment. Unmatched terms are dropped rather than guessed (per ADR-006).

> **Exception:** The `fact` field remains free text without vocabulary resolution due to the unbounded nature of case facts.

---

## Alternatives Considered

| Option | Reason Rejected |
| --- | --- |
| **Full vocabulary in prompt** | Causes attention loss (*Lost in the Middle*), increases latency, and hits TPM throughput ceilings. |
| **Embedding pre-filtering (`vocab_retrieval.py`)** | Short, critical terms scored poorly against long text blocks and were silently dropped. |
| **Unresolved open text output** | Results in an open vocabulary, preventing node sharing and cross-ruling graph queries. |

---

## Consequences

**Positive:**

* **Optimized Context Window:** Keeps prompts compact, deterministic, and provider-agnostic.
* **High Precision:** Resolution on short extracted phrases is significantly more accurate than case-level embedding search.
* **Zero Hallucination:** Graph nodes are strictly verified matches; invalid entities are dropped rather than misassigned.

**Negative:**

* **Two-Stage Pipeline:** Requires a deterministic post-processing step following the LLM call.
* **Recall Bounded by Coverage:** Unmatched valid terms are dropped (mitigated manually via `vocab_gap_audit.py` and `merge_approved_gaps.py`).