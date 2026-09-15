# ADR-001 — Multi-Channel Retrieval Architecture

**Date:** 2026-08
**Status:** Accepted  
**Component:** `apps/search`

## Context

The search service must retrieve relevant legal evidence for Persian legal queries.  
Legal knowledge is heterogeneous: fact-dependent concepts (e.g. نشوز, دفاع مشروع), judicial rulings, and statutory articles.  
A single retrieval strategy (pure embedding or pure keyword) fails to cover all three types reliably.

We also need a calibrated, interpretable confidence score that reflects the quality and completeness of the retrieved evidence.

## Decision

Adopt a fixed three-channel retrieval architecture:

1. **Feature channel**  
   - First attempt deterministic exact / token-sequence matching against a closed legal vocabulary (concepts, actions, roles, objects).  
   - On match → retrieve linked rulings via graph relationships (`HAS_CONCEPT`, `HAS_ROLE`, …).  
   - On miss → fall back to embedding similarity over feature nodes.

2. **Ruling channel**  
   - Dense retrieval over ruling summaries and ruling sections (bge-m3).  
   - The two sub-rankings are fused with Reciprocal Rank Fusion (`k=60`).

3. **Article channel**  
   - Dense retrieval over statutory article text (bge-m3).

All three channels are **always** executed in parallel.  
LLM-based channel gating was evaluated and rejected.

Final ranking across the three channels is produced by a second RRF pass (top-20).

Retrieval confidence is computed as:

\[
\text{Confidence} = (w_f \cdot q_f + w_r \cdot q_r + w_a \cdot q_a) - 0.1 \times \text{(number of empty channels)}
\]

where \(q_c\) is the mean of the top-3 similarity scores of channel \(c\) (or 0 if empty).  
Weights were obtained by logistic regression on 140 leakage-free samples; the ruling channel received the highest weight.

## Why

- Exact matching on the closed vocabulary eliminates silent errors caused by near-synonyms that are inseparable in embedding space (e.g. توقیف vs توقف, cosine gap 0.043).
- Keeping feature, ruling, and article as separate channels respects the different semantics and indexing needs of each knowledge type.
- Always querying all three channels improved both AUROC (0.756 vs 0.682) and rank correlation compared with LLM gating.
- A simple, calibrated confidence formula is more robust and interpretable in production than an uncalibrated LLM self-score.

## Consequences

**Positive**
- Deterministic behaviour for vocabulary terms → zero silent feature mismatches.
- Higher retrieval quality and better-calibrated confidence than gated or single-channel baselines.
- Clear separation of concerns makes the pipeline easy to debug and extend.

**Negative**
- Higher latency and cost (three retrievals instead of one or two).
- Lower recall for out-of-vocabulary legal terms (they are never guessed).
- Confidence formula deliberately under-weights feature and article channels relative to the raw regression coefficients for production robustness.

## Related
- ADR-006 — Deterministic Vocabulary Matching  
- `backend/apps/search/services.py`  
- `backend/apps/search/confidence.py`  
- `backend/apps/search/exact_feature_matcher.py`