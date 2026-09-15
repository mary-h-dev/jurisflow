# ADR-005 — Deterministic Uncertainty Fusion

**Date:** 2026-09  
**Status:** Accepted  
**Component:** `backend/apps/legal_agents/fusion`

## Context

After three agents produce independent verdicts and confidence scores,
these signals must be combined into a single final confidence score.
The natural approach — a fourth LLM call to synthesize the debate —
was rejected on principle: an LLM arbiter can resolve genuine
three-agent disagreement with false confidence, masking exactly the
signal deliberation is meant to expose.

## Decision

Use a fully deterministic weighted fusion with no additional LLM call.

**Step 1 — Majority verdict**

Plurality vote over {v1, v2, v3}.
If all three differ (split): majority defaults to `neutral`.

**Step 2 — Agent confidence summary**

- Mean: c̄ = (1/3) Σ ci
- Std:  σc = std(c1, c2, c3)

**Step 3 — Blended confidence**

- ũ = αp · p + αa · c̄
- p = Auditor prior confidence (incorporates retrieval quality and prune ratio)
- Deployed: αp=0.40, αa=0.60

**Step 4 — Split penalty**

- u = max(0, ũ − 0.10)   if consensus == SPLIT
- u = ũ                   otherwise

**Step 5 — Uncertainty flag**

Raised when σc > 0.20 OR consensus is SPLIT.
Designed as a disclosure signal for human review,
not an automatic abstention criterion.

## Why Deterministic

- Reproducible: same inputs always produce the same output
- Auditable: every number in the formula is inspectable
- No hallucinated confidence: genuine disagreement surfaces
  as a split penalty and uncertainty flag rather than being
  resolved by another LLM call

## Fusion Weight Calibration

Weights were selected by grid search on train_val (n=112).
Fitted optimum: αp=0.35 (Δ=0.05 from deployed).
Sweeping smaller subsets produced optima from 0.35 to 1.00 —
reliable calibration requires scale closer to n=112.
Deployed weights retained as closest available optimum.

See: `backend/apps/legal_agents/calibration/CALIBRATION_FUSION_NOTES.md`

## Alternatives Considered

| Option | Reason Rejected |
|--------|----------------|
| LLM arbiter call | Masks genuine uncertainty with false confidence |
| Simple average of agent scores | Ignores prior confidence and consensus structure |
| Weighted sum without penalty | Split consensus carries no additional signal |

## Consequences

**Positive:**
- Fully deterministic and reproducible
- Split consensus produces a measurable, reportable uncertainty signal
- No additional API cost or latency

**Negative:**
- Agent confidence is weakly calibrated (AUROC=0.318 end-to-end)
- Fusion weights unstable below n=112; re-tuning deferred to
  future work after dataset expansion

## Related
- ADR-004 — Multi-Agent Deliberation
- `backend/apps/legal_agents/fusion/deterministic.py`
- `backend/apps/legal_agents/calibration/CALIBRATION_FUSION_NOTES.md`