# ADR-004 — Multi-Agent Deliberation Architecture

**Date:** 2026-08  
**Status:** Accepted  
**Component:** `backend/apps/legal_agents`

## Context

Legal case analysis requires considering multiple perspectives
simultaneously: the strongest defense argument, the opposing weaknesses,
and a neutral assessment of the realistic outcome. A single LLM call
cannot reliably produce this multi-perspective analysis — it either
picks one side or produces a superficial balance.

We also needed a mechanism to detect genuine disagreement between
perspectives, not just produce a single confidence score.

## Decision

Use three specialized agents running in a structured debate:

```
Defender ──┐
           ├──→ Judge ──→ Fusion
Prosecutor ┘
```

- **Defender**: argues for the strongest applicable articles
- **Prosecutor**: challenges applicability, raises counter-arguments
- **Judge**: evaluates both positions and issues a verdict

Defender and Prosecutor run in parallel; Judge runs after, seeing
both opinions as context.

Each agent returns:
- A discrete verdict (`strong_for`, `moderate_for`, `neutral`, `moderate_against`, `strong_against`)
- A scalar confidence (0.0–1.0)
- Cited articles
- Supporting arguments

## Why Three Agents, Not Two

A three-way vote produces a genuine 2-1 split — a meaningful
uncertainty signal. Two agents can only agree or tie, collapsing
the uncertainty signal to a binary with no middle ground.

## Why No LLM Arbiter

Adding a fourth LLM call to resolve disagreement would mask the
signal that deliberation is meant to expose. A genuine split
between Defender and Prosecutor is itself informative — it should
surface as an uncertainty flag, not be silently resolved.

## Fault Tolerance

- Fewer than 2 agents return → explicit error raised
- Exactly 2 agents return → neutral filler opinion (confidence=0.5)
  added to preserve majority-voting structure without introducing
  a substantive synthetic judgment

## Alternatives Considered

| Option | Reason Rejected |
|--------|----------------|
| Single LLM call | Cannot reliably argue both sides simultaneously |
| Two agents only | No genuine split possible; uncertainty collapses to binary |
| LLM arbiter for disagreement | Masks uncertainty signal with false confidence |
| ReAct agent with tools | Unpredictable structure; hard to guarantee output format |

## Consequences

**Positive:**
- Multi-perspective output surfaces genuine legal uncertainty
- Consensus level (full/majority/split) is a meaningful signal
- Each agent's reasoning is visible and auditable

**Negative:**
- 3 sequential LLM calls → higher latency per query
- Judge quality depends on Defender and Prosecutor quality


## Related
- ADR-005 — Deterministic Fusion
- `backend/apps/legal_agents/services.py`