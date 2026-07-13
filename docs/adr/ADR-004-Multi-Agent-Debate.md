# ADR-004 — Multi-Agent Debate Architecture

**Date:** 2026-07  
**Status:** Accepted

---

## Context

Legal case analysis is not a single-answer problem.
A good legal opinion requires considering multiple perspectives:
- What is the strongest defense argument?
- What are the weaknesses the opposing side will exploit?
- What is the realistic outcome based on law and evidence?

A single LLM call cannot reliably produce this multi-perspective analysis.

---

## Decision

Use a **Lead + Teammates debate architecture** via LangGraph:

```
START
  ↓
Analyzer      — extract case facts, parties, keywords
  ↓
Searcher      — retrieve related laws from Neo4j
  ↓
Defender      — strongest defense arguments
  ↓
Prosecutor    — opposing arguments and weaknesses
  ↓
Judge         — neutral assessment (sees both opinions)
  ↓
Lead Agent    — synthesize debate → final recommendation
  ↓
END
```

Each teammate has a **distinct role and system prompt** that biases
their reasoning toward their perspective.

---

## Alternatives Considered

| Approach | Reason Rejected |
|----------|----------------|
| Single LLM call | Cannot reliably argue both sides simultaneously |
| Sequential pipeline (no debate) | Later agents blindly trust earlier agents |
| ReAct agent with tools | Too unpredictable; hard to guarantee structure |
| Pure parallel agents | Judge cannot see Defender/Prosecutor opinions |

---

## Why LangGraph

| Requirement | LangGraph Support |
|-------------|------------------|
| Stateful multi-step workflow | ✅ StateGraph + TypedDict |
| Conditional routing | ✅ `add_conditional_edges` |
| Parallel execution (planned) | ✅ `Send` API |
| Observable execution | ✅ Built-in tracing |
| Production-ready | ✅ Industry standard |

---

## Current Limitations

**Sequential, not parallel (v1):**

Currently Defender → Prosecutor → Judge run sequentially.
This means Judge always sees both opinions, which is correct,
but total latency is ~3× a single call.

Planned for v2: use LangGraph `Send` API for true parallel execution
of Defender and Prosecutor, then pass both results to Judge.

```python
# v2 planned
def route_to_parallel(state):
    return [
        Send("defender", state),
        Send("prosecutor", state),
    ]
```

---

## Plan-Based Access

Agent analysis is restricted to **Pro and Enterprise plans** only.

Reason: each case analysis makes 5+ LLM calls (Groq) and
1 Neo4j vector search — significantly more expensive than a simple search query.

---

## Consequences

**Positive:**
- Multi-perspective output is significantly more useful for legal decisions
- Each agent's reasoning is visible in the API response (transparency)
- LangGraph state is fully serializable for debugging

**Negative:**
- 5 sequential LLM calls → ~8-15 second latency per analysis
- Groq rate limits can cause failures under load
- Judge quality depends on Defender and Prosecutor quality
