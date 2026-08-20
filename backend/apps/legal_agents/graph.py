"""
LangGraph graph for the three-agent deliberation layer.

Topology:
                  ┌─ defender_node ──┐
  [START] ────────┤                  ├──► judge_node ──► fusion_node ──► [END]
                  └─ prosecutor_node ┘

defender and prosecutor run in parallel.
judge runs after both complete (enforced by edges).
fusion is deterministic — no LLM.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from legal_agents.nodes.graph_nodes import (
    KEY_AUDITOR_OUT,
    KEY_COMPLETED,
    KEY_DEFENDER_OPINION,
    KEY_ERROR,
    KEY_FUSION,
    KEY_JUDGE_OPINION,
    KEY_PROSECUTOR_OPINION,
    defender_node,
    fusion_node,
    judge_node,
    prosecutor_node,
)
from legal_agents.schemas import (
    AgentOpinion,
    AuditorOut,
    CombinedConfidenceOut,
    FusionOut,
)

from typing import Any, Optional
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class DeliberationState(TypedDict, total=False):
    # Input (required)
    auditor_out: AuditorOut

    # Agent outputs
    defender_opinion:   Optional[AgentOpinion]
    prosecutor_opinion: Optional[AgentOpinion]
    judge_opinion:      Optional[AgentOpinion]

    # Fusion output
    fusion:               Optional[FusionOut]
    applicable_articles:  list[Any]
    auditor_confidence:   Optional[CombinedConfidenceOut]

    # Metadata
    completed_nodes: list[str]
    error:           Optional[str]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    g = StateGraph(DeliberationState)

    g.add_node("defender",   defender_node)
    g.add_node("prosecutor", prosecutor_node)
    g.add_node("judge",      judge_node)
    g.add_node("fusion",     fusion_node)

    # Parallel fan-out from START
    g.add_edge(START,        "defender")
    g.add_edge(START,        "prosecutor")

    # Fan-in to judge
    g.add_edge("defender",   "judge")
    g.add_edge("prosecutor", "judge")

    # Sequential to fusion then end
    g.add_edge("judge",      "fusion")
    g.add_edge("fusion",     END)

    return g


# Pre-built compiled graph — import and invoke this in your API layer.
deliberation_graph = build_graph().compile()


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_deliberation(auditor_out: AuditorOut) -> DeliberationState:
    """
    Run the full deliberation pipeline given a completed AuditorOut.

    Returns the final DeliberationState, which the API layer converts
    to CaseAnalysisOut.
    """
    initial_state: DeliberationState = {
        KEY_AUDITOR_OUT: auditor_out,
        KEY_COMPLETED:   [],
        KEY_ERROR:       None,
    }
    return deliberation_graph.invoke(initial_state)