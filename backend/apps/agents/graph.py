from __future__ import annotations

import logging
import uuid
from typing import Any

from langgraph.graph import StateGraph, END
from langgraph.types import Send

from .state import CaseState
from .nodes.analyzer            import analyze_document
from .nodes.searcher            import search_laws
from .nodes.teammate_defender   import defend
from .nodes.teammate_prosecutor import prosecute
from .nodes.teammate_judge      import judge
from .nodes.lead                import lead_decision

logger = logging.getLogger(__name__)


# ── Edge Conditions ───────────────────────────────────────────────────────────

def should_continue_after_analyzer(state: CaseState) -> str:
    """
    بعد از analyzer چک می‌کنه اطلاعات کافی استخراج شده.
    اگه موضوع پیدا نشد → error_handler
    """
    if not state.get("case_subject") or not state.get("case_type"):
        logger.warning(f"[graph] analyzer failed — session={state['session_id']}")
        return "error"
    return "continue"


def should_continue_after_searcher(state: CaseState) -> str:
    """بعد از searcher چک می‌کنه قوانین پیدا شده"""
    if state.get("error"):
        return "error"
    return "continue"


def route_to_teammates(state: CaseState) -> list[Send]:
    """
    بعد از searcher سه teammate به صورت parallel اجرا میشن.
    هر کدوم همون state رو می‌گیرن و مستقل کار می‌کنن.
    """
    return [
        Send("defender",   state),
        Send("prosecutor", state),
        Send("judge",      state),
    ]


# ── Error Handler ─────────────────────────────────────────────────────────────

def handle_error(state: CaseState) -> dict:
    return {
        "error": (
            state.get("error") or
            "خطایی در پردازش پرونده رخ داد. لطفاً متن را بررسی کنید."
        ),
        "recommendation":  "امکان تحلیل پرونده وجود ندارد.",
        "completed_nodes": state.get("completed_nodes", []) + ["error_handler"],
    }


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph() -> Any:
    """
    Flow:
    START
      ↓
    analyzer
      ↓ (conditional)
    searcher
      ↓ (parallel via Send)
    ┌──────────────────────────┐
    defender  prosecutor  judge
    └──────────────────────────┘
      ↓ (همه به lead میرن)
    lead_decision
      ↓
    END
    """
    graph = StateGraph(CaseState)

    # ── Nodes ──────────────────────────────────────────────────────────────
    graph.add_node("analyzer",     analyze_document)
    graph.add_node("searcher",     search_laws)
    graph.add_node("defender",     defend)
    graph.add_node("prosecutor",   prosecute)
    graph.add_node("judge",        judge)
    graph.add_node("lead",         lead_decision)
    graph.add_node("error_handler",handle_error)

    # ── Edges ───────────────────────────────────────────────────────────────

    graph.set_entry_point("analyzer")

    # analyzer → conditional
    graph.add_conditional_edges(
        "analyzer",
        should_continue_after_analyzer,
        {
            "continue": "searcher",
            "error":    "error_handler",
        },
    )

    # searcher → conditional
    graph.add_conditional_edges(
        "searcher",
        should_continue_after_searcher,
        {
            "continue": "defender",  # LangGraph Send parallel رو handle می‌کنه
            "error":    "error_handler",
        },
    )

    # parallel teammates → همه به lead میرن
    graph.add_edge("defender",   "prosecutor")
    graph.add_edge("prosecutor", "judge")
    graph.add_edge("judge",      "lead")




    # lead → END
    graph.add_edge("lead",          END)
    graph.add_edge("error_handler", END)

    return graph.compile()


# ── Initial State ─────────────────────────────────────────────────────────────

def create_initial_state(
    extracted_text: str,
    user_id:        int,
    case_context:   str = "",
) -> CaseState:
    full_text = extracted_text
    if case_context:
        full_text = f"توضیح کاربر: {case_context}\n\n---\n\n{extracted_text}"

    return CaseState(
        raw_text=full_text,
        user_id=user_id,
        session_id=str(uuid.uuid4()),
        case_subject=None,
        case_type=None,
        parties=None,
        key_facts=None,
        legal_keywords=None,
        related_laws=None,
        cited_articles=None,
        defender_opinion=None,
        prosecutor_opinion=None,
        judge_opinion=None,
        strengths=None,
        weaknesses=None,
        risks=None,
        win_chance=None,
        strategy=None,
        recommendation=None,
        next_steps=None,
        error=None,
        completed_nodes=[],
    )


# ── Singleton ─────────────────────────────────────────────────────────────────
case_graph = build_graph()