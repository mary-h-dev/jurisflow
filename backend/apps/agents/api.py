from __future__ import annotations

import logging
from typing import Optional

from ninja import Router
from ninja.errors import HttpError

from core.auth import jwt_auth
from core.permissions import check_plan_limit
from .schemas import (
    CaseAnalysisIn,
    CaseAnalysisOut,
    PartyOut,
    RelatedLawOut,
    RiskOut,
    TeammateOpinionOut,
    ProgressOut,
)
from .graph import case_graph, create_initial_state
from .state import TeammateOpinion, RelatedLaw, Party, Risk

logger = logging.getLogger(__name__)
router = Router()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_party_out(p: Party) -> PartyOut:
    return PartyOut(
        role=p["role"],
        name=p["name"],
        details=p["details"],
    )


def _to_law_out(law: RelatedLaw) -> RelatedLawOut:
    return RelatedLawOut(
        article_ref=law["article_ref"],
        law_name=law["law_name"],
        summary=law["summary"],
        relevance=law["relevance"],
    )


def _to_risk_out(risk: Risk) -> RiskOut:
    return RiskOut(
        title=risk["title"],
        description=risk["description"],
        severity=risk["severity"],
    )


def _to_teammate_out(opinion: TeammateOpinion | None) -> TeammateOpinionOut | None:
    if not opinion:
        return None
    return TeammateOpinionOut(
        role=opinion["role"],
        position=opinion["position"],
        arguments=opinion["arguments"],
        cited_articles=opinion["cited_articles"],
        confidence=opinion["confidence"],
    )


def _build_response(state: dict) -> CaseAnalysisOut:
    """تبدیل state نهایی به response schema"""
    return CaseAnalysisOut(
        session_id=state.get("session_id", ""),

        # Analyzer
        case_subject=state.get("case_subject") or "",
        case_type=state.get("case_type")     or "",
        parties=[
            _to_party_out(p)
            for p in (state.get("parties") or [])
        ],
        key_facts=state.get("key_facts") or [],

        # Searcher
        related_laws=[
            _to_law_out(law)
            for law in (state.get("related_laws") or [])
        ],
        cited_articles=state.get("cited_articles") or [],

        # Teammates
        defender_opinion=_to_teammate_out(
            state.get("defender_opinion")
        ),
        prosecutor_opinion=_to_teammate_out(
            state.get("prosecutor_opinion")
        ),
        judge_opinion=_to_teammate_out(
            state.get("judge_opinion")
        ),

        # Lead
        strengths=state.get("strengths")  or [],
        weaknesses=state.get("weaknesses") or [],
        risks=[
            _to_risk_out(r)
            for r in (state.get("risks") or [])
        ],
        win_chance=state.get("win_chance")     or "متوسط",
        strategy=state.get("strategy")         or "دفاعی",
        recommendation=state.get("recommendation") or "",
        next_steps=state.get("next_steps")     or [],

        # Metadata
        completed_nodes=state.get("completed_nodes") or [],
        error=state.get("error"),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/analyze",
    auth=jwt_auth,
    response=CaseAnalysisOut,
    summary="تحلیل هوشمند پرونده حقوقی",
)
def analyze_case(request, data: CaseAnalysisIn):
    """
    تحلیل پرونده با Multi-Agent Debate:
    ۱. Analyzer  — استخراج اطلاعات پرونده
    ۲. Searcher  — جستجوی قوانین مرتبط از Neo4j
    ۳. Teammates — debate بین وکیل، دادستان، قاضی
    ۴. Lead      — جمع‌بندی و توصیه نهایی

    نیاز به پلن Pro یا بالاتر دارد.
    """
    user = request.user

    # فقط pro و enterprise می‌تونن از agent استفاده کنن
    if user.plan == "free":
        raise HttpError(
            403,
            "تحلیل پرونده فقط برای کاربران Pro و Enterprise در دسترس است."
        )

    # چک plan limit
    check_plan_limit(user)

    if len(data.extracted_text.strip()) < 50:
        raise HttpError(
            400,
            "متن پرونده خیلی کوتاه است. حداقل ۵۰ کاراکتر لازم است."
        )

    try:
        # ساخت state اولیه
        initial_state = create_initial_state(
            extracted_text=data.extracted_text,
            user_id=user.id,
            case_context=data.case_context,
        )

        logger.info(
            f"[api/analyze] user={user.id} "
            f"session={initial_state['session_id']} "
            f"text_len={len(data.extracted_text)}"
        )

        # اجرای graph
        final_state = case_graph.invoke(initial_state)

        # increment query count
        user.can_query_and_increment()

        return _build_response(final_state)

    except Exception as e:
        logger.error(f"[api/analyze] error: {e}", exc_info=True)
        raise HttpError(500, "خطا در پردازش پرونده. لطفاً دوباره تلاش کنید.")


@router.get(
    "/progress/{session_id}",
    auth=jwt_auth,
    response=ProgressOut,
    summary="وضعیت پیشرفت تحلیل",
)
def get_progress(request, session_id: str):
    """
    وضعیت realtime تحلیل پرونده.
    در آینده با WebSocket یا SSE جایگزین میشه.
    """
    # TODO: بعد از اضافه کردن Celery، از task status استفاده کن
    raise HttpError(501, "این endpoint در نسخه بعدی پیاده‌سازی میشه.")