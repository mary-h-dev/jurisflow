import logging

from ninja import Router
from ninja.errors import HttpError

from core.auth import jwt_auth
from .schemas import (
    SearchIn,
    SearchOut,
    EvidenceOut,
    ConfidenceOut,
    ChannelQualityOut,
    RoutingOut,
)
from .services import search_service

logger = logging.getLogger(__name__)
router = Router()


@router.post("/query", auth=jwt_auth, response=SearchOut)
def query(request, data: SearchIn):
    if len(data.query.strip()) < 5:
        raise HttpError(400, "Query is too short.")

    try:
        result = search_service.search(data.query)
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HttpError(500, "Search failed. Please try again.")

    conf = result.confidence
    routing = result.routing

    return SearchOut(
        query=result.query,
        evidences=[
            EvidenceOut(
                source_type=e.source_type,
                text=e.text,
                score=e.score,
                rrf_score=e.rrf_score,
                ruling_id=e.ruling_id,
                feature_value=e.feature_value,
                feature_category=e.feature_category,
                article_number=e.article_number,
                law_name=e.law_name,
                cited_articles=e.cited_articles,
            )
            for e in result.evidences
        ],
        confidence=ConfidenceOut(
            score=conf.score,
            level=conf.level,
            note=conf.note,
            channels=ChannelQualityOut(
                feature=conf.vector.feature_quality,
                ruling=conf.vector.ruling_quality,
                article=conf.vector.article_quality,
                missing=conf.vector.missing_channels,
            ),
            ),
        routing=RoutingOut(
            rewritten_query=routing.rewritten_query,
            channels=routing.channels,
            routing_confidence=routing.routing_confidence,
            intent=routing.intent,
            case_type_hint=routing.case_type_hint,
            ambiguity_flag=routing.ambiguity_flag,
            raw_ok=routing.raw_ok,
        ),
        ruling_ids=result.ruling_ids,
        article_refs=result.article_refs,
    )