from __future__ import annotations

import logging

from ninja import Router
from ninja.errors import HttpError

from core.auth import jwt_auth
from apps.search.schemas import SearchIn
from apps.search.services import search_service

from .schemas import AuditorOut
from .services import auditor_service

logger = logging.getLogger(__name__)
router = Router()


@router.post("/audit", auth=jwt_auth, response=AuditorOut)
def audit(request, data: SearchIn):
    if len(data.query.strip()) < 5:
        raise HttpError(400, "Query is too short.")

    try:
        search_result = search_service.search(data.query)
        result = auditor_service.audit(search_result)
    except Exception as e:
        logger.error(f"Audit failed: {e}", exc_info=True)
        raise HttpError(500, "Audit failed. Please try again.")

    return result