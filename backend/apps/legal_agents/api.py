from __future__ import annotations

import logging

from ninja import Router
from ninja.errors import HttpError

from core.auth import jwt_auth

from .schemas import AuditorOut, DeliberationOut
from .services import deliberation_service

logger = logging.getLogger(__name__)
router = Router()


@router.post("/deliberate", auth=jwt_auth, response=DeliberationOut)
def deliberate(request, data: AuditorOut):
    if not data.query.strip():
        raise HttpError(400, "Query is empty.")

    try:
        result = deliberation_service.run(data)
    except Exception as e:
        logger.error(f"Deliberation failed: {e}", exc_info=True)
        raise HttpError(500, "Deliberation failed. Please try again.")

    return result