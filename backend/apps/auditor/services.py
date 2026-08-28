from __future__ import annotations

import logging
import time

from django.conf import settings

from apps.search.services import SearchResult

from .checklist_builder import build_evidence_bundles
from .confidence import combine_confidence, compute_auditor_confidence
from .pruning import decide, prune
from .schemas import AuditorOut, CombinedConfidenceOut
from .verifier import AuditorAPIError, VerificationError, verify_article

logger = logging.getLogger(__name__)

_DEFAULT_REQUEST_DELAY_SECONDS = 0.5  # spacing between calls; lower this now that OpenRouter is a paid tier, not a free-tier rate limit


class AuditorService:

    def audit(self, search_result: SearchResult) -> AuditorOut:
        bundles = build_evidence_bundles(search_result)
        delay = getattr(settings, "AUDITOR_LLM_REQUEST_DELAY_SECONDS", _DEFAULT_REQUEST_DELAY_SECONDS)

        audited = []
        for i, bundle in enumerate(bundles):
            if i > 0 and delay:
                time.sleep(delay)

            try:
                result = verify_article(search_result.query, bundle)
            except AuditorAPIError as e:
                logger.error(f"Skipping article {bundle.article_ref}: {e}")
                continue
            except VerificationError as e:
                logger.warning(f"Skipping article {bundle.article_ref}: {e}")
                continue
            audited.append(decide(bundle.article_ref, result.checklist, result.topically_relevant))

        auditor_out = AuditorOut(
            query=search_result.query,
            verified_articles=audited,
            pruned_articles=prune(audited),
            confidence=CombinedConfidenceOut(
                score=0, level="low",
                retrieval_score=0, auditor_score=0, prune_ratio=0,
        ),
        )
        auditor_conf = compute_auditor_confidence(auditor_out)
        combined = combine_confidence(search_result.confidence, auditor_conf)
        auditor_out.confidence = CombinedConfidenceOut(
            score=combined.score,
            level=combined.level,
            note=combined.note,
            retrieval_score=combined.retrieval_score,
            auditor_score=combined.auditor_score,
            prune_ratio=combined.prune_ratio,
        )
        return auditor_out


auditor_service = AuditorService()