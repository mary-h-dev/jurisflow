from __future__ import annotations

import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

from celery import shared_task
from django.utils import timezone

logger = logging.getLogger(__name__)


@shared_task(
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    name="documents.process_ocr",
)
def process_ocr(self, document_id: int) -> dict:
    """
    Celery task برای پردازش OCR.
    پشتیبانی از PDF و تصویر.
    """
    from apps.documents.models import Document
    from pipeline.ocr import get_ocr_engine

    logger.info(f"[ocr_task] starting document_id={document_id}")

    try:
        doc = Document.objects.get(id=document_id)
    except Document.DoesNotExist:
        logger.error(f"[ocr_task] document {document_id} not found")
        return {"status": "error", "message": "سند یافت نشد"}

    doc.status = Document.Status.PROCESSING
    doc.save(update_fields=["status"])

    try:
        doc.file.open("rb")
        file_bytes = doc.file.read()
        doc.file.close()

        engine = get_ocr_engine()

        # بر اساس نوع فایل، متد مناسب رو انتخاب کن
        if doc.file_type == Document.FileType.IMAGE:
            result = engine.extract_from_image_bytes(
                file_bytes=file_bytes,
                filename=doc.filename,
            )
        else:
            result = engine.extract_from_bytes(
                file_bytes=file_bytes,
                filename=doc.filename,
            )

        if not result.text:
            raise ValueError("متنی از سند استخراج نشد")

        doc.extracted_text = result.text
        doc.page_count     = result.page_count
        doc.confidence     = result.confidence
        doc.ocr_engine     = result.engine.split("-")[0]
        doc.status         = Document.Status.COMPLETED
        doc.processed_at   = timezone.now()
        doc.save(update_fields=[
            "extracted_text", "page_count", "confidence",
            "ocr_engine", "status", "processed_at",
        ])

        logger.info(
            f"[ocr_task] completed document_id={document_id} "
            f"type={doc.file_type} pages={result.page_count} "
            f"chars={len(result.text)}"
        )

        return {
            "status":     "completed",
            "page_count": result.page_count,
            "char_count": len(result.text),
        }

    except Exception as e:
        logger.error(f"[ocr_task] failed document_id={document_id}: {e}", exc_info=True)

        try:
            raise self.retry(exc=e)
        except self.MaxRetriesExceededError:
            doc.status        = Document.Status.FAILED
            doc.error_message = str(e)
            doc.processed_at  = timezone.now()
            doc.save(update_fields=["status", "error_message", "processed_at"])
            return {"status": "failed", "error": str(e)}