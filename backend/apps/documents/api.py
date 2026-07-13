from __future__ import annotations

import logging

from ninja import Router, File
from ninja.errors import HttpError
from ninja.files import UploadedFile

from core.auth import jwt_auth
from core.permissions import check_plan_limit
from .models import Document
from .schemas import DocumentOut, DocumentTextOut, UploadOut, StatusOut
from .tasks import process_ocr

logger = logging.getLogger(__name__)
router = Router()

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

ALLOWED_TYPES = {
    # PDF
    "application/pdf",
    # تصاویر — عکس موبایل و اسکن
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/heic",   # عکس آیفون
    "image/heif",
}

IMAGE_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/webp", "image/heic", "image/heif",
}


def _detect_file_type(content_type: str) -> str:
    """تشخیص نوع فایل"""
    if content_type in IMAGE_TYPES:
        return Document.FileType.IMAGE
    return Document.FileType.PDF


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    auth=jwt_auth,
    response=UploadOut,
    summary="آپلود سند (PDF یا تصویر)",
)
def upload_document(request, file: UploadedFile = File(...)):
    """
    آپلود سند و شروع پردازش OCR.

    فرمت‌های مجاز:
    - PDF (متن‌محور یا اسکن‌شده)
    - JPG, PNG, WEBP (اسکن یا عکس موبایل)
    - HEIC (عکس آیفون)

    حداکثر حجم: ۱۰MB
    """
    user = request.user

    # چک فرمت
    if file.content_type not in ALLOWED_TYPES:
        raise HttpError(
            400,
            "فرمت فایل پشتیبانی نمی‌شود. "
            "فرمت‌های مجاز: PDF, JPG, PNG, WEBP, HEIC"
        )

    # چک حجم
    if file.size > MAX_FILE_SIZE:
        raise HttpError(
            400,
            f"حجم فایل بیش از حد مجاز است. حداکثر {MAX_FILE_SIZE // 1024 // 1024}MB"
        )

    try:
        file_type = _detect_file_type(file.content_type)

        doc = Document.objects.create(
            user=user,
            filename=file.name,
            file_size=file.size,
            file=file,
            file_type=file_type,
            status=Document.Status.UPLOADED,
        )

        # شروع Celery task
        task = process_ocr.delay(doc.id)
        doc.task_id = task.id
        doc.save(update_fields=["task_id"])

        logger.info(
            f"[documents/upload] user={user.id} "
            f"doc={doc.id} type={file_type} task={task.id}"
        )

        return UploadOut(
            id=doc.id,
            task_id=task.id,
            message="فایل با موفقیت آپلود شد. پردازش در حال انجام است.",
        )

    except Exception as e:
        logger.error(f"[documents/upload] error: {e}", exc_info=True)
        raise HttpError(500, "خطا در آپلود فایل.")


@router.get(
    "/{document_id}/status",
    auth=jwt_auth,
    response=StatusOut,
    summary="وضعیت پردازش سند",
)
def get_status(request, document_id: int):
    """وضعیت realtime پردازش OCR"""
    try:
        doc = Document.objects.get(id=document_id, user=request.user)
    except Document.DoesNotExist:
        raise HttpError(404, "سند یافت نشد.")

    messages = {
        "uploaded":   "در صف پردازش",
        "processing": "در حال استخراج متن",
        "completed":  "پردازش تکمیل شد",
        "failed":     f"پردازش ناموفق: {doc.error_message}",
    }

    return StatusOut(
        id=doc.id,
        status=doc.status,
        message=messages.get(doc.status, "وضعیت نامشخص"),
    )


@router.get(
    "/{document_id}",
    auth=jwt_auth,
    response=DocumentOut,
    summary="اطلاعات سند",
)
def get_document(request, document_id: int):
    """اطلاعات کامل سند"""
    try:
        doc = Document.objects.get(id=document_id, user=request.user)
    except Document.DoesNotExist:
        raise HttpError(404, "سند یافت نشد.")
    return doc


@router.get(
    "/{document_id}/text",
    auth=jwt_auth,
    response=DocumentTextOut,
    summary="متن استخراج‌شده",
)
def get_text(request, document_id: int):
    """
    متن استخراج‌شده رو برمیگردونه.
    برای ارسال دستی به /api/agents/analyze استفاده میشه.
    """
    try:
        doc = Document.objects.get(id=document_id, user=request.user)
    except Document.DoesNotExist:
        raise HttpError(404, "سند یافت نشد.")

    if not doc.is_ready:
        raise HttpError(
            400,
            f"سند هنوز آماده نیست. وضعیت فعلی: {doc.status}"
        )
    return doc


@router.post(
    "/{document_id}/analyze",
    auth=jwt_auth,
    summary="تحلیل مستقیم سند با Agent",
)
def analyze_document(request, document_id: int, case_context: str = ""):
    """
    Shortcut: OCR + Agent در یک مرحله.
    نیاز به پلن Pro یا بالاتر دارد.
    """
    user = request.user

    if user.plan == "free":
        raise HttpError(
            403,
            "تحلیل پرونده فقط برای کاربران Pro و Enterprise در دسترس است."
        )

    check_plan_limit(user)

    try:
        doc = Document.objects.get(id=document_id, user=user)
    except Document.DoesNotExist:
        raise HttpError(404, "سند یافت نشد.")

    if not doc.is_ready:
        raise HttpError(
            400,
            f"سند هنوز آماده نیست. وضعیت: {doc.status}. "
            "لطفاً ابتدا وضعیت را با /status چک کنید."
        )

    from apps.agents.graph import case_graph, create_initial_state
    from apps.agents.api import _build_response

    try:
        initial_state = create_initial_state(
            extracted_text=doc.extracted_text,
            user_id=user.id,
            case_context=case_context,
        )

        logger.info(
            f"[documents/analyze] user={user.id} "
            f"doc={document_id} "
            f"session={initial_state['session_id']}"
        )

        final_state = case_graph.invoke(initial_state)
        user.can_query_and_increment()

        return _build_response(final_state)

    except Exception as e:
        logger.error(f"[documents/analyze] error: {e}", exc_info=True)
        raise HttpError(500, "خطا در تحلیل سند.")


@router.get(
    "/",
    auth=jwt_auth,
    response=list[DocumentOut],
    summary="لیست اسناد کاربر",
)
def list_documents(request):
    """لیست همه اسناد آپلودشده"""
    return list(Document.objects.filter(user=request.user))


@router.delete(
    "/{document_id}",
    auth=jwt_auth,
    summary="حذف سند",
)
def delete_document(request, document_id: int):
    """حذف سند و فایل مربوطه"""
    try:
        doc = Document.objects.get(id=document_id, user=request.user)
    except Document.DoesNotExist:
        raise HttpError(404, "سند یافت نشد.")

    doc.file.delete(save=False)
    doc.delete()
    return {"message": "سند با موفقیت حذف شد."}