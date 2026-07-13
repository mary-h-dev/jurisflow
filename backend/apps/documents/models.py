from django.db import models
from apps.users.models import User


class Document(models.Model):
    """
    مدل ذخیره‌سازی اسناد آپلودشده.
    Flow: uploaded → processing → completed | failed
    """

    class Status(models.TextChoices):
        UPLOADED   = "uploaded",   "آپلود شده"
        PROCESSING = "processing", "در حال پردازش"
        COMPLETED  = "completed",  "تکمیل شده"
        FAILED     = "failed",     "ناموفق"

    class Engine(models.TextChoices):
        GEMINI = "gemini", "Gemini Flash"
        GLM    = "glm",    "GLM-OCR"

    class FileType(models.TextChoices):
        PDF   = "pdf",   "PDF"
        IMAGE = "image", "تصویر"

    # ── روابط ─────────────────────────────────────────────────────────────────
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="documents",
    )

    # ── فایل ──────────────────────────────────────────────────────────────────
    filename  = models.CharField(max_length=255)
    file_size = models.PositiveIntegerField(help_text="bytes")
    file      = models.FileField(upload_to="documents/%Y/%m/%d/")
    file_type = models.CharField(
        max_length=10,
        choices=FileType.choices,
        default=FileType.PDF,
    )

    # ── وضعیت ─────────────────────────────────────────────────────────────────
    status     = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.UPLOADED,
    )
    ocr_engine = models.CharField(
        max_length=20,
        choices=Engine.choices,
        default=Engine.GEMINI,
    )

    # ── خروجی OCR ─────────────────────────────────────────────────────────────
    extracted_text = models.TextField(blank=True)
    page_count     = models.PositiveIntegerField(default=0)
    confidence     = models.FloatField(default=0.0)
    error_message  = models.TextField(blank=True)

    # ── Celery task ───────────────────────────────────────────────────────────
    task_id = models.CharField(max_length=255, blank=True)

    # ── زمان‌ها ───────────────────────────────────────────────────────────────
    created_at   = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        verbose_name        = "سند"
        verbose_name_plural = "اسناد"
        ordering            = ["-created_at"]

    def __str__(self):
        return f"{self.filename} — {self.user.email}"

    @property
    def is_ready(self) -> bool:
        return self.status == self.Status.COMPLETED and bool(self.extracted_text)

    @property
    def file_size_mb(self) -> float:
        return round(self.file_size / 1024 / 1024, 2)