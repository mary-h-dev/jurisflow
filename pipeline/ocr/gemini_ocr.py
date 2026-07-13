from __future__ import annotations

import base64
import logging
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .base import BaseOCR, OCRResult

load_dotenv()

logger = logging.getLogger(__name__)


class GeminiOCR(BaseOCR):
    """
    OCR با Gemini 1.5 Flash.
    پشتیبانی از PDF و تصویر (عکس موبایل، اسکن، دست‌نویس فارسی).
    """

    MODEL = "models/gemini-2.5-flash"

    PROMPT = """
این تصویر یا سند فارسی/حقوقی است.
وظیفه تو استخراج دقیق تمام متن است.

قوانین:
۱. تمام متن را دقیقاً استخراج کن — حتی اعداد و تاریخ‌ها
۲. ساختار پاراگراف‌ها را حفظ کن
۳. جداول را به صورت متن ساختاریافته بنویس
۴. اگه متنی ناخوانا بود، [ناخوانا] بنویس
۵. هیچ چیزی اضافه نکن — فقط متن موجود در سند

متن استخراج‌شده:
"""

    # نگاشت پسوند به mime type
    IMAGE_MIME_TYPES = {
        "jpg":  "image/jpeg",
        "jpeg": "image/jpeg",
        "png":  "image/png",
        "webp": "image/webp",
        "heic": "image/heic",
        "heif": "image/heif",
    }

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY یافت نشد.")
        self._client = genai.Client(api_key=self._api_key)

    def _check_availability(self) -> None:
        self._client.models.list()

    # ── PDF helpers ───────────────────────────────────────────────────────────

    def _is_scanned_pdf(self, file_bytes: bytes) -> bool:
        """تشخیص PDF اسکن‌شده از متن‌محور"""
        try:
            import pypdf, io
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            total_text = ""
            for page in reader.pages[:2]:
                total_text += page.extract_text() or ""
            return len(total_text.strip()) < 100
        except Exception:
            return True

    def _extract_text_direct(self, file_bytes: bytes) -> tuple[str, int]:
        """استخراج مستقیم از PDF متن‌محور"""
        try:
            import pypdf, io
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            pages_text = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    pages_text.append(f"--- صفحه {i + 1} ---\n{text.strip()}")
            return "\n\n".join(pages_text), len(reader.pages)
        except Exception as e:
            logger.error(f"[gemini_ocr] direct extraction failed: {e}")
            return "", 0

    def _extract_pdf_with_vision(self, file_bytes: bytes) -> tuple[str, int]:
        """استخراج از PDF اسکن‌شده با Gemini Vision"""
        try:
            import pypdf, io
            page_count = len(pypdf.PdfReader(io.BytesIO(file_bytes)).pages)
        except Exception:
            page_count = 1

        try:
            response = self._client.models.generate_content(
                model=self.MODEL,
                contents=[
                    types.Part.from_bytes(
                        data=file_bytes,
                        mime_type="application/pdf",
                    ),
                    self.PROMPT,
                ],
            )
            return response.text.strip(), page_count
        except Exception as e:
            logger.error(f"[gemini_ocr] PDF vision failed: {e}")
            return "", page_count

    # ── Image helpers ─────────────────────────────────────────────────────────

    def _detect_image_mime(self, file_bytes: bytes, filename: str) -> str:
        """تشخیص mime type تصویر"""
        # اول از پسوند فایل
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext in self.IMAGE_MIME_TYPES:
            return self.IMAGE_MIME_TYPES[ext]

        # بعد از magic bytes
        if file_bytes[:4] == b"\x89PNG":
            return "image/png"
        if file_bytes[:2] in (b"\xff\xd8", b"\xff\xe0", b"\xff\xe1"):
            return "image/jpeg"
        if file_bytes[:4] == b"RIFF":
            return "image/webp"

        return "image/jpeg"  # default

    def _extract_image_with_vision(
        self,
        file_bytes: bytes,
        filename:   str,
    ) -> str:
        """استخراج متن از تصویر با Gemini Vision"""
        mime_type = self._detect_image_mime(file_bytes, filename)

        response = self._client.models.generate_content(
            model=self.MODEL,
            contents=[
                types.Part.from_bytes(
                    data=file_bytes,
                    mime_type=mime_type,
                ),
                self.PROMPT,
            ],
        )
        return response.text.strip()

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_from_bytes(
        self,
        file_bytes: bytes,
        filename:   str = "document.pdf",
    ) -> OCRResult:
        """
        استخراج متن از PDF.
        Strategy:
        - متن‌محور → مستقیم (سریع‌تر)
        - اسکن‌شده → Gemini Vision
        """
        logger.info(f"[gemini_ocr] PDF: {filename} ({len(file_bytes)} bytes)")

        is_scanned = self._is_scanned_pdf(file_bytes)

        if not is_scanned:
            text, page_count = self._extract_text_direct(file_bytes)
            engine_note, confidence = "direct", 0.95
        else:
            text, page_count = self._extract_pdf_with_vision(file_bytes)
            engine_note, confidence = "vision", 0.85

        if not text:
            return OCRResult(
                text="",
                page_count=0,
                confidence=0.0,
                engine=f"gemini-{engine_note}",
                metadata={"filename": filename, "error": "no_text"},
            )

        logger.info(
            f"[gemini_ocr] PDF done: pages={page_count} "
            f"chars={len(text)} engine={engine_note}"
        )

        return OCRResult(
            text=text,
            page_count=page_count,
            confidence=confidence,
            engine=f"gemini-{engine_note}",
            metadata={
                "filename":   filename,
                "is_scanned": is_scanned,
                "char_count": len(text),
            },
        )

    def extract_from_image_bytes(
        self,
        file_bytes: bytes,
        filename:   str = "image.jpg",
    ) -> OCRResult:
        """
        استخراج متن از تصویر.
        پشتیبانی: JPG, PNG, WEBP, HEIC (عکس آیفون)
        کارایی: دست‌نویس فارسی، اسناد عکاسی‌شده با موبایل
        """
        logger.info(f"[gemini_ocr] Image: {filename} ({len(file_bytes)} bytes)")

        try:
            text = self._extract_image_with_vision(file_bytes, filename)

            if not text:
                return OCRResult(
                    text="",
                    page_count=1,
                    confidence=0.0,
                    engine="gemini-vision",
                    metadata={"filename": filename, "error": "no_text"},
                )

            logger.info(f"[gemini_ocr] Image done: chars={len(text)}")

            return OCRResult(
                text=text,
                page_count=1,
                confidence=0.80,  # تصویر موبایل کمی کمتر از scan حرفه‌ای
                engine="gemini-vision",
                metadata={
                    "filename":   filename,
                    "char_count": len(text),
                    "type":       "image",
                },
            )

        except Exception as e:
            logger.error(f"[gemini_ocr] image extraction failed: {e}", exc_info=True)
            return OCRResult(
                text="",
                page_count=1,
                confidence=0.0,
                engine="gemini-vision",
                metadata={"filename": filename, "error": str(e)},
            )

    def extract_from_path(self, file_path: str) -> OCRResult:
        """استخراج از مسیر فایل — تشخیص خودکار PDF یا تصویر"""
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        filename = os.path.basename(file_path)
        ext      = filename.rsplit(".", 1)[-1].lower()

        if ext == "pdf":
            return self.extract_from_bytes(file_bytes, filename)
        else:
            return self.extract_from_image_bytes(file_bytes, filename)