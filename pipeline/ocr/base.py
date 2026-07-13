from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class OCRResult:
    """خروجی استاندارد همه OCR engine ها"""
    text:       str            # متن استخراج‌شده
    page_count: int            # تعداد صفحات
    confidence: float          # میانگین اطمینان ۰ تا ۱
    engine:     str            # gemini | glm
    metadata:   dict           # اطلاعات اضافه (اختیاری)


class BaseOCR(ABC):
    """
    Interface مشترک برای همه OCR engine ها.
    هر engine باید این رو پیاده کنه.
    """

    @abstractmethod
    def extract_from_bytes(
        self,
        file_bytes: bytes,
        filename:   str,
    ) -> OCRResult:
        """
        استخراج متن از bytes فایل PDF.
        این متد باید در هر engine پیاده‌سازی بشه.
        """
        ...

    @abstractmethod
    def extract_from_path(
        self,
        file_path: str,
    ) -> OCRResult:
        """استخراج متن از مسیر فایل"""
        ...

    def is_available(self) -> bool:
        """
        چک می‌کنه engine در دسترسه یا نه.
        برای health check استفاده میشه.
        """
        try:
            self._check_availability()
            return True
        except Exception:
            return False

    @abstractmethod
    def _check_availability(self) -> None:
        """چک در دسترس بودن — اگه نبود exception بده"""
        ...