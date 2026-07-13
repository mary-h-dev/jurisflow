from pydantic import BaseModel
from typing import Optional
from datetime import datetime


# ── Output ────────────────────────────────────────────────────────────────────

class DocumentOut(BaseModel):
    """اطلاعات سند برای نمایش"""
    id:             int
    filename:       str
    file_size_mb:   float
    status:         str
    ocr_engine:     str
    page_count:     int
    confidence:     float
    error_message:  Optional[str] = None
    created_at:     datetime
    processed_at:   Optional[datetime] = None
    is_ready:       bool

    class Config:
        from_attributes = True


class DocumentTextOut(BaseModel):
    """متن استخراج‌شده برای ارسال به agent"""
    id:             int
    filename:       str
    extracted_text: str
    page_count:     int
    confidence:     float

    class Config:
        from_attributes = True


class UploadOut(BaseModel):
    """خروجی بعد از آپلود"""
    id:      int
    task_id: str
    message: str


class StatusOut(BaseModel):
    """وضعیت پردازش"""
    id:      int
    status:  str
    message: str