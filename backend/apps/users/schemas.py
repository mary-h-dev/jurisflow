from pydantic import BaseModel, EmailStr
from typing import Optional


# ── Auth ──────────────────────────────────────────────────────────────────────

class RegisterIn(BaseModel):
    email:     EmailStr
    password:  str
    full_name: str = ""


class LoginIn(BaseModel):
    email:    EmailStr
    password: str


class RefreshIn(BaseModel):
    refresh: str


class TokenOut(BaseModel):
    access:  str
    refresh: str


# ── User ──────────────────────────────────────────────────────────────────────

class UserOut(BaseModel):
    id:        int
    email:     str
    full_name: str
    plan:      str

    class Config:
        from_attributes = True


# ── Onboarding ────────────────────────────────────────────────────────────────

class RoleIn(BaseModel):
    role: str  # lawyer | client


class LegalFieldOut(BaseModel):
    id:   int
    name: str
    slug: str
    icon: str

    class Config:
        from_attributes = True


class LegalFieldsIn(BaseModel):
    field_ids: list[int]  # آی‌دی زمینه‌های انتخاب‌شده


class OnboardingStatusOut(BaseModel):
    onboarding_step:        str
    is_onboarding_complete: bool
    role:                   Optional[str]
    legal_fields:           list[LegalFieldOut]

    class Config:
        from_attributes = True