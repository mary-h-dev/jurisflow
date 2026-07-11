from ninja import Router
from django.conf import settings

from core.auth import jwt_auth
from .schemas import (
    RegisterIn, LoginIn, RefreshIn,
    TokenOut, UserOut,
    RoleIn, LegalFieldsIn, LegalFieldOut, OnboardingStatusOut,
)
from .services import auth_service, onboarding_service

router = Router()


# ── Auth ──────────────────────────────────────────────────────────────────────

@router.post("/register", response=UserOut, summary="ثبت‌نام کاربر جدید")
def register(request, data: RegisterIn):
    return auth_service.register(
        email=data.email,
        password=data.password,
        full_name=data.full_name,
    )


@router.post("/login", response=TokenOut, summary="ورود و دریافت توکن")
def login(request, data: LoginIn):
    return auth_service.login(request, data.email, data.password)


@router.post("/refresh", response=TokenOut, summary="تجدید توکن")
def refresh(request, data: RefreshIn):
    return auth_service.refresh(data.refresh)


@router.get("/me", auth=jwt_auth, response=UserOut, summary="اطلاعات کاربر جاری")
def me(request):
    return auth_service.get_me(request.user)


@router.get("/limits", auth=jwt_auth, summary="وضعیت مصرف روزانه")
def limits(request):
    return {
        "plan": request.user.plan,
        **request.user.get_daily_usage(),
    }


# ── Onboarding ────────────────────────────────────────────────────────────────

@router.get(
    "/onboarding/legal-fields",
    auth=jwt_auth,
    response=list[LegalFieldOut],
    summary="لیست زمینه‌های حقوقی",
)
def get_legal_fields(request):
    """لیست همه زمینه‌های حقوقی فعال برای نمایش در مرحله ۳"""
    return onboarding_service.get_legal_fields()


@router.post(
    "/onboarding/role",
    auth=jwt_auth,
    response=OnboardingStatusOut,
    summary="مرحله ۲: تعیین نقش کاربر",
)
def set_role(request, data: RoleIn):
    return onboarding_service.set_role(request.user, data.role)


@router.post(
    "/onboarding/legal-fields",
    auth=jwt_auth,
    response=OnboardingStatusOut,
    summary="مرحله ۳: انتخاب زمینه‌های حقوقی",
)
def set_legal_fields(request, data: LegalFieldsIn):
    return onboarding_service.set_legal_fields(request.user, data.field_ids)


@router.get(
    "/onboarding/status",
    auth=jwt_auth,
    response=OnboardingStatusOut,
    summary="وضعیت onboarding کاربر",
)
def onboarding_status(request):
    return onboarding_service.get_status(request.user)