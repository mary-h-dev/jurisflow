from ninja.errors import HttpError

from core.auth import create_access_token, create_refresh_token, decode_token
from .models import User, UserProfile, LegalField
from .schemas import TokenOut, UserOut, OnboardingStatusOut, LegalFieldOut


class AuthService:

    def register(self, email: str, password: str, full_name: str) -> UserOut:
        if User.objects.filter(email=email).exists():
            raise HttpError(400, "این ایمیل قبلاً ثبت شده است.")

        user = User.objects.create_user(
            email=email,
            password=password,
            full_name=full_name,
        )
        # ساخت پروفایل همزمان با ثبت‌نام
        UserProfile.objects.create(user=user)

        return UserOut.model_validate(user)

    def login(self, request, email: str, password: str) -> TokenOut:
        from django.contrib.auth import authenticate
        user = authenticate(request, username=email, password=password)
        if not user:
            raise HttpError(401, "ایمیل یا رمز عبور اشتباه است.")

        return TokenOut(
            access=create_access_token(user.id),
            refresh=create_refresh_token(user.id),
        )

    def refresh(self, refresh_token: str) -> TokenOut:
        try:
            payload = decode_token(refresh_token)
            if payload.get("kind") != "refresh":
                raise HttpError(401, "توکن نامعتبر است.")
            user = User.objects.get(id=payload["user_id"], is_active=True)
        except HttpError:
            raise
        except Exception:
            raise HttpError(401, "توکن منقضی یا نامعتبر است.")

        return TokenOut(
            access=create_access_token(user.id),
            refresh=create_refresh_token(user.id),
        )

    def get_me(self, user: User) -> UserOut:
        return UserOut.model_validate(user)


class OnboardingService:

    def _get_profile(self, user: User) -> UserProfile:
        try:
            return user.profile
        except UserProfile.DoesNotExist:
            raise HttpError(404, "پروفایل کاربر یافت نشد.")

    def set_role(self, user: User, role: str) -> OnboardingStatusOut:
        VALID_ROLES = {"lawyer", "client"}
        if role not in VALID_ROLES:
            raise HttpError(400, f"نقش نامعتبر است. مقادیر مجاز: {VALID_ROLES}")

        profile = self._get_profile(user)

        if profile.is_onboarding_complete:
            raise HttpError(400, "فرآیند onboarding قبلاً تکمیل شده است.")

        profile.role            = role
        profile.onboarding_step = "2"
        profile.save(update_fields=["role", "onboarding_step", "updated_at"])

        return self._build_status(profile)

    def set_legal_fields(self, user: User, field_ids: list[int]) -> OnboardingStatusOut:
        profile = self._get_profile(user)

        if profile.is_onboarding_complete:
            raise HttpError(400, "فرآیند onboarding قبلاً تکمیل شده است.")

        if not profile.role:
            raise HttpError(400, "ابتدا باید نقش خود را تعیین کنید.")

        # چک کن همه id ها معتبرن
        fields = LegalField.objects.filter(id__in=field_ids, is_active=True)
        if fields.count() != len(field_ids):
            raise HttpError(400, "یک یا چند زمینه حقوقی نامعتبر است.")

        if not fields.exists():
            raise HttpError(400, "حداقل یک زمینه حقوقی انتخاب کنید.")

        profile.legal_fields.set(fields)
        profile.onboarding_step        = "done"
        profile.is_onboarding_complete = True
        profile.save(update_fields=[
            "onboarding_step",
            "is_onboarding_complete",
            "updated_at",
        ])

        return self._build_status(profile)

    def get_status(self, user: User) -> OnboardingStatusOut:
        profile = self._get_profile(user)
        return self._build_status(profile)

    def get_legal_fields(self) -> list[LegalFieldOut]:
        """لیست همه زمینه‌های حقوقی فعال"""
        fields = LegalField.objects.filter(is_active=True)
        return [LegalFieldOut.model_validate(f) for f in fields]

    def _build_status(self, profile: UserProfile) -> OnboardingStatusOut:
        return OnboardingStatusOut(
            onboarding_step=profile.onboarding_step,
            is_onboarding_complete=profile.is_onboarding_complete,
            role=profile.role or None,
            legal_fields=[
                LegalFieldOut.model_validate(f)
                for f in profile.legal_fields.all()
            ],
        )


auth_service       = AuthService()
onboarding_service = OnboardingService()