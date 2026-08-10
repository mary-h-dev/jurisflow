from ninja.errors import HttpError

from core.auth import create_access_token, create_refresh_token, decode_token
from .models import User, UserProfile
from .schemas import TokenOut, UserOut, ProfileOut

VALID_ROLES = {"lawyer", "student", "citizen"}


class AuthService:

    def register(self, email: str, password: str, full_name: str) -> UserOut:
        if User.objects.filter(email=email).exists():
            raise HttpError(400, "This email is already registered.")

        user = User.objects.create_user(
            email=email,
            password=password,
            full_name=full_name,
        )
        UserProfile.objects.create(user=user)
        return UserOut.model_validate(user)

    def login(self, request, email: str, password: str) -> TokenOut:
        from django.contrib.auth import authenticate
        user = authenticate(request, username=email, password=password)
        if not user:
            raise HttpError(401, "Invalid email or password.")

        return TokenOut(
            access=create_access_token(user.id),
            refresh=create_refresh_token(user.id),
        )

    def refresh(self, refresh_token: str) -> TokenOut:
        try:
            payload = decode_token(refresh_token)
            if payload.get("kind") != "refresh":
                raise HttpError(401, "Invalid token.")
            user = User.objects.get(id=payload["user_id"], is_active=True)
        except HttpError:
            raise
        except Exception:
            raise HttpError(401, "Token expired or invalid.")

        return TokenOut(
            access=create_access_token(user.id),
            refresh=create_refresh_token(user.id),
        )

    def get_me(self, user: User) -> UserOut:
        return UserOut.model_validate(user)


class ProfileService:

    def set_role(self, user: User, role: str) -> ProfileOut:
        if role not in VALID_ROLES:
            raise HttpError(400, f"Invalid role. Allowed: {sorted(VALID_ROLES)}")

        profile = user.profile
        profile.role = role
        profile.save(update_fields=["role", "updated_at"])
        return ProfileOut.model_validate(profile)

    def get_profile(self, user: User) -> ProfileOut:
        return ProfileOut.model_validate(user.profile)


auth_service    = AuthService()
profile_service = ProfileService()