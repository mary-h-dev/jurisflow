import jwt
from django.utils import timezone
from django.conf import settings
from ninja.security import HttpBearer

def create_access_token(user_id: int) -> str:
    payload = {
        "user_id": user_id,
        "kind":    "access",
        "exp":     timezone.now() + settings.JWT_ACCESS_LIFETIME,
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

def create_refresh_token(user_id: int) -> str:
    payload = {
        "user_id": user_id,
        "kind":    "refresh",
        "exp":     timezone.now() + settings.JWT_REFRESH_LIFETIME,
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

def decode_token(token: str) -> dict:
    return jwt.decode(
        token,
        settings.JWT_SECRET_KEY,
        algorithms=[settings.JWT_ALGORITHM],
    )

class JWTAuth(HttpBearer):
    def authenticate(self, request, token: str):
        try:
            payload = decode_token(token)
            if payload.get("kind") != "access":
                return None
            
            from apps.users.models import User
            user = User.objects.get(id=payload["user_id"], is_active=True)
            request.user = user
            return user
        except Exception:
            return None

jwt_auth = JWTAuth()