from ninja import Router

from core.auth import jwt_auth
from .schemas import RegisterIn, LoginIn, RefreshIn, TokenOut, UserOut, RoleIn, ProfileOut
from .services import auth_service, profile_service

router = Router()


# Auth
@router.post("/register", response=UserOut)
def register(request, data: RegisterIn):
    return auth_service.register(
        email=data.email,
        password=data.password,
        full_name=data.full_name,
    )


@router.post("/login", response=TokenOut)
def login(request, data: LoginIn):
    return auth_service.login(request, data.email, data.password)


@router.post("/refresh", response=TokenOut)
def refresh(request, data: RefreshIn):
    return auth_service.refresh(data.refresh)


@router.get("/me", auth=jwt_auth, response=UserOut)
def me(request):
    return auth_service.get_me(request.user)


# Profile
@router.post("/profile/role", auth=jwt_auth, response=ProfileOut)
def set_role(request, data: RoleIn):
    return profile_service.set_role(request.user, data.role)


@router.get("/profile", auth=jwt_auth, response=ProfileOut)
def get_profile(request):
    return profile_service.get_profile(request.user)