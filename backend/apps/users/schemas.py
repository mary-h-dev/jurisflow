from pydantic import BaseModel, EmailStr
from typing import Optional


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


class UserOut(BaseModel):
    id:        int
    email:     str
    full_name: str

    class Config:
        from_attributes = True


class RoleIn(BaseModel):
    role: str  # lawyer | student | citizen


class ProfileOut(BaseModel):
    role: Optional[str]

    class Config:
        from_attributes = True