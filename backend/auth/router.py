# Location: backend/auth/router.py
import bcrypt
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from auth.database import get_db
from auth.dependencies import get_current_user
from auth.models import User
from auth.tokens import make_token
from config import (
    COOKIE_NAME,
    COOKIE_MAX_AGE,
    COOKIE_SECURE,
    COOKIE_SAMESITE,
    BCRYPT_ROUNDS,
)

router = APIRouter(prefix="/auth", tags=["auth"])

# Rate limiter — keyed by client IP address.
# Exported so main.py can register it on the app.
limiter = Limiter(key_func=get_remote_address)


# ── Cookie helpers ────────────────────────────────────────────────────────────

def _set_auth_cookie(response: JSONResponse, user_id: int) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value=make_token(user_id),
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
    )


def _clear_auth_cookie(response: JSONResponse) -> None:
    response.delete_cookie(
        key=COOKIE_NAME,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
    )


# ── Schemas ───────────────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    username: str
    email   : EmailStr
    password: str

    @field_validator("username")
    @classmethod
    def username_valid(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("Username must be at least 3 characters.")
        if len(v) > 64:
            raise ValueError("Username must be 64 characters or fewer.")
        return v

    @field_validator("password")
    @classmethod
    def password_valid(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters.")
        return v


class LoginRequest(BaseModel):
    email   : EmailStr
    password: str


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/signup", status_code=status.HTTP_201_CREATED)
@limiter.limit("5/minute")
# Why 5/minute: legitimate users never need more than 1-2 signup attempts.
# This blocks bots spamming account creation without affecting real users.
async def signup(
    request: Request,           # required first param for slowapi to read the IP
    body   : SignupRequest,
    db     : AsyncSession = Depends(get_db),
) -> JSONResponse:
    if (await db.execute(select(User).where(User.email == body.email))).scalar_one_or_none():
        raise HTTPException(
            status_code=409,
            detail="This email is already registered. Please sign in instead.",
        )

    pw_hash = bcrypt.hashpw(body.password.encode(), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode()
    user    = User(username=body.username, email=body.email, password_hash=pw_hash)
    db.add(user)
    await db.commit()
    await db.refresh(user)

    response = JSONResponse(
        content={"id": user.id, "username": user.username, "email": user.email},
        status_code=status.HTTP_201_CREATED,
    )
    _set_auth_cookie(response, user.id)
    return response


@router.post("/login")
@limiter.limit("10/minute")
# Why 10/minute: allows a user to mistype their password a few times without
# getting locked out, while still blocking automated brute-force attacks.
async def login(
    request: Request,           # required first param for slowapi to read the IP
    body   : LoginRequest,
    db     : AsyncSession = Depends(get_db),
) -> JSONResponse:
    result = await db.execute(select(User).where(User.email == body.email))
    user   = result.scalar_one_or_none()

    if user is None or not bcrypt.checkpw(body.password.encode(), user.password_hash.encode()):
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password. Please try again.",
        )

    response = JSONResponse(content={"id": user.id, "username": user.username, "email": user.email})
    _set_auth_cookie(response, user.id)
    return response


@router.post("/logout")
async def logout() -> JSONResponse:
    response = JSONResponse(content={"status": "logged out"})
    _clear_auth_cookie(response)
    return response


@router.get("/me")
async def me(current_user: User = Depends(get_current_user)) -> dict:
    return {
        "id"        : current_user.id,
        "username"  : current_user.username,
        "email"     : current_user.email,
        "created_at": current_user.created_at.isoformat(),
    }