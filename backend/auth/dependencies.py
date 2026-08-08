# Location: backend/auth/dependencies.py
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from auth.database import get_db
from auth.models import User
from auth.tokens import verify_token
from config import COOKIE_NAME


async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> User:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    parsed = verify_token(token)
    if parsed is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired")

    user_id, token_version = parsed

    result = await db.execute(select(User).where(User.id == user_id))
    user   = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    # Reject tokens issued before the last logout. We already fetch the user
    # on every request, so this check is free.
    if token_version != user.token_version:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired")

    return user


async def get_current_user_optional(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """Returns None instead of raising when not authenticated."""
    try:
        return await get_current_user(request, db)
    except HTTPException:
        return None