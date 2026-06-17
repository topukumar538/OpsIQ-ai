# Location: backend/auth/tokens.py
"""
Stateless signed session tokens.

Format:  "<user_id>.<issued_at_unix>.<hmac_sha256_hex>"
TTL is enforced by comparing issued_at against COOKIE_MAX_AGE from config.
"""
import hmac
import hashlib
import time
from typing import Optional

from config import SECRET_KEY, COOKIE_MAX_AGE


def _sign(user_id: int, ts: int) -> str:
    msg = f"{user_id}:{ts}".encode()
    return hmac.new(SECRET_KEY.encode(), msg, hashlib.sha256).hexdigest()


def make_token(user_id: int) -> str:
    ts  = int(time.time())
    sig = _sign(user_id, ts)
    return f"{user_id}.{ts}.{sig}"


def verify_token(token: str) -> Optional[int]:
    """Return user_id if the token is valid and unexpired, else None."""
    try:
        # split(".", 2) caps at exactly 3 parts.
        # Without the 2, a token with extra dots (tampered cookie, encoding
        # issue, future format change) returns more than 3 elements and the
        # unpacking raises ValueError — silently caught as unauthenticated.
        # The 2 makes the parser robust regardless of what's in the sig.
        uid_str, ts_str, sig = token.split(".", 2)
        user_id = int(uid_str)
        ts      = int(ts_str)
    except (ValueError, AttributeError):
        return None

    if int(time.time()) - ts > COOKIE_MAX_AGE:
        return None

    if not hmac.compare_digest(_sign(user_id, ts), sig):
        return None

    return user_id