# Location: backend/auth/tokens.py
"""
Stateless signed session tokens.

Format:  "<user_id>.<token_version>.<issued_at_unix>.<hmac_sha256_hex>"
TTL is enforced by comparing issued_at against COOKIE_MAX_AGE from config.

token_version lets the server invalidate tokens it never stored: the caller
compares the version in the token against the user's current token_version in
the database, and bumping that column makes every existing token fail.
"""
import hmac
import hashlib
import time
from typing import Optional

from config import SECRET_KEY, COOKIE_MAX_AGE


def _sign(user_id: int, version: int, ts: int) -> str:
    msg = f"{user_id}:{version}:{ts}".encode()
    return hmac.new(SECRET_KEY.encode(), msg, hashlib.sha256).hexdigest()


def make_token(user_id: int, version: int = 1) -> str:
    ts  = int(time.time())
    sig = _sign(user_id, version, ts)
    return f"{user_id}.{version}.{ts}.{sig}"


def verify_token(token: str) -> Optional[tuple[int, int]]:
    """
    Return (user_id, token_version) if the signature is valid and the token
    is unexpired, else None.

    The caller must still check token_version against the database — this
    function only proves the token wasn't forged or altered.
    """
    try:
        # split(".", 3) caps at exactly 4 parts. Without the 3, a token with
        # extra dots (tampering, encoding issue, future format change) yields
        # more elements and unpacking raises ValueError. Capping keeps the
        # parser robust regardless of what's in the signature.
        uid_str, ver_str, ts_str, sig = token.split(".", 3)
        user_id = int(uid_str)
        version = int(ver_str)
        ts      = int(ts_str)
    except (ValueError, AttributeError):
        return None

    if int(time.time()) - ts > COOKIE_MAX_AGE:
        return None

    if not hmac.compare_digest(_sign(user_id, version, ts), sig):
        return None

    return user_id, version