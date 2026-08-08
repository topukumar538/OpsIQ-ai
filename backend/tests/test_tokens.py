# Location: backend/tests/test_tokens.py
"""
Tests for HMAC-signed session token generation and verification.

These are pure unit tests — no DB, no network, no FastAPI app needed.
Run with: pytest tests/test_tokens.py -v

Token format: "<user_id>.<token_version>.<issued_at>.<hmac_hex>"
token_version lets the server invalidate tokens it never stored — logout
bumps the user's version in the database and every older token stops
verifying. verify_token() returns (user_id, token_version); the caller
compares the version against the DB.
"""
import time
import pytest

# Set a known secret key before importing tokens
import os
os.environ["SECRET_KEY"]    = "a" * 32
os.environ["GROQ_API_KEY"]  = "test-key-not-real"
os.environ["DATABASE_URL"]  = "postgresql+asyncpg://u:p@localhost/test"

from auth.tokens import make_token, verify_token


# ── make_token ────────────────────────────────────────────────────────────────

def test_make_token_returns_string():
    token = make_token(1)
    assert isinstance(token, str)


def test_make_token_has_four_parts():
    """user_id.token_version.issued_at.signature"""
    token = make_token(1)
    assert len(token.split(".")) == 4


def test_make_token_contains_user_id():
    token = make_token(42)
    assert token.split(".")[0] == "42"


def test_make_token_contains_version():
    token = make_token(42, 3)
    assert token.split(".")[1] == "3"


def test_make_token_version_defaults_to_one():
    token = make_token(42)
    assert token.split(".")[1] == "1"


def test_make_token_different_users_different_tokens():
    assert make_token(1) != make_token(2)


def test_make_token_same_user_different_timestamps():
    t1 = make_token(1)
    time.sleep(1)
    t2 = make_token(1)
    # Different timestamps mean different signatures
    assert t1 != t2


def test_make_token_different_versions_different_signatures():
    """Bumping the version must produce a different token, or revocation
    would be defeated by an attacker reusing the old signature."""
    t1 = make_token(5, 1)
    t2 = make_token(5, 2)
    assert t1.split(".")[3] != t2.split(".")[3]


# ── verify_token ──────────────────────────────────────────────────────────────

def test_verify_valid_token_returns_user_id_and_version():
    token = make_token(7)
    assert verify_token(token) == (7, 1)


def test_verify_returns_version_it_was_signed_with():
    token = make_token(7, 4)
    assert verify_token(token) == (7, 4)


def test_verify_returns_none_for_empty_string():
    assert verify_token("") is None


def test_verify_returns_none_for_garbage():
    assert verify_token("not.a.token") is None


def test_verify_returns_none_for_missing_parts():
    assert verify_token("123.456") is None


def test_verify_returns_none_for_extra_dots():
    # split(".", 3) caps at 4 parts, so extra dots land in the signature
    # and fail the HMAC check rather than crashing the unpack.
    token = make_token(1)
    assert verify_token(token + ".extra.dots") is None


def test_verify_returns_none_for_tampered_signature():
    parts = make_token(1).split(".")
    parts[3] = "a" * 64
    assert verify_token(".".join(parts)) is None


def test_verify_returns_none_for_tampered_user_id():
    parts = make_token(1).split(".")
    parts[0] = "999"
    assert verify_token(".".join(parts)) is None


def test_verify_returns_none_for_tampered_version():
    """The version is inside the signed message, so editing it in a stolen
    cookie can't bypass the revocation check."""
    parts = make_token(5, 1).split(".")
    parts[1] = "2"
    assert verify_token(".".join(parts)) is None


def test_verify_returns_none_for_tampered_timestamp():
    parts = make_token(1).split(".")
    parts[2] = "0"
    assert verify_token(".".join(parts)) is None


def test_verify_returns_none_for_expired_token(monkeypatch):
    # Make the token appear to have been created 8 days ago.
    # COOKIE_MAX_AGE is 7 days, so it should be expired.
    old_time = time.time() - (8 * 24 * 60 * 60)
    monkeypatch.setattr("auth.tokens.time.time", lambda: old_time)
    token = make_token(1)

    monkeypatch.undo()
    assert verify_token(token) is None


def test_verify_returns_none_for_non_integer_user_id():
    assert verify_token("abc.1.123.defsig") is None


def test_verify_returns_none_for_non_integer_version():
    assert verify_token("1.abc.123.defsig") is None


def test_verify_different_secret_key_rejects_token(monkeypatch):
    token = make_token(1)
    monkeypatch.setattr("auth.tokens.SECRET_KEY", "b" * 32)
    assert verify_token(token) is None


def test_verify_multiple_users_correct_ids():
    for user_id in [1, 42, 100, 99999]:
        assert verify_token(make_token(user_id)) == (user_id, 1)