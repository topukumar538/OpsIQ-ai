# Location: backend/tests/test_tokens.py
"""
Tests for HMAC-signed session token generation and verification.

These are pure unit tests — no DB, no network, no FastAPI app needed.
Run with: pytest tests/test_tokens.py -v
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


def test_make_token_has_three_parts():
    token = make_token(1)
    parts = token.split(".")
    assert len(parts) == 3


def test_make_token_contains_user_id():
    token = make_token(42)
    uid_str = token.split(".")[0]
    assert uid_str == "42"


def test_make_token_different_users_different_tokens():
    t1 = make_token(1)
    t2 = make_token(2)
    assert t1 != t2


def test_make_token_same_user_different_timestamps():
    t1 = make_token(1)
    time.sleep(1)
    t2 = make_token(1)
    # Different timestamps mean different signatures
    assert t1 != t2


# ── verify_token ──────────────────────────────────────────────────────────────

def test_verify_valid_token_returns_user_id():
    token = make_token(7)
    assert verify_token(token) == 7


def test_verify_returns_none_for_empty_string():
    assert verify_token("") is None


def test_verify_returns_none_for_garbage():
    assert verify_token("not.a.token") is None


def test_verify_returns_none_for_missing_parts():
    assert verify_token("123.456") is None


def test_verify_returns_none_for_extra_dots():
    # Token with extra dots should not crash — split(".", 2) handles this
    token = make_token(1)
    tampered = token + ".extra.dots"
    assert verify_token(tampered) is None


def test_verify_returns_none_for_tampered_signature():
    token = make_token(1)
    parts = token.split(".")
    parts[2] = "a" * 64   # replace sig with garbage
    tampered = ".".join(parts)
    assert verify_token(tampered) is None


def test_verify_returns_none_for_tampered_user_id():
    token = make_token(1)
    parts = token.split(".")
    parts[0] = "999"   # change user_id — sig no longer matches
    tampered = ".".join(parts)
    assert verify_token(tampered) is None


def test_verify_returns_none_for_tampered_timestamp():
    token = make_token(1)
    parts = token.split(".")
    parts[1] = "0"   # change timestamp — sig no longer matches
    tampered = ".".join(parts)
    assert verify_token(tampered) is None


def test_verify_returns_none_for_expired_token(monkeypatch):
    # Make the token appear to have been created 8 days ago
    # COOKIE_MAX_AGE is 7 days so this should be expired
    old_time = time.time() - (8 * 24 * 60 * 60)
    monkeypatch.setattr("auth.tokens.time.time", lambda: old_time)
    token = make_token(1)

    # Restore real time for verification
    monkeypatch.undo()
    assert verify_token(token) is None


def test_verify_returns_none_for_non_integer_user_id():
    # Manually construct a malformed token
    assert verify_token("abc.123.defsig") is None


def test_verify_different_secret_key_rejects_token(monkeypatch):
    # Token signed with key A should be rejected when key B is used
    token = make_token(1)
    monkeypatch.setattr("auth.tokens.SECRET_KEY", "b" * 32)
    assert verify_token(token) is None


def test_verify_multiple_users_correct_ids():
    for user_id in [1, 42, 100, 99999]:
        token = make_token(user_id)
        assert verify_token(token) == user_id