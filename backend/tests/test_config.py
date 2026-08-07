"""
Tests for config validation.

Settings() normally reads .env files and the process environment. Every test
passes _env_file=None plus explicit values, so these test the validators
themselves rather than whatever happens to be in the developer's .env.
"""
import pytest
from pydantic import ValidationError

from config import Settings

# Minimum valid config. Each test starts here and overrides one field.
VALID = {
    "groq_api_key": "gsk_realkeylooking_value_123",
    "secret_key"  : "a" * 64,
    "_env_file"   : None,
}


def make(**overrides) -> Settings:
    return Settings(**{**VALID, **overrides})  # type: ignore


# ── Secret key ────────────────────────────────────────────────────────────────

def test_valid_secret_key_accepted():
    assert make(secret_key="f" * 64).secret_key == "f" * 64


def test_empty_secret_key_rejected():
    with pytest.raises(ValidationError):
        make(secret_key="")


def test_short_secret_key_rejected():
    with pytest.raises(ValidationError):
        make(secret_key="tooshort")


def test_known_placeholder_secret_key_rejected():
    with pytest.raises(ValidationError):
        make(secret_key="your_secret_key_here")


def test_instruction_text_as_secret_key_rejected():
    """Regression: this string is 58 chars, so the length check alone passed it."""
    bad = 'generate_with_python_-c_"import_secrets;print(secrets.token_hex(32))"'
    with pytest.raises(ValidationError):
        make(secret_key=bad)


def test_secret_key_with_spaces_rejected():
    with pytest.raises(ValidationError):
        make(secret_key="please replace this with a real key value ok")


# ── Groq key ──────────────────────────────────────────────────────────────────

def test_empty_groq_key_rejected():
    with pytest.raises(ValidationError):
        make(groq_api_key="")


def test_placeholder_groq_key_rejected():
    with pytest.raises(ValidationError):
        make(groq_api_key="your_groq_api_key_here")


# ── Temperatures ──────────────────────────────────────────────────────────────

def test_temperature_in_range_accepted():
    assert make(chat_temperature=1.5).chat_temperature == 1.5


def test_temperature_above_two_rejected():
    with pytest.raises(ValidationError):
        make(chat_temperature=2.1)


def test_negative_temperature_rejected():
    with pytest.raises(ValidationError):
        make(rag_temperature=-0.1)


# ── Bcrypt ────────────────────────────────────────────────────────────────────

def test_bcrypt_rounds_too_low_rejected():
    with pytest.raises(ValidationError):
        make(bcrypt_rounds=3)


def test_bcrypt_rounds_too_high_rejected():
    with pytest.raises(ValidationError):
        make(bcrypt_rounds=32)


# ── Chunk overlap ─────────────────────────────────────────────────────────────

def test_rag_overlap_equal_to_chunk_size_rejected():
    with pytest.raises(ValidationError):
        make(rag_chunk_size=500, rag_chunk_overlap=500)


def test_rag_overlap_larger_than_chunk_size_rejected():
    with pytest.raises(ValidationError):
        make(rag_chunk_size=100, rag_chunk_overlap=200)


def test_pm_overlap_equal_to_chunk_lines_rejected():
    """Equal values make the log chunker advance zero lines per step."""
    with pytest.raises(ValidationError):
        make(pm_chunk_lines=30, pm_overlap_lines=30)


# ── Origins parsing ───────────────────────────────────────────────────────────

def test_allowed_origins_splits_on_comma():
    s = make(allowed_origins="http://a.com,http://b.com")
    assert s.allowed_origins_list == ["http://a.com", "http://b.com"]


def test_allowed_origins_strips_whitespace():
    s = make(allowed_origins=" http://a.com , http://b.com ")
    assert s.allowed_origins_list == ["http://a.com", "http://b.com"]


def test_allowed_origins_ignores_empty_entries():
    s = make(allowed_origins="http://a.com,,http://b.com,")
    assert s.allowed_origins_list == ["http://a.com", "http://b.com"]


# ── SSL flag ──────────────────────────────────────────────────────────────────

def test_db_ssl_defaults_to_false():
    """Local and Docker Postgres have SSL off; defaulting on breaks both."""
    assert make().db_ssl is False


def test_db_ssl_can_be_enabled():
    assert make(db_ssl=True).db_ssl is True