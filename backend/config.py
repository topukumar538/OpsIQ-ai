# Location: backend/config.py
import secrets
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


_SECRET_KEY_PLACEHOLDERS = {
    "",
    "change-me-in-production-use-a-long-random-string",
    "change-me",
    "secret",
    "secret_key",
    "your_secret_key_here",
}
_MIN_SECRET_KEY_LENGTH = 32


class Settings(BaseSettings):
    """
    All configuration is declared here with types and defaults.
    pydantic-settings reads from environment variables and .env automatically.
    Type coercion and validation run at startup — bad config fails fast with
    a clear message instead of a cryptic AttributeError or ValueError later.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,   # GROQ_API_KEY and groq_api_key both work
        extra="ignore",         # silently ignore unknown env vars
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    groq_api_key: str
    model_name: str   = "llama-3.3-70b-versatile"
    temperature: float = 0.7

    # ── Memory ────────────────────────────────────────────────────────────────
    max_token_limit: int = 2000

    # ── Embeddings ────────────────────────────────────────────────────────────
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── RAG ───────────────────────────────────────────────────────────────────
    rag_chunk_size: int    = 500
    rag_chunk_overlap: int = 50
    rag_top_k: int         = 4

    # ── Postmortem ────────────────────────────────────────────────────────────
    pm_chunk_lines: int   = 30
    pm_overlap_lines: int = 5
    pm_top_k: int         = 4

    # ── Auth / Session ────────────────────────────────────────────────────────
    secret_key: str                            = ""
    cookie_name: str                           = "opsiq_session"
    cookie_max_age: int                        = 7 * 24 * 60 * 60   # 7 days
    cookie_secure: bool                        = False
    cookie_samesite: Literal["lax", "strict", "none"] = "lax"
    bcrypt_rounds: int                         = 12

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = (
        "postgresql+asyncpg://myuser:mypassword@localhost:5432/opsiq"
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("groq_api_key")
    @classmethod
    def groq_api_key_must_be_set(cls, v: str) -> str:
        if not v or v == "your_groq_api_key_here":
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )
        return v

    @field_validator("secret_key")
    @classmethod
    def secret_key_must_be_secure(cls, v: str) -> str:
        if not v or v in _SECRET_KEY_PLACEHOLDERS:
            suggestion = secrets.token_hex(32)
            raise ValueError(
                "\n\n"
                "SECRET_KEY is not set or is using an insecure placeholder.\n"
                "All session tokens are signed with this key — a known or missing\n"
                "key allows anyone to forge valid authentication cookies.\n\n"
                "Add this to your .env file:\n\n"
                f"  SECRET_KEY={suggestion}\n\n"
                "Generate a new one anytime with:\n"
                "  python -c \"import secrets; print(secrets.token_hex(32))\"\n"
            )
        if len(v) < _MIN_SECRET_KEY_LENGTH:
            raise ValueError(
                f"SECRET_KEY is too short ({len(v)} chars). "
                f"Minimum is {_MIN_SECRET_KEY_LENGTH} characters.\n"
                "Generate a secure key with:\n"
                "  python -c \"import secrets; print(secrets.token_hex(32))\"\n"
            )
        return v

    @field_validator("temperature")
    @classmethod
    def temperature_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError(f"TEMPERATURE must be between 0.0 and 2.0, got {v}")
        return v

    @field_validator("bcrypt_rounds")
    @classmethod
    def bcrypt_rounds_sane(cls, v: int) -> int:
        if not 4 <= v <= 31:
            raise ValueError(f"BCRYPT_ROUNDS must be between 4 and 31, got {v}")
        return v

    @model_validator(mode="after")
    def rag_overlap_less_than_chunk(self) -> "Settings":
        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError(
                f"RAG_CHUNK_OVERLAP ({self.rag_chunk_overlap}) must be "
                f"less than RAG_CHUNK_SIZE ({self.rag_chunk_size})"
            )
        return self


# Single shared instance — imported by all other modules.
settings = Settings() # type: ignore

# ── Flat aliases for backwards compatibility ───────────────────────────────────
# Every other file does `from config import GROQ_API_KEY` etc.
# These aliases mean zero changes are needed anywhere else in the codebase.
GROQ_API_KEY       = settings.groq_api_key
MODEL_NAME         = settings.model_name
TEMPERATURE        = settings.temperature
MAX_TOKEN_LIMIT    = settings.max_token_limit
EMBED_MODEL        = settings.embed_model
RAG_CHUNK_SIZE     = settings.rag_chunk_size
RAG_CHUNK_OVERLAP  = settings.rag_chunk_overlap
RAG_TOP_K          = settings.rag_top_k
PM_CHUNK_LINES     = settings.pm_chunk_lines
PM_OVERLAP_LINES   = settings.pm_overlap_lines
PM_TOP_K           = settings.pm_top_k
SECRET_KEY         = settings.secret_key
COOKIE_NAME        = settings.cookie_name
COOKIE_MAX_AGE     = settings.cookie_max_age
COOKIE_SECURE      = settings.cookie_secure
COOKIE_SAMESITE    = settings.cookie_samesite
BCRYPT_ROUNDS      = settings.bcrypt_rounds
DATABASE_URL       = settings.database_url

# ── Constants (not env-driven) ────────────────────────────────────────────────
RAG_EXTENSIONS       = {".pdf", ".docx", ".doc", ".txt"}
POSTMORTEM_EXTENSION = ".log"