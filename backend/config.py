# Location: backend/config.py
import secrets
from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _BACKEND_DIR.parent
_ENV_FILES    = [
    p for p in (_PROJECT_ROOT / ".env", _BACKEND_DIR / ".env") if p.is_file()
]

_SECRET_KEY_PLACEHOLDERS = {
    "",
    "change-me-in-production-use-a-long-random-string",
    "change-me",
    "secret",
    "secret_key",
    "your_secret_key_here",
    'generate_with_python_-c_"import_secrets;print(secrets.token_hex(32))"',
}
_MIN_SECRET_KEY_LENGTH = 32


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=_ENV_FILES or None,  # type: ignore
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # ── LLM ───────────────────────────────────────────────────────────────────
    # Model availability changes: llama-3.3-70b-versatile moved behind an
    # enterprise tier and started returning model_not_found on standard keys.
    # Check https://console.groq.com/docs/models if requests start failing.
    groq_api_key    : str   = ""
    model_name      : str   = "openai/gpt-oss-120b"
    chat_temperature: float = 0.7
    rag_temperature : float = 0.3
    pm_temperature  : float = 0.1

    # ── Memory ────────────────────────────────────────────────────────────────
    max_token_limit: int = 2000

    # ── Embeddings ────────────────────────────────────────────────────────────
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── RAG ───────────────────────────────────────────────────────────────────
    rag_chunk_size   : int = 500
    rag_chunk_overlap: int = 50
    rag_top_k        : int = 4

    # ── Upload ────────────────────────────────────────────────────────────────
    # Sized by LLM processing time rather than storage. A larger log produces
    # thousands of chunks to embed and exhausts the Groq quota before the
    # analysis finishes — the upload succeeds, then the pipeline fails with a
    # rate-limit message.
    max_upload_size_mb: int = 10

    # ── Postmortem ────────────────────────────────────────────────────────────
    pm_chunk_lines  : int = 30
    pm_overlap_lines: int = 5
    pm_top_k        : int = 4
    # Ceiling on the pipeline. It holds the session lock across six LLM calls,
    # and cleanup_expired_sessions won't evict a locked session — so without a
    # timeout a hung request blocks that session and leaks it from the cache
    # for the lifetime of the process.
    postmortem_timeout_seconds: int = 300

    # ── Auth / Session ────────────────────────────────────────────────────────
    secret_key      : str                              = ""
    cookie_name     : str                              = "opsiq_session"
    cookie_max_age  : int                              = 7 * 24 * 60 * 60
    cookie_secure   : bool                             = True
    cookie_samesite : Literal["lax", "strict", "none"] = "lax"
    bcrypt_rounds   : int                              = 12
    session_ttl_seconds              : int = 2 * 60 * 60
    session_cleanup_interval_seconds : int = 15 * 60

    allowed_origins: str = (
        "http://localhost:5173,"
        "http://localhost:8000,"
        "http://127.0.0.1:8000"
    )

    # ── FAISS persistence ─────────────────────────────────────────────────────
    faiss_store_dir: str = "/tmp/opsiq_stores"

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str  = "postgresql+asyncpg://opsiq:opsiq_dev_password@localhost:5432/opsiq"
    db_schema   : str  = "opsiq"
    # Set explicitly, never inferred from the hostname. An earlier version
    # enabled SSL whenever the host wasn't localhost, assuming anything else
    # must be a cloud provider — that broke the moment Postgres moved into a
    # container, where the host is "db" but SSL is off and asyncpg fails with
    # "rejected SSL upgrade" rather than falling back.
    db_ssl      : bool = False

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("groq_api_key")
    @classmethod
    def groq_api_key_must_be_set(cls, v: str) -> str:
        if not v or v == "your_groq_api_key_here":
            raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")
        return v

    @field_validator("secret_key")
    @classmethod
    def secret_key_must_be_secure(cls, v: str) -> str:
        if not v or v.strip().lower() in _SECRET_KEY_PLACEHOLDERS:
            suggestion = secrets.token_hex(32)
            raise ValueError(
                "\n\nSECRET_KEY is not set or is using an insecure placeholder.\n"
                f"Add this to your .env file:\n\n  SECRET_KEY={suggestion}\n"
            )
        if len(v) < _MIN_SECRET_KEY_LENGTH:
            raise ValueError(
                f"SECRET_KEY is too short ({len(v)} chars). "
                f"Minimum is {_MIN_SECRET_KEY_LENGTH} characters."
            )
        # A real key is random hex — no spaces, quotes, brackets, or English
        # words. Length alone lets long instruction strings slip through, e.g.
        # 'generate_with_python_-c_"import secrets; print(...)"' is 58 chars.
        if any(c in v for c in " \"'()<>") or "generate" in v.lower():
            raise ValueError(
                "\n\nSECRET_KEY looks like placeholder text, not a real key.\n"
                f"Generate one:\n\n  SECRET_KEY={secrets.token_hex(32)}\n"
            )
        return v

    @field_validator("chat_temperature", "rag_temperature", "pm_temperature")
    @classmethod
    def temperature_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {v}")
        return v

    @field_validator("bcrypt_rounds")
    @classmethod
    def bcrypt_rounds_sane(cls, v: int) -> int:
        if not 4 <= v <= 31:
            raise ValueError(f"BCRYPT_ROUNDS must be between 4 and 31, got {v}")
        return v

    @field_validator("postmortem_timeout_seconds")
    @classmethod
    def postmortem_timeout_sane(cls, v: int) -> int:
        # Below ~30s the pipeline can't finish even on a small log, so a low
        # value would fail every upload rather than catching hung requests.
        if v < 30:
            raise ValueError(
                f"POSTMORTEM_TIMEOUT_SECONDS must be at least 30, got {v}"
            )
        return v

    @model_validator(mode="after")
    def chunk_overlaps_less_than_chunk_size(self) -> "Settings":
        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError(
                f"RAG_CHUNK_OVERLAP ({self.rag_chunk_overlap}) must be "
                f"less than RAG_CHUNK_SIZE ({self.rag_chunk_size})"
            )
        # Same rule for the log chunker: equal values make it advance zero
        # lines per step, so chunking never terminates.
        if self.pm_overlap_lines >= self.pm_chunk_lines:
            raise ValueError(
                f"PM_OVERLAP_LINES ({self.pm_overlap_lines}) must be "
                f"less than PM_CHUNK_LINES ({self.pm_chunk_lines})"
            )
        return self

    @property
    def allowed_origins_list(self) -> list[str]:
        """Parse the comma-separated ALLOWED_ORIGINS string into a list."""
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


settings = Settings()

# ── Exports ───────────────────────────────────────────────────────────────────
# Module-level constants so callers can `from config import X` rather than
# threading the settings object through every function.

# LLM
GROQ_API_KEY     = settings.groq_api_key
MODEL_NAME       = settings.model_name
CHAT_TEMPERATURE = settings.chat_temperature
RAG_TEMPERATURE  = settings.rag_temperature
PM_TEMPERATURE   = settings.pm_temperature
MAX_TOKEN_LIMIT  = settings.max_token_limit
EMBED_MODEL      = settings.embed_model

# RAG
RAG_CHUNK_SIZE    = settings.rag_chunk_size
RAG_CHUNK_OVERLAP = settings.rag_chunk_overlap
RAG_TOP_K         = settings.rag_top_k

# Postmortem
PM_CHUNK_LINES             = settings.pm_chunk_lines
PM_OVERLAP_LINES           = settings.pm_overlap_lines
PM_TOP_K                   = settings.pm_top_k
POSTMORTEM_TIMEOUT_SECONDS = settings.postmortem_timeout_seconds

# Auth / session
SECRET_KEY      = settings.secret_key
COOKIE_NAME     = settings.cookie_name
COOKIE_MAX_AGE  = settings.cookie_max_age
COOKIE_SECURE   = settings.cookie_secure
COOKIE_SAMESITE = settings.cookie_samesite
BCRYPT_ROUNDS   = settings.bcrypt_rounds
SESSION_TTL_SECONDS              = settings.session_ttl_seconds
SESSION_CLEANUP_INTERVAL_SECONDS = settings.session_cleanup_interval_seconds

# Database
DATABASE_URL = settings.database_url
DB_SCHEMA    = settings.db_schema
DB_SSL       = settings.db_ssl

# Storage / upload
FAISS_STORE_DIR    = settings.faiss_store_dir
MAX_UPLOAD_SIZE_MB = settings.max_upload_size_mb

# Misc
ALLOWED_ORIGINS = settings.allowed_origins_list

# File type routing
RAG_EXTENSIONS       = {".pdf", ".docx", ".doc", ".txt"}
POSTMORTEM_EXTENSION = ".log"