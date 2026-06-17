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
}
_MIN_SECRET_KEY_LENGTH = 32


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=_ENV_FILES or None, # type: ignore
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    groq_api_key    : str   = ""
    model_name      : str   = "llama-3.3-70b-versatile"
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
    max_upload_size_mb: int = 50

    # ── Postmortem ────────────────────────────────────────────────────────────
    pm_chunk_lines  : int = 30
    pm_overlap_lines: int = 5
    pm_top_k        : int = 4

    # ── Auth / Session ────────────────────────────────────────────────────────
    secret_key      : str                               = ""
    cookie_name     : str                               = "opsiq_session"
    cookie_max_age  : int                               = 7 * 24 * 60 * 60
    cookie_secure   : bool                              = False
    cookie_samesite : Literal["lax", "strict", "none"] = "lax"
    bcrypt_rounds   : int                               = 12
    session_ttl_seconds              : int = 2 * 60 * 60
    session_cleanup_interval_seconds : int = 15 * 60

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Comma-separated list of allowed origins.
    #
    # Why not "*":
    #   The CORS spec forbids "Access-Control-Allow-Origin: *" combined with
    #   "Access-Control-Allow-Credentials: true". Browsers reject the response
    #   entirely — cookies never get set and login silently fails.
    #   An explicit origin list is required whenever cookies are involved.
    #
    # Add your Railway URL in .env:
    #   ALLOWED_ORIGINS=https://opsiq.up.railway.app,https://yourfrontend.com
    allowed_origins: str = (
        "http://localhost:5173,"
        "http://localhost:8000,"
        "http://127.0.0.1:8000"
    )

    # ── FAISS persistence ─────────────────────────────────────────────────────
    faiss_store_dir: str = "/tmp/opsiq_stores"

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://myuser:mypassword@localhost:5432/opsiq"
    db_schema   : str = "opsiq"

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
        if not v or v in _SECRET_KEY_PLACEHOLDERS:
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

    @model_validator(mode="after")
    def rag_overlap_less_than_chunk(self) -> "Settings":
        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError(
                f"RAG_CHUNK_OVERLAP ({self.rag_chunk_overlap}) must be "
                f"less than RAG_CHUNK_SIZE ({self.rag_chunk_size})"
            )
        return self

    @property
    def allowed_origins_list(self) -> list[str]:
        """Parse comma-separated ALLOWED_ORIGINS string into a list."""
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


settings = Settings()

GROQ_API_KEY      = settings.groq_api_key
MODEL_NAME        = settings.model_name
CHAT_TEMPERATURE  = settings.chat_temperature
RAG_TEMPERATURE   = settings.rag_temperature
PM_TEMPERATURE    = settings.pm_temperature
MAX_TOKEN_LIMIT   = settings.max_token_limit
EMBED_MODEL       = settings.embed_model
RAG_CHUNK_SIZE    = settings.rag_chunk_size
RAG_CHUNK_OVERLAP = settings.rag_chunk_overlap
RAG_TOP_K         = settings.rag_top_k
PM_CHUNK_LINES    = settings.pm_chunk_lines
PM_OVERLAP_LINES  = settings.pm_overlap_lines
PM_TOP_K          = settings.pm_top_k
SECRET_KEY        = settings.secret_key
COOKIE_NAME       = settings.cookie_name
COOKIE_MAX_AGE    = settings.cookie_max_age
COOKIE_SECURE     = settings.cookie_secure
COOKIE_SAMESITE   = settings.cookie_samesite
BCRYPT_ROUNDS     = settings.bcrypt_rounds
DATABASE_URL                     = settings.database_url
DB_SCHEMA                        = settings.db_schema
FAISS_STORE_DIR                  = settings.faiss_store_dir
SESSION_TTL_SECONDS              = settings.session_ttl_seconds
SESSION_CLEANUP_INTERVAL_SECONDS = settings.session_cleanup_interval_seconds
MAX_UPLOAD_SIZE_MB               = settings.max_upload_size_mb
ALLOWED_ORIGINS                  = settings.allowed_origins_list   # ← new export

RAG_EXTENSIONS       = {".pdf", ".docx", ".doc", ".txt"}
POSTMORTEM_EXTENSION = ".log"

# ── .env.example addition ─────────────────────────────────────────────────────
# Add this line to your .env.example so teammates know to set it:
#
# ALLOWED_ORIGINS=http://localhost:5173,https://yourapp.up.railway.app