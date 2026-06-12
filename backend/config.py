# Location: backend/config.py
import os
from dotenv import load_dotenv
from typing import Literal

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
TEMPERATURE  = float(os.getenv("TEMPERATURE", "0.7"))

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")

# ── Memory ────────────────────────────────────────────────────────────────────
MAX_TOKEN_LIMIT = int(os.getenv("MAX_TOKEN_LIMIT", "2000"))

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ── RAG ───────────────────────────────────────────────────────────────────────
RAG_CHUNK_SIZE    = int(os.getenv("RAG_CHUNK_SIZE", "500"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
RAG_TOP_K         = int(os.getenv("RAG_TOP_K", "4"))

# ── Postmortem ────────────────────────────────────────────────────────────────
PM_CHUNK_LINES   = int(os.getenv("PM_CHUNK_LINES", "30"))
PM_OVERLAP_LINES = int(os.getenv("PM_OVERLAP_LINES", "5"))
PM_TOP_K         = int(os.getenv("PM_TOP_K", "4"))

# ── File types ────────────────────────────────────────────────────────────────
RAG_EXTENSIONS       = {".pdf", ".docx", ".doc", ".txt"}
POSTMORTEM_EXTENSION = ".log"

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://myuser:mypassword@localhost:5432/opsiq"
)

# ── Auth / Session ────────────────────────────────────────────────────────────
SECRET_KEY         = os.getenv("SECRET_KEY", "change-me-in-production-use-a-long-random-string")
COOKIE_NAME        = os.getenv("COOKIE_NAME", "opsiq_session")
COOKIE_MAX_AGE     = int(os.getenv("COOKIE_MAX_AGE", str(7 * 24 * 60 * 60)))   # 7 days in seconds
COOKIE_SECURE      = os.getenv("COOKIE_SECURE", "false").lower() == "true"      # True in production (HTTPS)
COOKIE_SAMESITE: Literal["lax", "strict", "none"] = "lax"# lax | strict | none
BCRYPT_ROUNDS      = int(os.getenv("BCRYPT_ROUNDS", "12"))