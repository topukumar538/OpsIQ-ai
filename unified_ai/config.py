# Location: unified_ai/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ───────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
TEMPERATURE  = float(os.getenv("TEMPERATURE", "0.7"))

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")

# ── Memory ────────────────────────────────────────────────────────────────────
CHAT_TOKEN_LIMIT        = int(os.getenv("CHAT_TOKEN_LIMIT", "500"))
RAG_TOKEN_LIMIT         = int(os.getenv("RAG_TOKEN_LIMIT", "500"))
POSTMORTEM_TOKEN_LIMIT  = int(os.getenv("POSTMORTEM_TOKEN_LIMIT", "500"))

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ── RAG (pdf/docx/txt) ────────────────────────────────────────────────────────
RAG_CHUNK_SIZE    = int(os.getenv("RAG_CHUNK_SIZE", "500"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
RAG_TOP_K         = int(os.getenv("RAG_TOP_K", "4"))

# ── Postmortem (log files) ────────────────────────────────────────────────────
PM_CHUNK_LINES    = int(os.getenv("PM_CHUNK_LINES", "30"))
PM_OVERLAP_LINES  = int(os.getenv("PM_OVERLAP_LINES", "5"))
PM_TOP_K          = int(os.getenv("PM_TOP_K", "4"))

# ── File type rules ───────────────────────────────────────────────────────────
RAG_EXTENSIONS       = {".pdf", ".docx", ".doc", ".txt"}
POSTMORTEM_EXTENSION = ".log"