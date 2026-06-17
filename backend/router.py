# Location: backend/router.py
"""
File type classifier for the /upload route.
"""
from pathlib import Path

from config import RAG_EXTENSIONS, POSTMORTEM_EXTENSION


def classify_input(file_path: str) -> str:
    """
    Classify an uploaded file by its extension.

    Returns one of:
        "rag_file"   — PDF, DOCX, DOC, TXT → goes to RAG ingestion
        "log_file"   — LOG → goes to postmortem pipeline
        "bad_path"   — unsupported extension → rejected with error message
    """
    suffix = Path(file_path).suffix.lower()

    if suffix in RAG_EXTENSIONS:
        return "rag_file"

    if suffix == POSTMORTEM_EXTENSION:
        return "log_file"

    return "bad_path"


def supported_extensions() -> set[str]:
    """Return all extensions the app accepts — used for client-side validation hint."""
    return RAG_EXTENSIONS | {POSTMORTEM_EXTENSION}