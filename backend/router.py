# Location: backend/router.py
"""
File type classifier for the /upload route and CLI file paths.

IMPORTANT — scope of this module:
    classify_input() is for real file paths (upload temp files or CLI paths).
    classify_cli_input() additionally handles plain chat messages for the CLI.
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


def classify_cli_input(user_input: str) -> str:
    """
    Classify CLI input as a chat message or a file path.

    Returns one of:
        "message"    — plain text chat
        "rag_file"   — existing path to a RAG document
        "log_file"   — existing path to a log file
        "bad_path"   — path exists but unsupported extension
    """
    path = Path(user_input)
    if path.exists() and path.is_file():
        return classify_input(str(path))
    return "message"


def supported_extensions() -> set[str]:
    """Return all extensions the app accepts — used for client-side validation hint."""
    return RAG_EXTENSIONS | {POSTMORTEM_EXTENSION}
