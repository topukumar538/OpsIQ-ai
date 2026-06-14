# Location: backend/router.py
"""
File type classifier for the /upload route.

IMPORTANT — scope of this module:
    This classifier is ONLY called from /upload after a file has already
    been written to a temp path. It should NEVER be called on raw user
    chat messages.

    The original implementation used Path(input).exists() + looks_like_path()
    heuristics to distinguish "is this a file path or a message?". That caused
    false positives when users typed things like:
        "Check config.log for errors"   → misclassified as log_file
        "See https://docs.co/guide.pdf" → misclassified as rag_file
        "Fixed in version 3.0.1"        → triggered looks_like_path()

    In the web app, messages go through /chat and files go through /upload —
    they are already separated at the route level. The classifier only needs
    to answer one question: given a real file path, what kind of file is it?
    Path existence checks and message-vs-path heuristics are not needed.
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

    Args:
        file_path: absolute path to a temp file that has already been
                   written to disk by the /upload route. This must always
                   be a real file path — never a raw user message string.
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