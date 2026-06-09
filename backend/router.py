# Location: backend/router.py
import re
from pathlib import Path

from config import RAG_EXTENSIONS, POSTMORTEM_EXTENSION


def normalize_path(text: str) -> Path:
    # Strip quotes and convert Windows path to WSL /mnt/c/...
    p = text.strip().strip('"').strip("'").replace("\\", "/")
    win = re.match(r"^([A-Za-z]):/(.*)$", p)
    if win:
        p = f"/mnt/{win.group(1).lower()}/{win.group(2)}"
    return Path(p).expanduser().resolve()


def looks_like_path(text: str) -> bool:
    # Detect if input looks like a file path
    p = text.strip().strip('"').strip("'")
    return bool(p) and (
        "\\" in p or "/" in p or
        re.match(r"^[A-Za-z]:", p) is not None or
        "." in Path(p).suffix
    )


def is_rag_file(text: str) -> bool:
    path = normalize_path(text)
    return path.exists() and path.is_file() and path.suffix.lower() in RAG_EXTENSIONS


def is_log_file(text: str) -> bool:
    path = normalize_path(text)
    return path.exists() and path.is_file() and path.suffix.lower() == POSTMORTEM_EXTENSION


def classify_input(text: str) -> str:
    """
    Returns one of:
      'rag_file'   — valid pdf/docx/txt file
      'log_file'   — valid .log file
      'bad_path'   — looks like a path but invalid/unsupported
      'message'    — normal chat message
    """
    if is_rag_file(text):
        return "rag_file"
    if is_log_file(text):
        return "log_file"
    if looks_like_path(text):
        return "bad_path"
    return "message"