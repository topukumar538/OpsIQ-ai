# Location: backend/tests/test_router.py
"""
Tests for the file type classifier used in the upload route.

Pure unit tests — no DB, no network, no FastAPI app needed.
Run with: pytest tests/test_router.py -v
"""
import os
import tempfile
import pytest

os.environ["SECRET_KEY"]    = "a" * 32
os.environ["GROQ_API_KEY"]  = "test-key-not-real"
os.environ["DATABASE_URL"]  = "postgresql+asyncpg://u:p@localhost/test"

from router import classify_input, supported_extensions


# ── classify_input ────────────────────────────────────────────────────────────

def test_pdf_classified_as_rag():
    assert classify_input("/tmp/report.pdf") == "rag_file"

def test_docx_classified_as_rag():
    assert classify_input("/tmp/runbook.docx") == "rag_file"

def test_doc_classified_as_rag():
    assert classify_input("/tmp/runbook.doc") == "rag_file"

def test_txt_classified_as_rag():
    assert classify_input("/tmp/notes.txt") == "rag_file"

def test_log_classified_as_postmortem():
    assert classify_input("/tmp/incident.log") == "log_file"

def test_unsupported_extension_is_bad_path():
    assert classify_input("/tmp/data.csv") == "bad_path"

def test_exe_is_bad_path():
    assert classify_input("/tmp/virus.exe") == "bad_path"

def test_no_extension_is_bad_path():
    assert classify_input("/tmp/noextension") == "bad_path"

def test_uppercase_extension_classified_correctly():
    # Extensions should be case-insensitive
    assert classify_input("/tmp/report.PDF") == "rag_file"
    assert classify_input("/tmp/incident.LOG") == "log_file"

def test_mixed_case_extension():
    assert classify_input("/tmp/report.Pdf") == "rag_file"

def test_deep_path_classified_correctly():
    assert classify_input("/home/user/documents/reports/q3.pdf") == "rag_file"

def test_filename_with_dots_uses_last_extension():
    # file.backup.pdf should be classified as pdf not backup
    assert classify_input("/tmp/file.backup.pdf") == "rag_file"


# ── supported_extensions ──────────────────────────────────────────────────────

def test_supported_extensions_returns_set():
    assert isinstance(supported_extensions(), set)

def test_supported_extensions_includes_pdf():
    assert ".pdf" in supported_extensions()

def test_supported_extensions_includes_log():
    assert ".log" in supported_extensions()

def test_supported_extensions_includes_docx():
    assert ".docx" in supported_extensions()

def test_supported_extensions_includes_txt():
    assert ".txt" in supported_extensions()

def test_supported_extensions_does_not_include_csv():
    assert ".csv" not in supported_extensions()