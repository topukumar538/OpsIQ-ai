# Location: backend/tests/test_ingest.py
"""
Tests for log file parsing — error extraction and line chunking.

Pure unit tests — no DB, no FAISS, no LLM needed.
Run with: pytest tests/test_ingest.py -v
"""
import os
os.environ["SECRET_KEY"]    = "a" * 32
os.environ["GROQ_API_KEY"]  = "test-key-not-real"
os.environ["DATABASE_URL"]  = "postgresql+asyncpg://u:p@localhost/test"

from postmortem.ingest import extract_errors, chunk_by_lines


# ── extract_errors ────────────────────────────────────────────────────────────

def test_extract_errors_empty_log():
    assert extract_errors("") == {}

def test_extract_errors_no_errors():
    log = "INFO starting service\nINFO connected to database\nINFO ready"
    assert extract_errors(log) == {}

def test_extract_errors_single_error():
    log = "ERROR something went wrong"
    result = extract_errors(log)
    assert len(result) == 1

def test_extract_errors_counts_multiple_occurrences():
    log = "ERROR NullPointerException\nERROR NullPointerException\nERROR NullPointerException"
    result = extract_errors(log)
    assert result.get("NullPointerException") == 3

def test_extract_errors_no_double_counting():
    # A line with both ERROR and EXCEPTION should be counted only once
    log = "ERROR Exception in thread main"
    result = extract_errors(log)
    total = sum(result.values())
    assert total == 1, f"Line counted {total} times, expected 1"

def test_extract_errors_named_exception_extracted():
    log = "2024-01-01 ERROR DatabaseException: connection refused"
    result = extract_errors(log)
    assert "DatabaseException" in result

def test_extract_errors_critical_pattern():
    log = "CRITICAL system failure detected"
    result = extract_errors(log)
    assert len(result) == 1

def test_extract_errors_case_insensitive_pattern():
    log = "error something failed"
    result = extract_errors(log)
    assert len(result) == 1

def test_extract_errors_multiple_error_types():
    log = (
        "ERROR NullPointerException\n"
        "ERROR NullPointerException\n"
        "CRITICAL DatabaseException\n"
        "INFO all good\n"
        "ERROR TimeoutException\n"
    )
    result = extract_errors(log)
    assert result.get("NullPointerException") == 2
    assert result.get("DatabaseException") == 1
    assert result.get("TimeoutException") == 1

def test_extract_errors_fatal_pattern():
    log = "FATAL out of memory"
    result = extract_errors(log)
    assert len(result) == 1

def test_extract_errors_traceback_pattern():
    log = "TRACEBACK (most recent call last):"
    result = extract_errors(log)
    assert len(result) == 1


# ── chunk_by_lines ────────────────────────────────────────────────────────────

def test_chunk_empty_log_returns_empty():
    assert chunk_by_lines("") == []

def test_chunk_single_line():
    chunks = chunk_by_lines("one line")
    assert len(chunks) == 1
    assert chunks[0].page_content == "one line"

def test_chunk_metadata_has_start_line():
    chunks = chunk_by_lines("line1\nline2")
    assert "start_line" in chunks[0].metadata
    assert chunks[0].metadata["start_line"] == 1

def test_chunk_metadata_has_chunk_index():
    chunks = chunk_by_lines("line1\nline2")
    assert "chunk_index" in chunks[0].metadata
    assert chunks[0].metadata["chunk_index"] == 0

def test_chunk_large_log_creates_multiple_chunks():
    # Create a log with 100 lines — should produce multiple chunks
    log = "\n".join([f"line {i}" for i in range(100)])
    chunks = chunk_by_lines(log)
    assert len(chunks) > 1

def test_chunk_overlap_means_lines_appear_in_multiple_chunks():
    # With chunk_lines=30, overlap=5, line 26-30 should appear in both chunk 0 and chunk 1
    log = "\n".join([f"line {i}" for i in range(60)])
    chunks = chunk_by_lines(log)
    assert len(chunks) >= 2
    # Last lines of chunk 0 should appear at start of chunk 1
    chunk0_lines = set(chunks[0].page_content.split("\n"))
    chunk1_lines = set(chunks[1].page_content.split("\n"))
    overlap = chunk0_lines & chunk1_lines
    assert len(overlap) > 0

def test_chunk_skips_empty_content():
    # Lines that produce empty content after strip should be skipped
    log = "\n\n\n"
    chunks = chunk_by_lines(log)
    assert len(chunks) == 0