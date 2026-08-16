# Location: backend/tests/test_rag_ingest.py
"""
Tests for RAG document ingestion — the guard against files that yield no
extractable text.

No DB, no network, no embeddings model: _load_and_chunk() raises before
get_embeddings() is ever called, so the failure paths cost nothing to test.
Run with: pytest tests/test_rag_ingest.py -v
"""
import os
import pytest

os.environ["SECRET_KEY"]    = "a" * 32
os.environ["GROQ_API_KEY"]  = "test-key-not-real"
os.environ["DATABASE_URL"]  = "postgresql+asyncpg://u:p@localhost/test"

from rag.ingest import _load_and_chunk, build_rag_store


# ── Files with no usable text ─────────────────────────────────────────────────

def test_empty_file_raises_value_error(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("")
    with pytest.raises(ValueError, match="No readable text"):
        _load_and_chunk(str(f))


def test_whitespace_only_file_raises_value_error(tmp_path):
    """
    A scanned PDF often yields "\\n\\n   " rather than "". That is technically
    non-empty, so without the strip() filter it survives the splitter and gets
    embedded as a meaningless vector — ingestion reports success while
    retrieval silently returns nothing.
    """
    f = tmp_path / "blank.txt"
    f.write_text("   \n\n\t  \n  ")
    with pytest.raises(ValueError, match="No readable text"):
        _load_and_chunk(str(f))


def test_error_message_names_the_offending_file(tmp_path):
    """The user needs to know which upload failed, not just that one did."""
    f = tmp_path / "scan.txt"
    f.write_text("  ")
    with pytest.raises(ValueError, match="scan"):
        _load_and_chunk(str(f))


# ── The working path still works ──────────────────────────────────────────────

def test_real_text_produces_chunks(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("The database connection pool was exhausted at 14:32.\n" * 20)

    chunks = _load_and_chunk(str(f))

    assert len(chunks) > 0
    assert all(c.page_content.strip() for c in chunks)


def test_short_file_still_produces_one_chunk(tmp_path):
    """A one-line document is valid input, not an empty one."""
    f = tmp_path / "short.txt"
    f.write_text("Postgres refused the connection.")

    assert len(_load_and_chunk(str(f))) == 1


# ── Regression ────────────────────────────────────────────────────────────────

def test_build_rag_store_rejects_empty_file_before_reaching_faiss(tmp_path):
    """
    Regression for "list index out of range".

    FAISS.from_documents() embeds the texts then reads embeddings[0] to get the
    vector dimension, so an empty chunk list raised IndexError from four frames
    deep — an error naming neither the file nor the cause. The guard must fire
    first, which also means this test needs no embeddings model.
    """
    f = tmp_path / "unreadable.txt"
    f.write_text("\n\n")

    with pytest.raises(ValueError):
        build_rag_store(str(f), user_id=1, token="test-token")