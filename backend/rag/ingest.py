# Location: backend/rag/ingest.py
import hashlib
import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth.models import SessionFile
from core.retriever import get_embeddings
from core.faiss_store import save_store
from config import RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP

logger = logging.getLogger(__name__)

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=RAG_CHUNK_SIZE,
    chunk_overlap=RAG_CHUNK_OVERLAP,
)

def hash_file(file_path: str, chunk_size: int = 65536) -> str:
    """Return SHA-256 hex digest of file contents for duplicate detection."""
    h = hashlib.sha256()
    try:
        with Path(file_path).open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")
    return h.hexdigest()


async def is_duplicate(db: AsyncSession, session_id: int, file_hash: str) -> bool:
    """
    Check if this exact file (by content hash) was already ingested
    into this session's store.

    Why DB not in-memory set: the in-memory set is lost on restart.
    DB check survives restarts so duplicate detection works even after
    the server comes back up.
    """
    result = await db.execute(
        select(SessionFile).where(
            SessionFile.session_id == session_id,
            SessionFile.file_hash  == file_hash,
        )
    )
    return result.scalar_one_or_none() is not None


async def record_file(
    db        : AsyncSession,
    session_id: int,
    filename  : str,
    file_hash : str,
) -> None:
    """Insert a file record into session_files after successful ingestion."""
    db.add(SessionFile(
        session_id=session_id,
        filename=filename,
        file_hash=file_hash,
    ))
    await db.commit()


def _load_documents(file_path: str):
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(file_path)
    elif suffix in {".docx", ".doc"}:
        loader = Docx2txtLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def build_rag_store(
    file_path : str,
    user_id   : int,
    token     : str,
) -> FAISS:
    """
    Load a document, chunk it, embed it, build a fresh FAISS store
    and persist to disk immediately.

    Note: duplicate check happens in the route BEFORE calling this —
    this function always builds unconditionally.
    """
    docs   = _load_documents(file_path)
    chunks = _splitter.split_documents(docs)
    store  = FAISS.from_documents(chunks, get_embeddings())
    save_store(store, user_id, token, "rag")
    logger.info(
        "Built RAG store: %d chunks from '%s' for user=%s token=%s",
        len(chunks), Path(file_path).name, user_id, token,
    )
    return store


def add_to_store(
    store     : FAISS,
    file_path : str,
    user_id   : int,
    token     : str,
) -> FAISS:
    """
    Merge a new document into an existing FAISS store and re-persist.

    Why merge not rebuild: rebuilding re-embeds all prior documents on every
    upload — slow and expensive for large stores. merge_from() adds only the
    new vectors. FAISS cannot delete old vectors so modified files will have
    both old and new chunks — the RAG system prompt instructs the LLM to
    prefer the most recent and consolidate duplicates.
    """
    docs      = _load_documents(file_path)
    chunks    = _splitter.split_documents(docs)
    new_store = FAISS.from_documents(chunks, get_embeddings())
    store.merge_from(new_store)
    save_store(store, user_id, token, "rag")
    logger.info(
        "Merged %d chunks from '%s' — store total: %d vectors",
        len(chunks), Path(file_path).name, store.index.ntotal,
    )
    return store


# Alias — some files import build_store, others build_rag_store.
# Both names point to the same function.
build_store = build_rag_store