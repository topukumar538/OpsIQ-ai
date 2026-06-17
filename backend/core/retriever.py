# Location: backend/core/retriever.py
"""
FAISS retrieval helpers — shared across RAG and postmortem nodes.

The embeddings model is lazily loaded on first call so a failed model
download doesn't crash the entire app at import time (fixes issue #17).

"""
import logging
from functools import lru_cache
from typing import Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import EMBED_MODEL

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Lazily load the embeddings model — cached after first call.

    Why lazy: loading at module import time means a network error or
    disk-space issue during model download crashes the whole app with
    a cryptic error. Lazy loading gives a clear error message at the
    point of first use, and the app can still start and serve requests
    that don't need embeddings (auth, session management, chat).
    """
    try:
        logger.info("Loading embeddings model: %s", EMBED_MODEL)
        return HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load embeddings model '{EMBED_MODEL}'. "
            "Check your internet connection and available disk space.\n"
            f"Original error: {e}"
        ) from e


def retrieve(store: Optional[FAISS], query: str, k: int) -> str:
    """
    Retrieve top-k relevant chunks from a FAISS store.
    Returns empty string if store is None (graceful fallback).
    """
    if store is None:
        return ""
    docs = store.similarity_search(query, k=k)
    return "\n\n".join(doc.page_content for doc in docs)