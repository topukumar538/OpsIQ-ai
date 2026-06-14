# Location: backend/core/faiss_store.py
"""
FAISS persistence helpers.

Store layout on disk:
  {FAISS_STORE_DIR}/
    {user_id}/
      {session_token}/
        rag/          ← RAG document store
          index.faiss
          index.pkl
        pm/           ← Postmortem log store
          index.faiss
          index.pkl

Why user_id in path:
    Previously the path was just /{session_token}/.
    If two users somehow had the same token, their stores would collide.
    Including user_id makes paths globally unique and adds a second layer
    of isolation — even a path traversal bug can't cross user boundaries.
"""
import logging
import shutil
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS

from core.retriever import get_embeddings
from config import FAISS_STORE_DIR

logger = logging.getLogger(__name__)


def _store_path(user_id: int, token: str, kind: str) -> Path:
    """Return the directory path for a store. kind is 'rag' or 'pm'."""
    return Path(FAISS_STORE_DIR) / str(user_id) / token / kind


def save_store(store: FAISS, user_id: int, token: str, kind: str) -> None:
    """
    Persist a FAISS store to disk.

    Called immediately after building or updating a store so the on-disk
    copy is always in sync with the in-memory copy.
    Errors are logged but not raised — a save failure should never crash
    the user's upload flow. The store still works in-memory.
    """
    path = _store_path(user_id, token, kind)
    path.mkdir(parents=True, exist_ok=True)
    try:
        store.save_local(str(path))
        logger.debug("Saved %s store → %s", kind, path)
    except Exception:
        logger.exception(
            "Failed to save %s store for user=%s token=%s — "
            "data will not survive a restart", kind, user_id, token,
        )


def load_store(user_id: int, token: str, kind: str) -> Optional[FAISS]:
    """
    Load a FAISS store from disk if one exists.

    Returns None if no saved store is found — new session or first upload.
    Called during session restore so users don't lose document context
    after a server restart.
    """
    path = _store_path(user_id, token, kind)
    index_file = path / "index.faiss"

    if not index_file.exists():
        return None

    try:
        store = FAISS.load_local(
            str(path),
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
        logger.info("Restored %s store ← %s", kind, path)
        return store
    except Exception:
        logger.warning(
            "Could not load %s store for user=%s token=%s — starting fresh",
            kind, user_id, token, exc_info=True,
        )
        return None


def delete_store(user_id: int, token: str) -> None:
    """
    Remove all on-disk stores for a session.

    Called when a session is explicitly deleted by the user.
    TTL-based memory eviction does NOT call this — stores stay on disk
    so a reconnecting user gets their context back automatically.
    """
    session_dir = Path(FAISS_STORE_DIR) / str(user_id) / token
    if session_dir.exists():
        try:
            shutil.rmtree(session_dir)
            logger.debug("Deleted stores for user=%s token=%s", user_id, token)
        except Exception:
            logger.warning(
                "Failed to delete stores for user=%s token=%s",
                user_id, token, exc_info=True,
            )