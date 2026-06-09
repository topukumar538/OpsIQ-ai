# Location: backend/session.py
from dataclasses import dataclass, field
from typing import Any

CHAT       = "chat"
RAG        = "rag"
POSTMORTEM = "postmortem"


@dataclass
class SessionState:
    mode:        str = CHAT
    chat_memory: Any = None
    rag_store:   Any = None
    rag_memory:  Any = None
    pm_store:    Any = None
    pm_memory:   Any = None
    report_str:  str = ""


# In-memory store — session_id -> SessionState
_sessions: dict[str, SessionState] = {}


def get_session(session_id: str) -> SessionState:
    if session_id not in _sessions:
        _sessions[session_id] = SessionState()
    return _sessions[session_id]


def delete_session(session_id: str) -> None:
    _sessions.pop(session_id, None)