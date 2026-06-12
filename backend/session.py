# Location: backend/session.py
from typing import Any
from core.llm import get_llm
from graph.builder import build_graph, make_initial_state

# session_id -> {"graph": compiled_graph, "state": current_state}
_sessions: dict[str, dict[str, Any]] = {}


def create_session(session_id: str) -> None:
    llm = get_llm()
    _sessions[session_id] = {
        "graph": build_graph(),
        "state": make_initial_state(llm),
        "llm":   llm,
    }


def get_session(session_id: str) -> dict[str, Any]:
    if session_id not in _sessions:
        create_session(session_id)
    return _sessions[session_id]


def delete_session(session_id: str) -> None:
    _sessions.pop(session_id, None)


def run_graph(session_id: str, user_input: str = "", file_path: str = "") -> dict[str, Any]:
    session = get_session(session_id)
    state   = session["state"]

    # Update input
    state["user_input"] = user_input
    state["file_path"]  = file_path

    # Run graph
    result = session["graph"].invoke(state)

    # Persist updated state
    session["state"] = result
    return result