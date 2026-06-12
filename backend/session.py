# Location: backend/session.py
import asyncio
import time
from typing import Any
from core.llm import get_llm
from graph.builder import build_graph, make_initial_state

# session_id -> {"graph", "state", "llm", "lock", "last_accessed"}
_sessions: dict[str, dict[str, Any]] = {}


def create_session(session_id: str) -> None:
    llm = get_llm()
    _sessions[session_id] = {
        "graph":         build_graph(),
        "state":         make_initial_state(llm),
        "llm":           llm,
        # Each session gets its own asyncio.Lock.
        # This ensures only one request at a time can read or write this
        # session's state, preventing race conditions when the frontend
        # fires concurrent requests (e.g. chat + memory poll simultaneously).
        "lock":          asyncio.Lock(),
        # Tracks when this session was last used — needed for TTL cleanup (issue #4).
        "last_accessed": time.time(),
    }


def get_session(session_id: str) -> dict[str, Any]:
    if session_id not in _sessions:
        create_session(session_id)
    session = _sessions[session_id]
    session["last_accessed"] = time.time()
    return session


def delete_session(session_id: str) -> None:
    _sessions.pop(session_id, None)


def run_graph(session_id: str, user_input: str = "", file_path: str = "") -> dict[str, Any]:
    """Synchronous graph runner — used by CLI and internally by run_graph_async."""
    session = get_session(session_id)
    state   = session["state"]

    state["user_input"] = user_input
    state["file_path"]  = file_path

    result = session["graph"].invoke(state)

    session["state"] = result
    return result


async def run_graph_async(
    session_id: str,
    user_input: str = "",
    file_path: str = "",
) -> dict[str, Any]:
    """
    Async wrapper for run_graph.

    Why: run_graph() triggers the full LangGraph postmortem pipeline which
    includes FAISS embedding and multiple blocking LLM calls — potentially
    30-90 seconds of CPU/IO work. Calling it directly inside an async FastAPI
    route would freeze the event loop for ALL users during that time.

    run_in_executor(None, ...) hands the synchronous function off to the
    default ThreadPoolExecutor so the event loop stays free to handle other
    requests. The awaiting route still waits for the result, but nothing
    else is blocked.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,           # None = use the default ThreadPoolExecutor
        run_graph,      # the synchronous function
        session_id,
        user_input,
        file_path,
    )