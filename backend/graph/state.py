# Location: backend/graph/state.py
from typing import Optional, Any
from typing_extensions import TypedDict

RAG        = "rag"
POSTMORTEM = "postmortem"


class OpsState(TypedDict, total=False):
    # ── Routing ───────────────────────────────────────────────────────────────
    mode          : str           # "chat" | "rag" | "postmortem"
    is_locked     : bool          # True after postmortem report generated

    # ── Per-turn inputs ───────────────────────────────────────────────────────
    user_input    : str
    file_path     : str           # temp path set by upload route, cleared after use
    response      : str

    # ── Session identity ──────────────────────────────────────────────────────
    # Passed through state so nodes can construct FAISS paths without
    # needing a DB connection directly inside the graph.
    user_id       : int
    session_token : str

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm           : Any           # ChatGroq instance (chat temperature)

    # ── Memory ────────────────────────────────────────────────────────────────
    chat_memory   : Optional[Any]
    rag_memory    : Optional[Any]
    pm_memory     : Optional[Any]

    # ── FAISS stores ──────────────────────────────────────────────────────────
    rag_store     : Optional[Any]
    pm_store      : Optional[Any]

    # ── Postmortem ────────────────────────────────────────────────────────────
    report_str    : str
    rag_warning   : str           # set when log uploaded in RAG mode