# Location: backend/graph/state.py
from typing import Optional, Any
from typing_extensions import TypedDict
from core.memory import make_memory

RAG        = "rag"
POSTMORTEM = "postmortem"


class OpsState(TypedDict, total=False):
    # ── Routing ───────────────────────────────────────────────────────────────
    mode      : str   # "chat" | "rag" | "postmortem"
    is_locked : bool  # True after postmortem report generated

    # ── Per-turn inputs ───────────────────────────────────────────────────────
    user_input : str
    file_path  : str  # temp path set by upload route, cleared after use
    response   : str

    # ── Session identity ──────────────────────────────────────────────────────
    user_id       : int
    session_token : str

    # ── LLM ───────────────────────────────────────────────────────────────────
    # Three cached instances at different temperatures — created once per
    # session and reused. Previously postmortem_node called get_pm_llm() on
    # every message, constructing a new ChatGroq client each time.
    llm    : Any   # chat  — temperature 0.7
    rag_llm: Any   # rag   — temperature 0.3
    pm_llm : Any   # postmortem — temperature 0.1

    # ── Memory ────────────────────────────────────────────────────────────────
    chat_memory: Optional[Any]
    rag_memory : Optional[Any]
    pm_memory  : Optional[Any]

    # ── FAISS stores ──────────────────────────────────────────────────────────
    rag_store: Optional[Any]
    pm_store : Optional[Any]

    # ── Postmortem ────────────────────────────────────────────────────────────
    report_str : str
    rag_warning: str  # set when log uploaded in RAG mode

def make_initial_state(llm) -> OpsState:
    """
    Blank OpsState for a new session. Memory objects are created lazily
    except chat_memory, since chat is the default mode.

    Moved here from graph/builder.py, which no longer exists — the top-level
    graph it built only ever dispatched to a single node.
    """
    return OpsState(
        mode          = "chat",
        file_path     = "",
        chat_memory   = make_memory(llm),
        rag_memory    = None,
        pm_memory     = None,
        rag_store     = None,
        pm_store      = None,
        report_str    = "",
        rag_warning   = "",
        is_locked     = False,
        llm           = llm,
        user_id       = 0,
        session_token = "",
    )