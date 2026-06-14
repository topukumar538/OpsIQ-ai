# Location: backend/graph/builder.py
from langgraph.graph import StateGraph, END

from graph.state import OpsState, POSTMORTEM, RAG
from graph.nodes.chat import chat_node
from graph.nodes.rag import rag_node
from graph.nodes.postmortem import postmortem_node
from core.memory import make_memory


def _route_from_start(state: OpsState) -> str:
    """Entry routing — dispatch to whichever mode is active."""
    mode = state.get("mode", "chat")
    if mode == RAG:        return "rag"
    if mode == POSTMORTEM: return "postmortem"
    return "chat"


def _route_from_chat(state: OpsState) -> str:
    """After chat node — check if a file was uploaded and needs routing."""
    if state.get("file_path"):
        suffix = state["file_path"].rsplit(".", 1)[-1].lower()
        if suffix == "log": return "postmortem"
        return "rag"
    return END


def _route_from_rag(state: OpsState) -> str:
    """After rag node — a log file in RAG mode produces a warning, not a transition."""
    if state.get("file_path"):
        suffix = state["file_path"].rsplit(".", 1)[-1].lower()
        if suffix == "log": return "postmortem"
    return END


def build_graph() -> StateGraph:
    graph = StateGraph(OpsState)

    graph.add_node("chat",       chat_node)
    graph.add_node("rag",        rag_node)
    graph.add_node("postmortem", postmortem_node)

    graph.set_conditional_entry_point(_route_from_start)

    graph.add_conditional_edges("chat",       _route_from_chat, {"rag": "rag", "postmortem": "postmortem", END: END})
    graph.add_conditional_edges("rag",        _route_from_rag,  {"postmortem": "postmortem", END: END})
    graph.add_edge("postmortem", END)

    return graph.compile()


def make_initial_state(llm) -> OpsState:
    """
    Create a blank OpsState for a new session.
    Memory objects are created lazily (on first use) except chat_memory
    which is seeded immediately since chat is the default mode.
    """
    return OpsState(
        mode          = "chat",
        user_input    = "",
        file_path     = "",
        response      = "",
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