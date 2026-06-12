# Location: backend/graph/builder.py
from typing import Any
from langgraph.graph import StateGraph, END, START

from graph.state import OpsState, CHAT, RAG, POSTMORTEM
from graph.nodes.chat import chat_node
from graph.nodes.rag import rag_node
from graph.nodes.postmortem import postmortem_node


def _route_from_chat(state: dict) -> str:
    # After chat_node runs — if file_path is set transition, else end
    fp = state.get("file_path") or ""
    if not fp:
        return END
    from router import classify_input
    kind = classify_input(fp)
    if kind == "log_file":  return POSTMORTEM
    if kind == "rag_file":  return RAG
    return END


def _route_from_rag(state: dict) -> str:
    # After rag_node runs — log file transitions to postmortem, else end
    fp = state.get("file_path") or ""
    if not fp:
        return END
    from router import classify_input
    kind = classify_input(fp)
    if kind == "log_file":  return POSTMORTEM
    return END


def build_graph():
    graph = StateGraph(OpsState)

    graph.add_node(CHAT,       chat_node)
    graph.add_node(RAG,        rag_node)
    graph.add_node(POSTMORTEM, postmortem_node)

    # Entry — go directly to current mode node
    graph.add_conditional_edges(START, lambda s: s["mode"], {
        CHAT:       CHAT,
        RAG:        RAG,
        POSTMORTEM: POSTMORTEM,
    })

    # From chat — end or transition
    graph.add_conditional_edges(CHAT, _route_from_chat, {
        END:        END,
        RAG:        RAG,
        POSTMORTEM: POSTMORTEM,
    })

    # From rag — end or transition to postmortem
    graph.add_conditional_edges(RAG, _route_from_rag, {
        END:        END,
        POSTMORTEM: POSTMORTEM,
    })

    # Postmortem always ends
    graph.add_edge(POSTMORTEM, END)

    return graph.compile()


def make_initial_state(llm) -> dict[str, Any]:
    from core.memory import build_memory
    return {
        "mode":        CHAT,
        "user_input":  "",
        "file_path":   "",
        "response":    "",
        "chat_memory": build_memory(llm),
        "rag_store":   None,
        "rag_memory":  None,
        "pm_store":    None,
        "pm_memory":   None,
        "report_str":  "",
        "llm":         llm,
        "rag_warning":  "",
    }