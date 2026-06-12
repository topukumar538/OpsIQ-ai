# Location: backend/graph/edges.py
from router import classify_input
from graph.state import CHAT, RAG, POSTMORTEM


def route_chat(state: dict) -> str:
    # Routes after chat_node — only file input causes transition
    kind = classify_input(state.get("file_path") or "")
    if kind == "log_file":  return POSTMORTEM
    if kind == "rag_file":  return RAG
    return END_CHAT  # normal message — end graph, response already set


def route_rag(state: dict) -> str:
    # Routes after rag_node
    kind = classify_input(state.get("file_path") or "")
    if kind == "log_file":  return POSTMORTEM
    return END_RAG  # normal message or rag file — end graph


END_CHAT = "__end_chat__"
END_RAG  = "__end_rag__"