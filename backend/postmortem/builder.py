# Location: backend/postmortem/builder.py
from functools import lru_cache
from typing import Any

from langgraph.graph import StateGraph, END, START

from core.faiss_store import save_store
from core.retriever import get_embeddings
from postmortem.ingest import read_log, build_store
from postmortem.state import PostmortemState
from postmortem.nodes.log_analyzer import log_analyzer
from postmortem.nodes.timeline import timeline
from postmortem.nodes.root_cause import root_cause
from postmortem.nodes.remediation import remediation
from postmortem.nodes.report_summarizer import report_summarizer


@lru_cache(maxsize=1)
def build_postmortem_graph():
    """
    Compiled once and reused.

    The graph holds no session or log data — it is pure structure ("node A
    feeds node B"), identical on every run, so rebuilding and recompiling it
    per upload was wasted work. Log data is passed in at .invoke().

    START fans out to node_log and node_timeline, which run concurrently;
    both must finish before node_root_cause begins.
    """
    graph = StateGraph(PostmortemState)

    graph.add_node("node_log",               log_analyzer)
    graph.add_node("node_timeline",          timeline)
    graph.add_node("node_root_cause",        root_cause)
    graph.add_node("node_remediation",       remediation)
    graph.add_node("node_report_summarizer", report_summarizer)

    graph.add_edge(START, "node_log")
    graph.add_edge(START, "node_timeline")
    graph.add_edge("node_log",               "node_root_cause")
    graph.add_edge("node_timeline",          "node_root_cause")
    graph.add_edge("node_root_cause",        "node_remediation")
    graph.add_edge("node_remediation",       "node_report_summarizer")
    graph.add_edge("node_report_summarizer", END)

    return graph.compile()


def run_postmortem(
    log_path     : str,
    log_filename : str,
    llm,
    user_id      : int,
    session_token: str,
) -> dict[str, Any]:
    """
    Read a log file, build the FAISS store, run the analysis graph,
    persist the store to disk, and return results for the session state.
    """
    raw_log    = read_log(log_path)
    embeddings = get_embeddings()      # lru_cached — loaded once per process
    store, error_counts = build_store(raw_log, llm, embeddings)

    pm_state = build_postmortem_graph().invoke({
        "llm":               llm,
        "store":             store,
        "error_counts":      error_counts,
        "log_filename":      log_filename,
        "log_analysis":      "",
        "timeline_analysis": "",
        "root_cause":        "",
        "remediation":       "",
        "report_str":        "",
        "report_summary":    "",
    })

    # Saved only after the graph succeeds. If a node fails, /upload's DB writes
    # never run either — so a store on disk here would be restored into memory
    # but unreachable, since /chat only queries pm_store when mode is
    # "postmortem" and mode stays "chat" after a failure.
    save_store(store, user_id, session_token, "pm")

    return {
        "pm_store":       store,
        # .get() rather than [] — a missing key would raise after all five LLM
        # calls had already been paid for.
        "report_str":     pm_state.get("report_str", ""),
        "report_summary": pm_state.get("report_summary", ""),
        "error_counts":   error_counts,
    }