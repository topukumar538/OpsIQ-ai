# Location: backend/postmortem/builder.py
from typing import Any

from langgraph.graph import StateGraph, END, START

from core.faiss_store import save_store
from postmortem.ingest import read_log, build_store
from postmortem.state import PostmortemState
from postmortem.nodes.log_analyzer import log_analyzer
from postmortem.nodes.timeline import timeline
from postmortem.nodes.root_cause import root_cause
from postmortem.nodes.remediation import remediation
from postmortem.nodes.report_summarizer import report_summarizer


def build_postmortem_graph():
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
    raw_log = read_log(log_path)
    store, error_counts = build_store(raw_log, llm)

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

    save_store(store, user_id, session_token, "pm")

    return {
        "pm_store":       store,
        "report_str":     pm_state["report_str"],
        "report_summary": pm_state.get("report_summary", ""),
        "error_counts":   error_counts,
    }
