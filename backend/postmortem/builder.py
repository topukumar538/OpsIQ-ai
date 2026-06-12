# Location: backend/postmortem/builder.py
from typing import Any
from langgraph.graph import StateGraph, END, START

from postmortem.state import PostmortemState
from postmortem.nodes.log_analyzer import log_analyzer
from postmortem.nodes.timeline import timeline
from postmortem.nodes.root_cause import root_cause
from postmortem.nodes.remediation import remediation
from postmortem.nodes.report_summarizer import report_summarizer


def build_postmortem_graph():
    graph = StateGraph(PostmortemState)

    graph.add_node("node_log",      log_analyzer)
    graph.add_node("node_timeline", timeline)
    graph.add_node("node_root_cause", root_cause)
    graph.add_node("node_remediation", remediation)
    graph.add_node("node_report_summarizer", report_summarizer)

    # log_analyzer and timeline run in parallel from START
    graph.add_edge(START, "node_log")
    graph.add_edge(START, "node_timeline")
    graph.add_edge("node_log",      "node_root_cause")
    graph.add_edge("node_timeline", "node_root_cause")
    graph.add_edge("node_root_cause",  "node_remediation")
    graph.add_edge("node_remediation", "node_report_summarizer")
    graph.add_edge("node_report_summarizer", END)

    return graph.compile()


def run_postmortem(llm, store, error_counts: dict, log_filename: str) -> dict[str, Any]:
    app = build_postmortem_graph()
    return app.invoke({
        "llm":              llm,
        "store":            store,
        "error_counts":     error_counts,
        "log_filename":     log_filename,
        "log_analysis":     "",
        "timeline_analysis": "",
        "root_cause":       "",
        "remediation":      "",
        "report_str":       "",
        "report_summary":   "",
    })