# Location: backend/postmortem/graph.py
from typing import TypedDict, Any

from langgraph.graph import StateGraph, END, START
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from postmortem.nodes import node_log, node_timeline, node_root_cause, node_remediation


class PostmortemState(TypedDict):
    llm: ChatGroq
    store: FAISS
    error_counts: dict
    log_analysis: str
    timeline_analysis: str
    root_cause: str
    remediation: str


def build_graph():
    graph = StateGraph(PostmortemState)

    graph.add_node("node_log",         node_log)
    graph.add_node("node_timeline",    node_timeline)
    graph.add_node("node_root_cause",  node_root_cause)
    graph.add_node("node_remediation", node_remediation)

    # log and timeline run in parallel from START
    graph.add_edge(START,          "node_log")
    graph.add_edge(START,          "node_timeline")
    graph.add_edge("node_log",     "node_root_cause")
    graph.add_edge("node_timeline","node_root_cause")
    graph.add_edge("node_root_cause",  "node_remediation")
    graph.add_edge("node_remediation", END)

    return graph.compile()


def run(llm: ChatGroq, store: FAISS, error_counts: dict) -> dict[str, Any]:
    app = build_graph()
    return app.invoke({
        "llm": llm,
        "store": store,
        "error_counts": error_counts,
        "log_analysis": "",
        "timeline_analysis": "",
        "root_cause": "",
        "remediation": "",
    })