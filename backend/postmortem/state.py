# Location: backend/postmortem/state.py
from typing import TypedDict, Any


class PostmortemState(TypedDict):
    llm: Any
    store: Any
    error_counts: dict
    log_analysis: str
    timeline_analysis: str
    root_cause: str
    remediation: str
    report_str: str
    report_summary: str