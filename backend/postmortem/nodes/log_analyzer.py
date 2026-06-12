# Location: backend/postmortem/nodes/log_analyzer.py
from core.retriever import retrieve
from config import PM_TOP_K


def log_analyzer(state: dict) -> dict:
    print("  [log_analyzer] Analyzing errors, services, severity...")
    context  = retrieve(state["store"], "errors critical failures affected services severity level", PM_TOP_K)
    response = state["llm"].invoke(
        "You are an expert SRE analyzing an incident log.\n\n"
        "Using the context below, identify:\n"
        "1. All errors and failures that occurred\n"
        "2. Affected services or components\n"
        "3. Severity level (critical / high / medium / low)\n"
        "4. Any patterns or repeated failures\n\n"
        f"Context:\n{context}"
    )
    return {"log_analysis": str(response.content)}