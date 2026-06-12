# Location: backend/postmortem/nodes/timeline.py
from core.retriever import retrieve
from config import PM_TOP_K


def timeline(state: dict) -> dict:
    print("  [timeline] Analyzing event sequence and timestamps...")
    context  = retrieve(state["store"], "timestamp time sequence order of events when started duration recovery", PM_TOP_K)
    response = state["llm"].invoke(
        "You are an expert SRE analyzing an incident timeline.\n\n"
        "Using the context below, reconstruct:\n"
        "1. Sequence of events in chronological order\n"
        "2. When the incident started and was resolved\n"
        "3. Duration of the incident\n"
        "4. Key moments — escalation, peak, recovery\n\n"
        f"Context:\n{context}"
    )
    return {"timeline_analysis": str(response.content)}