# Location: backend/postmortem/nodes/timeline.py
import logging

from core.retriever import retrieve
from config import PM_TOP_K

logger = logging.getLogger(__name__)


def timeline(state: dict) -> dict:
    """
    Returns only its own key. Runs in parallel with log_analyzer(), and
    returning a full state dict would let one node overwrite the other.
    """
    logger.info("timeline: analyzing event sequence and timestamps")
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