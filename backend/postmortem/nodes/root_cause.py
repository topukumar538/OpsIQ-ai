# Location: backend/postmortem/nodes/root_cause.py
import logging

from core.retriever import retrieve
from config import PM_TOP_K

logger = logging.getLogger(__name__)


def root_cause(state: dict) -> dict:
    """
    Fan-in node — runs only after both log_analyzer and timeline complete,
    and uses both of their outputs as context.
    """
    logger.info("root_cause: identifying root cause")
    context  = retrieve(state["store"], "root cause trigger reason why failure cascading dependency", PM_TOP_K)
    response = state["llm"].invoke(
        "You are an expert SRE performing root cause analysis.\n\n"
        f"Log Analysis:\n{state.get('log_analysis', '')}\n\n"
        f"Timeline Analysis:\n{state.get('timeline_analysis', '')}\n\n"
        f"Additional Context:\n{context}\n\n"
        "Identify:\n"
        "1. The root cause of the incident\n"
        "2. Contributing factors\n"
        "3. Why it cascaded or escalated\n"
        "4. Confidence level (high / medium / low) and why"
    )
    return {"root_cause": str(response.content)}