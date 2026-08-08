# Location: backend/postmortem/nodes/report_summarizer.py
import logging

from postmortem.ingest import add_report_to_store
from postmortem.report import build_report

logger = logging.getLogger(__name__)


def report_summarizer(state: dict) -> dict:
    logger.info("report_summarizer: building report and memory context")

    report_str = build_report(state, state.get("log_filename", "incident.log"))

    # Add the full report to the FAISS store so chat can retrieve passages
    # from it alongside raw log chunks.
    add_report_to_store(state["store"], report_str)
    logger.info("Report added to FAISS — %d vectors", state["store"].index.ntotal)

    # Short summary that seeds pm_memory. Deliberately brief: the full report
    # is already passed into every postmortem chat prompt as {report}, so a
    # detailed briefing here would duplicate it — costing tokens on every
    # subsequent question for no added context.
    response = state["llm"].invoke(
        "Write a 3-4 sentence summary of this incident: what failed, the root "
        "cause, and the impact. This seeds an assistant's memory. The full "
        "report is available to it separately, so do not restate the report — "
        "give it a compact anchor to orient from.\n\n"
        f"POSTMORTEM REPORT:\n{report_str}"
    )
    report_summary = str(response.content)
    logger.info("Memory context generated")

    return {"report_str": report_str, "report_summary": report_summary}