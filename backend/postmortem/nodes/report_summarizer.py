# Location: backend/postmortem/nodes/report_summarizer.py
from postmortem.ingest import add_report_to_store
from postmortem.report import build_report


def report_summarizer(state: dict) -> dict:
    print("  [report_summarizer] Building report and generating memory context...")

    # Build full report string
    report_str = build_report(state, state.get("log_filename", "incident.log"))

    # Add full report to existing FAISS store so chat can retrieve it
    add_report_to_store(state["store"], report_str)
    print(f"  Report added to FAISS. Total vectors: {state['store'].index.ntotal}")

    # Generate detailed memory context — this seeds the postmortem chatbot memory
    # Goal: give the LLM a rich, structured summary it can reference in every turn
    response = state["llm"].invoke(
        "You are an expert SRE writing a detailed incident briefing for an AI assistant.\n"
        "This briefing will be stored as the AI's memory so it can accurately answer questions about this incident.\n\n"
        "Based on the postmortem report below, write a comprehensive briefing that covers:\n\n"
        "1. INCIDENT OVERVIEW — What happened, when, which services were affected, and severity level\n"
        "2. ERRORS & FAILURES — All specific error types, their names, counts, and which components they hit\n"
        "3. TIMELINE — Precise chronological sequence: when it started, escalated, peaked, and resolved\n"
        "4. ROOT CAUSE — The exact root cause, contributing factors, and why it cascaded\n"
        "5. REMEDIATION — Immediate actions taken, short-term fixes, long-term prevention, and owners\n"
        "6. KEY FACTS — Any specific numbers, thresholds, service names, error codes, or durations worth remembering\n\n"
        "Be detailed and specific. Preserve exact names, numbers, and technical details from the report.\n"
        "This briefing will be the AI's primary reference — accuracy and completeness matter most.\n\n"
        f"POSTMORTEM REPORT:\n{report_str}"
    )
    report_summary = str(response.content)
    print("  Memory context generated.\n")

    return {"report_str": report_str, "report_summary": report_summary}