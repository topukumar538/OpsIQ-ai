# Location: backend/postmortem/nodes/remediation.py
from core.retriever import retrieve
from config import PM_TOP_K


def remediation(state: dict) -> dict:
    print("  [remediation] Generating remediation plan...")
    context  = retrieve(state["store"], "fix recovery restart solution prevention mitigation", PM_TOP_K)
    response = state["llm"].invoke(
        "You are an expert SRE creating a remediation plan.\n\n"
        f"Root Cause:\n{state.get('root_cause', '')}\n\n"
        f"Additional Context:\n{context}\n\n"
        "Provide:\n"
        "1. Immediate actions to resolve the incident\n"
        "2. Short-term fixes to prevent recurrence\n"
        "3. Long-term improvements and preventive measures\n"
        "4. Who should own each action item"
    )
    return {"remediation": str(response.content)}