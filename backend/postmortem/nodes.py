# Location: backend/postmortem/nodes.py
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS

from config import PM_TOP_K


def retrieve(store: FAISS, query: str) -> str:
    docs = store.as_retriever(search_kwargs={"k": PM_TOP_K}).invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])


def node_log(state: dict) -> dict:
    llm: ChatGroq = state["llm"]
    store: FAISS  = state["store"]
    print("  [log_analyzer] Analyzing errors, services, severity...")
    context  = retrieve(store, "errors critical failures affected services severity level")
    response = llm.invoke(
        "You are an expert SRE analyzing an incident log.\n\n"
        "Using the context below, identify:\n"
        "1. All errors and failures that occurred\n"
        "2. Affected services or components\n"
        "3. Severity level (critical / high / medium / low)\n"
        "4. Any patterns or repeated failures\n\n"
        f"Context:\n{context}"
    )
    return {"log_analysis": response.content}


def node_timeline(state: dict) -> dict:
    llm: ChatGroq = state["llm"]
    store: FAISS  = state["store"]
    print("  [timeline_analyzer] Analyzing event sequence and timestamps...")
    context  = retrieve(store, "timestamp time sequence order of events when started duration recovery")
    response = llm.invoke(
        "You are an expert SRE analyzing an incident timeline.\n\n"
        "Using the context below, reconstruct:\n"
        "1. Sequence of events in chronological order\n"
        "2. When the incident started and when it was resolved\n"
        "3. Duration of the incident\n"
        "4. Key moments — when it escalated, peaked, and recovered\n\n"
        f"Context:\n{context}"
    )
    return {"timeline_analysis": response.content}


def node_root_cause(state: dict) -> dict:
    llm: ChatGroq = state["llm"]
    store: FAISS  = state["store"]
    print("  [root_cause] Identifying root cause...")
    context  = retrieve(store, "root cause trigger reason why failure cascading dependency")
    response = llm.invoke(
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
    return {"root_cause": response.content}


def node_remediation(state: dict) -> dict:
    llm: ChatGroq = state["llm"]
    store: FAISS  = state["store"]
    print("  [remediation] Generating remediation plan...")
    context  = retrieve(store, "fix recovery restart workaround solution prevention mitigation")
    response = llm.invoke(
        "You are an expert SRE creating a remediation plan.\n\n"
        f"Root Cause:\n{state.get('root_cause', '')}\n\n"
        f"Additional Context:\n{context}\n\n"
        "Provide:\n"
        "1. Immediate actions to resolve the incident\n"
        "2. Short-term fixes to prevent recurrence\n"
        "3. Long-term improvements and preventive measures\n"
        "4. Who should own each action item"
    )
    return {"remediation": response.content}