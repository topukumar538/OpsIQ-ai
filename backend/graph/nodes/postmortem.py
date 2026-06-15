# Location: backend/graph/nodes/postmortem.py
from pathlib import Path

from langchain.prompts import PromptTemplate

from core.llm import get_pm_llm
from core.memory import make_memory, get_history, save_turn
from core.retriever import retrieve
from graph.state import OpsState, POSTMORTEM
from config import PM_TOP_K

prompt = PromptTemplate.from_template("""
You are a senior site reliability engineer reviewing a postmortem report.
Answer questions based on the incident report and retrieved log context.
Be precise, factual, and concise.

Incident report:
{report}

Conversation history:
{history}

Relevant log context:
{context}

Question: {input}
""".strip())


def postmortem_node(state: OpsState) -> OpsState:
    """
    Handles both initial log processing and follow-up Q&A in postmortem mode.
    """
    file_path = state.get("file_path", "")
    llm       = get_pm_llm()

    if file_path:
        from postmortem.builder import run_postmortem
        log_filename = Path(file_path).name

        pm_state = run_postmortem(
            log_path      = file_path,
            log_filename  = log_filename,
            llm           = llm,
            user_id       = state["user_id"],
            session_token = state["session_token"],
        )

        state["pm_store"]   = pm_state["pm_store"]
        state["report_str"] = pm_state["report_str"]
        state["mode"]       = POSTMORTEM
        state["is_locked"]  = True
        state["file_path"]  = ""

        if state.get("pm_memory") is None:
            pm_memory = make_memory(llm)
            summary = pm_state.get("report_summary", "")
            if summary:
                pm_memory.moving_summary_buffer = summary
            state["pm_memory"] = pm_memory

        return state

    memory  = state.get("pm_memory") or make_memory(llm)
    history = get_history(memory)
    context = retrieve(state["pm_store"], state["user_input"], PM_TOP_K)
    filled  = prompt.format(
        report  = state.get("report_str", ""),
        history = history,
        context = context,
        input   = state["user_input"],
    )
    result   = llm.invoke(filled)
    response = str(result.content)

    save_turn(memory, state["user_input"], response)
    state["pm_memory"] = memory
    state["response"]  = response
    return state
