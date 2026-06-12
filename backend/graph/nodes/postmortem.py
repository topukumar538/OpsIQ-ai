# Location: backend/graph/nodes/postmortem.py
from langchain.prompts import PromptTemplate

from core.memory import build_memory, seed_memory, get_history, save_turn
from core.retriever import retrieve
from postmortem.ingest import read_log, build_store
from postmortem.builder import run_postmortem
from router import normalize_path
from config import PM_TOP_K
from graph.state import POSTMORTEM

TEMPLATE = (
    "You are an expert SRE assistant helping the user understand a postmortem report.\n"
    "Answer using the postmortem context. If the answer is not there, say so clearly.\n"
    "Use conversation history for follow-up continuity.\n\n"
    "Postmortem Report:\n{report}\n\n"
    "Conversation History:\n{history}\n\n"
    "Retrieved Context:\n{context}\n\n"
    "Human: {input}\nAI:"
)

prompt = PromptTemplate(
    input_variables=["report", "history", "context", "input"],
    template=TEMPLATE
)


def postmortem_node(state: dict) -> dict:
    llm = state["llm"]

    # If pipeline not run yet — run it
    if not state.get("report_str") and state.get("file_path"):
        log_name = normalize_path(state["file_path"]).name
        print(f"\n  Reading '{log_name}'...")
        raw_log = read_log(state["file_path"])
        print(f"  {len(raw_log.splitlines())} lines read\n")

        print("  Building knowledge store...")
        store, error_counts = build_store(raw_log, llm)

        print("  Running postmortem pipeline...")
        print("  (log_analyzer and timeline running in parallel)\n")
        result = run_postmortem(llm, store, error_counts, log_name)

        report_str     = result["report_str"]
        report_summary = result["report_summary"]
        pm_store       = result["store"]

        # Build memory and seed with report summary
        pm_memory = build_memory(llm)
        seed_memory(pm_memory, report_summary)

        print(f"\n  Session locked to postmortem report.\n")

        return {
            "mode":       POSTMORTEM,
            "pm_store":   pm_store,
            "pm_memory":  pm_memory,
            "report_str": report_str,
            "response":   report_str,
            "file_path":  "",
            "llm":        llm,
        }

    # Pipeline already run — chat about the report
    memory  = state["pm_memory"]
    store   = state["pm_store"]
    history = get_history(memory)
    context = retrieve(store, state["user_input"], PM_TOP_K)

    response = llm.invoke(prompt.format(
        report=state["report_str"],
        history=history,
        context=context,
        input=state["user_input"]
    ))
    answer = str(response.content)

    save_turn(memory, state["user_input"], answer)
    return {"response": answer, "pm_memory": memory, "mode": POSTMORTEM, "llm": llm}