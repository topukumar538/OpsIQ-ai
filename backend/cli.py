# Location: backend/cli.py
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_groq import ChatGroq

from config import GROQ_API_KEY, MODEL_NAME, TEMPERATURE
from router import classify_input, normalize_path
from session import SessionState, CHAT, RAG, POSTMORTEM
import modes.chat as chat_mode
import modes.rag as rag_mode
import modes.postmortem as pm_mode
import postmortem.ingest as pm_ingest
import postmortem.graph as pm_graph
import postmortem.report as pm_report


def show_memory(state: SessionState) -> None:
    if state.mode == CHAT:
        memory = state.chat_memory
    elif state.mode == RAG:
        memory = state.rag_memory
    else:
        memory = state.pm_memory

    if not memory:
        print("  (no memory yet)\n")
        return

    msgs = memory.chat_memory.messages
    print(f"\n[Summary]\n{memory.moving_summary_buffer or '(none yet)'}")
    print(f"[Buffer: {len(msgs)} messages]")
    for m in msgs:
        role = "Human" if m.type == "human" else "AI"
        print(f"  {role}: {str(m.content)[:120]}")
    print()


def main():
    print("=== OpsIQ (CLI) ===")
    print("Chat freely. Drop a pdf/docx/txt to switch to RAG mode.")
    print("Drop a .log file to run PostMortem analysis.")
    print("Commands: mem | quit\n")

    llm   = ChatGroq(api_key=GROQ_API_KEY, model=MODEL_NAME, temperature=TEMPERATURE) # type: ignore
    state = SessionState()
    state.chat_memory = chat_mode.build_memory(llm)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if user_input.lower() == "mem":
            show_memory(state)
            continue

        kind = classify_input(user_input)

        if kind == "bad_path":
            print("  [!] Unsupported or invalid file.\n")
            continue

        # Postmortem mode — locked
        if state.mode == POSTMORTEM:
            if kind in {"rag_file", "log_file"}:
                print("  [!] Session locked to current report. Open a new session.\n")
                continue
            answer = pm_mode.chat(user_input, state.report_str, state.pm_store, llm, state.pm_memory)
            print(f"\nAI: {answer}\n")
            continue

        # Log file → postmortem pipeline
        if kind == "log_file":
            log_name = normalize_path(user_input).name
            print(f"\n  Reading '{log_name}'...")
            raw_log = pm_ingest.read_log(user_input)
            print(f"  {len(raw_log.splitlines())} lines read\n")

            print("  Building knowledge store...")
            store, error_counts = pm_ingest.build_store(raw_log, llm)

            print("  Running PostMortem pipeline...")
            print("  (log_analyzer and timeline_analyzer running in parallel)\n")
            result = pm_graph.run(llm, store, error_counts)

            state.report_str = pm_report.build_report(result, log_name)
            print(state.report_str)

            state.mode      = POSTMORTEM
            state.pm_memory = pm_mode.build_memory(llm)
            state.pm_store  = pm_mode.build_report_store(state.report_str)
            print(f"  {state.pm_store.index.ntotal} vectors ready.\n")
            print("  You can now ask questions about this report.\n")
            continue

        # RAG file
        if kind == "rag_file":
            import doc_ingest
            file_name = normalize_path(user_input).name
            if state.mode == CHAT:
                print(f"\n  Loading '{file_name}'...")
                state.rag_store  = doc_ingest.build_store(user_input)
                state.rag_memory = rag_mode.build_memory(llm)
                state.mode       = RAG
                print(f"  Done. {state.rag_store.index.ntotal} vectors loaded.")
                print("  RAG mode activated.\n")
            elif state.mode == RAG:
                print(f"\n  [!] '{file_name}' detected. Add to store? [add / cancel]: ", end="")
                if input().strip().lower() == "add":
                    doc_ingest.add_to_store(state.rag_store, user_input)
                    print(f"  Added. {state.rag_store.index.ntotal} vectors.\n")
                else:
                    print("  Cancelled.\n")
            continue

        # Normal message
        if state.mode == CHAT:
            answer = chat_mode.chat(user_input, llm, state.chat_memory)
        elif state.mode == RAG:
            answer = rag_mode.chat(user_input, state.rag_store, llm, state.rag_memory)

        print(f"\nAI: {answer}\n")


if __name__ == "__main__":
    main()