# Location: unified_ai/main.py
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from typing import Any
from langchain_groq import ChatGroq

from config import GROQ_API_KEY, MODEL_NAME, TEMPERATURE, TEMPERATURE
from router import classify_input, normalize_path
from models import chat as chat_mode
from models import rag as rag_mode
from models import postmortem as pm_mode
from postmortem import ingest as pm_ingest
from postmortem import graph as pm_graph
from postmortem import report as pm_report

# ── Mode constants ────────────────────────────────────────────────────────────
CHAT        = "chat"
RAG         = "rag"
POSTMORTEM  = "postmortem"

# ── Show memory helper ────────────────────────────────────────────────────────

def show_memory(memory) -> None:
    msgs = memory.chat_memory.messages
    print(f"\n[Summary]\n{memory.moving_summary_buffer or '(none yet)'}")
    print(f"[Buffer: {len(msgs)} messages]")
    for m in msgs:
        role = "Human" if m.type == "human" else "AI"
        print(f"  {role}: {str(m.content)[:120]}")
    print()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Unified AI ===")
    print("Chat freely. Drop a pdf/docx/txt to switch to RAG mode.")
    print("Drop a .log file to run PostMortem analysis.")
    print("Commands: mem | quit\n")

    llm = ChatGroq(api_key=GROQ_API_KEY, model=MODEL_NAME, temperature=TEMPERATURE) # type: ignore

    # State
    mode          = CHAT
    chat_memory   = chat_mode.build_memory(llm)
    rag_store:  Any = None
    rag_memory: Any = None
    pm_store:   Any = None
    pm_memory:  Any = None
    report_str: str = ""

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
            if mode == CHAT:
                show_memory(chat_memory)
            elif mode == RAG:
                show_memory(rag_memory)
            elif mode == POSTMORTEM:
                show_memory(pm_memory)
            continue

        kind = classify_input(user_input)

        # ── Bad path ──────────────────────────────────────────────────────────
        if kind == "bad_path":
            print("  [!] Unsupported or invalid file.\n")
            continue

        # ── Postmortem mode — locked, no files accepted ───────────────────────
        if mode == POSTMORTEM:
            if kind in {"rag_file", "log_file"}:
                print("  [!] This session is locked to the current postmortem report.")
                print("      Open a new session to analyze a different report.\n")
                continue
            answer = pm_mode.chat(user_input, report_str, pm_store, llm, pm_memory)
            print(f"\nAI: {answer}\n")
            continue

        # ── Log file → trigger postmortem pipeline ────────────────────────────
        if kind == "log_file":
            log_name = normalize_path(user_input).name
            print(f"\n  Reading '{log_name}'...")
            raw_log = pm_ingest.read_log(user_input)
            print(f"  {len(raw_log.splitlines())} lines read\n")

            print("  Building knowledge store...")
            pm_store, error_counts = pm_ingest.build_store(raw_log, llm)

            print("  Running PostMortem pipeline...")
            print("  (log_analyzer and timeline_analyzer running in parallel)\n")
            result = pm_graph.run(llm, pm_store, error_counts)

            report_str = pm_report.build_report(result, log_name)
            print(report_str)

            # Switch to postmortem chat mode
            mode      = POSTMORTEM
            pm_memory = pm_mode.build_memory(llm)

            # Build a fresh FAISS store from the report for RAG
            print("  Indexing report for chat...")
            pm_store = pm_mode.build_report_store(report_str)
            print(f"  {pm_store.index.ntotal} vectors ready.\n")
            print("  You can now ask questions about this report.")
            print("  This session is locked to this report.\n")
            continue

        # ── RAG file → switch to or stay in RAG mode ──────────────────────────
        if kind == "rag_file":
            file_name = normalize_path(user_input).name

            if mode == CHAT:
                # First doc — switch to RAG mode silently
                print(f"\n  Loading '{file_name}'...")
                rag_store  = rag_mode.load_file(user_input, None)
                rag_memory = rag_mode.build_memory(llm)
                mode       = RAG
                print("  RAG mode activated. Ask questions about your documents.\n")

            elif mode == RAG:
                # Additional doc — ask to merge
                print(f"\n  [!] '{file_name}' detected. Add to knowledge store? [add / cancel]: ", end="")
                if input().strip().lower() == "add":
                    rag_store = rag_mode.load_file(user_input, rag_store)
                else:
                    print("  Cancelled.\n")
            continue

        # ── Normal message → active mode handles it ───────────────────────────
        if mode == CHAT:
            answer = chat_mode.chat(user_input, llm, chat_memory)
        elif mode == RAG:
            answer = rag_mode.chat(user_input, rag_store, llm, rag_memory)

        print(f"\nAI: {answer}\n")


if __name__ == "__main__":
    main()