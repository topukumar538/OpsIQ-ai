# Location: backend/cli.py
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from core.llm import get_llm
from graph.builder import build_graph, make_initial_state
from graph.state import POSTMORTEM, RAG
from router import classify_cli_input


def show_memory(state: dict) -> None:
    mode   = state["mode"]
    memory = (
        state["chat_memory"] if mode == "chat" else
        state["rag_memory"]  if mode == RAG    else
        state["pm_memory"]
    )
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
    print("Chat freely. Drop a pdf/docx/txt for RAG. Drop a .log for PostMortem.")
    print("Commands: mem | quit\n")

    llm   = get_llm()
    graph = build_graph()
    state = make_initial_state(llm)

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

        kind = classify_cli_input(user_input)

        if kind == "bad_path":
            print("  [!] Unsupported or invalid file.\n")
            continue

        if state["mode"] == RAG and kind == "log_file":
            print("  [!] Log files cannot be added in RAG mode. Open a new session for PostMortem analysis.\n")
            continue

        if state["mode"] == POSTMORTEM and kind in {"rag_file", "log_file"}:
            print("  [!] Session locked to current report. Open a new session.\n")
            continue

        if kind in {"rag_file", "log_file"}:
            state["user_input"] = ""
            state["file_path"]  = user_input
        else:
            state["user_input"] = user_input
            state["file_path"]  = ""

        state = graph.invoke(state)

        if state.get("rag_warning"):
            print(f"  [!] {state['rag_warning']}\n")
            state["rag_warning"] = ""
            continue

        if state["response"] and kind == "message":
            print(f"\nAI: {state['response']}\n")
        elif state["mode"] == POSTMORTEM and state.get("report_str") and not state.get("_report_printed"):
            print(state["report_str"])
            state["_report_printed"] = True
            print("  Session locked. Ask questions about this report.\n")
        elif kind in {"rag_file", "log_file"} and state["mode"] == RAG:
            print("  Document loaded. Ask questions about it.\n")


if __name__ == "__main__":
    main()
