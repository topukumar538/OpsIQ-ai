# Location: backend/graph/nodes/rag.py
from langchain.prompts import PromptTemplate
from core.memory import get_history, save_turn, build_memory, seed_memory
from core.retriever import retrieve
from rag.ingest import build_store, add_to_store
from router import normalize_path, classify_input
from config import RAG_TOP_K
from graph.state import RAG, POSTMORTEM

TEMPLATE = (
    "You are a document-aware AI assistant. Answer using the retrieved context. "
    "If the answer is not in the context, fall back to your own knowledge and say so.\n"
    "Use conversation history for follow-up continuity.\n\n"
    "Conversation History:\n{history}\n\n"
    "Retrieved Context:\n{context}\n\n"
    "Human: {input}\nAI:"
)

prompt = PromptTemplate(input_variables=["history", "context", "input"], template=TEMPLATE)


def rag_node(state: dict) -> dict:
    llm   = state["llm"]
    store = state["rag_store"]
    fp    = state.get("file_path") or ""

    # File upload
    if fp:
        kind = classify_input(fp)

        # Log file in RAG mode — reject, don't transition
        if kind == "log_file":
            return {
                "mode": RAG, "response": "",
                "file_path": "",  # clear so routing ends here
                "llm": llm,
                "rag_store": store,
                "rag_memory": state["rag_memory"],
                "rag_warning": "Log files cannot be added in RAG mode. Open a new session for PostMortem analysis.",
            }

        # RAG file — build or merge store
        if store is None:
            # First doc — seed RAG memory with full chat history
            memory = build_memory(llm)
            chat_mem = state.get("chat_memory")
            if chat_mem:
                parts = []
                if chat_mem.moving_summary_buffer:
                    parts.append(f"Summary of earlier conversation:\n{chat_mem.moving_summary_buffer}")
                msgs = chat_mem.chat_memory.messages
                if msgs:
                    recent = "\n".join([
                        f"{'Human' if m.type == 'human' else 'AI'}: {str(m.content)}"
                        for m in msgs
                    ])
                    parts.append(f"Recent conversation:\n{recent}")
                if parts:
                    seed_memory(memory, "\n\n".join(parts))
            print(f"  Loading '{normalize_path(fp).name}'...")
            store = build_store(fp)
            print(f"  {store.index.ntotal} vectors loaded.\n")
        else:
            memory = state["rag_memory"] or build_memory(llm)
            print(f"  Adding '{normalize_path(fp).name}' to store...")
            add_to_store(store, fp)
            print(f"  Store now has {store.index.ntotal} vectors.\n")

        return {
            "rag_store": store, "rag_memory": memory,
            "mode": RAG, "response": "", "file_path": "", "llm": llm,
        }

    # Normal question — retrieve and answer
    memory  = state["rag_memory"] or build_memory(llm)
    history = get_history(memory)
    context = retrieve(store, state["user_input"], RAG_TOP_K)
    response = llm.invoke(prompt.format(history=history, context=context, input=state["user_input"]))
    answer   = str(response.content)

    save_turn(memory, state["user_input"], answer)
    return {
        "response": answer, "rag_memory": memory,
        "rag_store": store, "mode": RAG, "llm": llm,
    }