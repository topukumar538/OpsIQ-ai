# Location: backend/graph/nodes/rag.py
from pathlib import Path
from langchain.prompts import PromptTemplate

from core.memory import make_memory, get_history, save_turn
from core.retriever import retrieve
from graph.state import OpsState, RAG
from config import RAG_TOP_K

prompt = PromptTemplate.from_template("""
You are an expert assistant answering questions from uploaded documents.
You have access to both the conversation history and relevant document context.

When answering:
- Use the conversation history for personal context (names, preferences, prior discussion)
- Use the document context for factual questions about the uploaded documents
- If the answer is in neither, say so clearly rather than guessing
- You may receive overlapping or duplicate chunks — consolidate and avoid repeating

Conversation history:
{history}

Relevant document context:
{context}

Question: {input}
""".strip())


def rag_node(state: OpsState) -> OpsState:
    """
    Handles both document loading and Q&A in RAG mode.

    If file_path is set → load/merge document into FAISS store, seed memory.
    If file_path is empty → answer user's question from existing store.
    """
    file_path = state.get("file_path", "")
    llm       = state["llm"]

    if file_path:
        state["file_path"] = ""

        suffix = Path(file_path).suffix.lower()
        if suffix == ".log":
            state["rag_warning"] = (
                "Log files cannot be added in RAG mode. "
                "Open a new session to run a postmortem analysis."
            )
            return state

        existing = state.get("rag_store")
        if existing is None:
            from rag.ingest import build_rag_store
            store = build_rag_store(
                file_path,
                state["user_id"],
                state["session_token"],
            )
            state["rag_store"] = store
        else:
            from rag.ingest import add_to_store
            store = add_to_store(
                existing,
                file_path,
                state["user_id"],
                state["session_token"],
            )

        state["mode"] = RAG

        # Seed rag_memory with full chat history so context isn't lost
        # when switching from chat mode to RAG mode.
        # Copy both summary AND raw messages — summary alone is empty for
        # short conversations that haven't triggered summarisation yet.
        if state.get("rag_memory") is None:
            rag_memory  = make_memory(llm)
            chat_memory = state.get("chat_memory")

            if chat_memory:
                if chat_memory.moving_summary_buffer:
                    rag_memory.moving_summary_buffer = chat_memory.moving_summary_buffer
                for msg in chat_memory.chat_memory.messages:
                    rag_memory.chat_memory.add_message(msg)

            state["rag_memory"] = rag_memory

        return state

    # Q&A path
    memory  = state.get("rag_memory") or make_memory(llm)
    history = get_history(memory)
    context = retrieve(state["rag_store"], state["user_input"], RAG_TOP_K)
    filled  = prompt.format(history=history, context=context, input=state["user_input"])
    result  = llm.invoke(filled)
    response = str(result.content)

    save_turn(memory, state["user_input"], response)
    state["rag_memory"] = memory
    state["response"]   = response
    return state