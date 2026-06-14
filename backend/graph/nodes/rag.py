# Location: backend/graph/nodes/rag.py
from pathlib import Path
from langchain.prompts import PromptTemplate

from core.memory import make_memory, get_history, save_turn
from core.retriever import build_store, retrieve          # build_store lives in retriever
from graph.state import OpsState, RAG
from config import RAG_TOP_K

prompt = PromptTemplate.from_template("""
You are an expert assistant answering questions from uploaded documents.
You may receive overlapping or duplicate chunks if the same document was
uploaded more than once — always consolidate your answer and avoid repeating
information.

Conversation history:
{history}

Relevant document context:
{context}

Question: {input}

Answer based only on the provided context. If the answer is not in the context,
say so clearly rather than guessing.
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
        # Document ingestion path — build_store and add_to_store handle
        # embedding and persistence. Memory seeding happens in session.py
        # after the pipeline completes so it has access to the DB session.
        state["file_path"] = ""

        suffix = Path(file_path).suffix.lower()
        if suffix == ".log":
            # Log files cannot be processed in RAG mode
            state["rag_warning"] = (
                "Log files cannot be added in RAG mode. "
                "Open a new session to run a postmortem analysis."
            )
            return state

        existing = state.get("rag_store")
        if existing is None:
            from rag.ingest import build_rag_store
            # user_id and token are passed via state for FAISS path construction
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
        if state.get("rag_memory") is None:
            state["rag_memory"] = make_memory(llm)

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