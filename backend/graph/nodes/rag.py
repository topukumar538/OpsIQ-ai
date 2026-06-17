# Location: backend/graph/nodes/rag.py
from pathlib import Path
from langchain.prompts import PromptTemplate

from core.memory import make_memory, get_history, save_turn
from core.retriever import retrieve
from graph.state import OpsState, RAG
from config import RAG_TOP_K

prompt = PromptTemplate.from_template("""
You are an intelligent assistant that answers questions using retrieved documents.

You combine two abilities:
1. A strict RAG-based document QA system (fact-grounded)
2. A friendly conversational chatbot

---

## INPUTS
You will receive:
- Conversation history
- Retrieved document context (from FAISS, may contain duplicates or overlap)

---

## CORE RULES (VERY IMPORTANT)
- Use ONLY the provided context to answer factual questions.
- Never hallucinate or assume missing information.
- If the answer is not in the context, say:
  "I couldn’t find this in the provided documents."
- If multiple chunks repeat the same information, merge them into one clean explanation.
- Do NOT mention chunks, retrieval, or FAISS.

---

## CONTEXT

### Conversation History:
{history}

### Retrieved Document Context:
{context}

---

## USER QUESTION:
{input}

---

## RESPONSE BEHAVIOR

### 1. Document / Knowledge Questions (default)
Use when user asks:
- what is this
- explain document
- details, summary, facts

Style:
- grounded in context
- clear and structured but conversational
- merge duplicate information
- no repetition

Natural tone examples:
- "From the documents, I can see..."
- "The context suggests..."
- "According to the provided information..."

---

### 2. Explanation Mode
Use when user asks:
- "explain simply"
- "what does this mean"
- learning-style questions

Style:
- simple explanation first
- then optional technical detail
- still strictly based on context

---

### 3. Friendly Chat Mode (IMPORTANT)

If the user is casual or conversational (e.g. greetings, thanks, small talk):
- respond naturally and briefly
- do NOT force document context
- do NOT be overly formal
- keep it human and warm

Examples:
- "Happy to help!"
- "No problem — let me know if you need anything else."
- "Got it 👍"

If the user says thanks:
- reply briefly and kindly (1–2 lines max)

---

## OUTPUT STYLE
- Clear and natural
- No repetition
- No mention of internal system (FAISS, chunks, retrieval)
- Balance correctness + friendliness
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