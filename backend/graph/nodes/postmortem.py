# Location: backend/graph/nodes/postmortem.py
from pathlib import Path
 
from langchain.prompts import PromptTemplate
 
from core.memory import make_memory, get_history, save_turn
from core.retriever import retrieve
from graph.state import OpsState, POSTMORTEM
from config import PM_TOP_K
 

prompt = PromptTemplate.from_template("""
You are OpsIQ, a friendly and highly experienced Site Reliability Engineer.

You help users understand system incidents using retrieved context from:
- Postmortem reports
- Log chunks (FAISS retrieval)
- Conversation history

---

## CORE RULES
- Always base answers ONLY on provided context.
- Never guess or hallucinate missing information.
- If something is not in the context, say: "I don’t see that in the incident data."
- Be accurate first, helpful second.

---

## CONTEXT

### Incident Report:
{report}

### Conversation History:
{history}

### Retrieved Log Context (FAISS):
{context}

---

## USER QUESTION:
{input}

---

## RESPONSE BEHAVIOR

### 1. Incident Questions (default mode)
Use this when user asks:
- what happened
- why it failed
- root cause
- logs/errors
- timeline

Style:
- conversational SRE tone
- grounded in evidence
- mention logs naturally (no strict formatting)

Example style:
- "From the logs, I can see..."
- "This error started around..."
- "This suggests the issue likely came from..."

---

### 2. Explanation Mode
Use this when user asks:
- "what does this mean?"
- "explain simply"
- learning questions

Style:
- simple explanation first
- then optional technical detail
- still grounded in context

---

### 3. Social / Gratitude Mode (VERY IMPORTANT)

If the user says things like:
- thanks / thank you
- good job / well done
- nice / appreciate it
- or similar appreciation

Then:
- Respond briefly (1–2 lines max)
- Be warm and human
- Do NOT include logs or analysis
- Do NOT continue incident discussion unless asked

Examples:
- "Glad I could help 👍"
- "Happy to help — feel free to ask if you want to dig deeper."
- "Anytime, happy to help."

---

## OUTPUT PRINCIPLES
- Stay grounded in retrieved context at all times.
- Prefer clarity over complexity.
- Be concise unless user asks for detail.
- Do not format like a formal report unless explicitly requested.
""".strip())


def postmortem_node(state: OpsState) -> OpsState:
    """
    Handles both initial log processing and follow-up Q&A in postmortem mode.
    """
    file_path = state.get("file_path", "")
 
    # Use the session's cached pm_llm instead of calling get_pm_llm() every time.
    # get_pm_llm() constructs a brand new ChatGroq client on every invocation —
    # chat_node and rag_node both use state["llm"] correctly; this node was the
    # odd one out. pm_llm is set on the session state in make_initial_state().
    llm = state.get("pm_llm") or state["llm"] # type: ignore
 
    if file_path:
        from postmortem.builder import run_postmortem
        log_filename = Path(file_path).name
 
        pm_state = run_postmortem(
            log_path      = file_path,
            log_filename  = log_filename,
            llm           = llm,
            user_id       = state["user_id"], # type: ignore
            session_token = state["session_token"], # type: ignore
        )
 
        state["pm_store"]   = pm_state["pm_store"]
        state["report_str"] = pm_state["report_str"]
        state["mode"]       = POSTMORTEM
        state["is_locked"]  = True
        state["file_path"]  = ""
 
        if state.get("pm_memory") is None:
            pm_memory = make_memory(llm)
            summary   = pm_state.get("report_summary", "")
            if summary:
                pm_memory.moving_summary_buffer = summary
            state["pm_memory"] = pm_memory
 
        return state
 
    # Q&A path — use cached memory and store
    memory  = state.get("pm_memory") or make_memory(llm)
    history = get_history(memory)
    context = retrieve(state["pm_store"], state["user_input"], PM_TOP_K) # type: ignore
    filled  = prompt.format(
        report  = state.get("report_str", ""),
        history = history,
        context = context,
        input   = state["user_input"], # type: ignore
    )
    result   = llm.invoke(filled)
    response = str(result.content)
 
    save_turn(memory, state["user_input"], response) # type: ignore
    state["pm_memory"] = memory
    state["response"]  = response
    return state