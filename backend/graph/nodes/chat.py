# Location: backend/graph/nodes/chat.py
from langchain.prompts import PromptTemplate
from core.memory import make_memory, get_history, save_turn
from graph.state import OpsState

prompt = PromptTemplate.from_template("""
You are OpsIQ, a helpful AI assistant for DevOps and site reliability engineering.
Be concise, accurate, and practical. Use technical language appropriate for engineers.

Conversation history:
{history}

User: {input}
Assistant:""".strip())


def chat_node(state: OpsState) -> OpsState:
    """Standard chat — no document context, just conversation history."""
    llm    = state["llm"]
    memory = state.get("chat_memory") or make_memory(llm)

    history  = get_history(memory)
    filled   = prompt.format(history=history, input=state["user_input"])
    result   = llm.invoke(filled)
    response = str(result.content)

    save_turn(memory, state["user_input"], response)
    state["chat_memory"] = memory
    state["response"]    = response
    return state