# Location: backend/graph/nodes/chat.py
from langchain.prompts import PromptTemplate
from core.memory import get_history, save_turn, build_memory
from graph.state import CHAT

TEMPLATE = (
    "You are a sharp, helpful AI assistant. Answer directly and concisely.\n"
    "Use conversation history for follow-up continuity.\n\n"
    "Conversation History:\n{history}\n\n"
    "Human: {input}\nAI:"
)

prompt = PromptTemplate(input_variables=["history", "input"], template=TEMPLATE)


def chat_node(state: dict) -> dict:
    llm    = state["llm"]
    memory = state["chat_memory"] or build_memory(llm)

    # File upload — just pass through, routing handles transition
    if state.get("file_path"):
        return {"chat_memory": memory, "mode": CHAT, "response": "", "llm": llm}

    history  = get_history(memory)
    response = llm.invoke(prompt.format(history=history, input=state["user_input"]))
    answer   = str(response.content)

    save_turn(memory, state["user_input"], answer)
    return {"response": answer, "chat_memory": memory, "mode": CHAT, "llm": llm}