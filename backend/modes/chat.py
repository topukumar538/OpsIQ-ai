# Location: backend/modes/chat.py
from langchain_groq import ChatGroq
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate

from config import CHAT_TOKEN_LIMIT

TEMPLATE = (
    "You are a sharp, helpful AI assistant. Answer directly and concisely.\n"
    "Use conversation history for follow-up continuity.\n\n"
    "Conversation History:\n{history}\n\n"
    "Human: {input}\n"
    "AI:"
)

prompt = PromptTemplate(input_variables=["history", "input"], template=TEMPLATE)


def build_memory(llm: ChatGroq) -> ConversationSummaryBufferMemory:
    return ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=CHAT_TOKEN_LIMIT,
        memory_key="history",
        human_prefix="Human",
        ai_prefix="AI",
        return_messages=False,
    )


def chat(user_input: str, llm: ChatGroq, memory: ConversationSummaryBufferMemory) -> str:
    history  = memory.load_memory_variables({})["history"]
    response = llm.invoke(prompt.format(history=history, input=user_input))
    answer   = str(response.content)
    memory.save_context({"input": user_input}, {"output": answer})
    return answer