# Location: backend/core/memory.py
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain.memory import ConversationSummaryBufferMemory
from langchain_groq import ChatGroq
from config import MAX_TOKEN_LIMIT


def build_memory(llm: ChatGroq) -> ConversationSummaryBufferMemory:
    return ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=MAX_TOKEN_LIMIT,
        memory_key="history",
        human_prefix="Human",
        ai_prefix="AI",
        return_messages=False,
    )


def seed_memory(memory: ConversationSummaryBufferMemory, content: str) -> None:
    # Seed memory with initial context (e.g. report summary)
    memory.save_context(
        {"input": "Here is the context for this session."},
        {"output": content},
    )


def get_history(memory: ConversationSummaryBufferMemory) -> str:
    return memory.load_memory_variables({})["history"]


def save_turn(memory: ConversationSummaryBufferMemory, user: str, ai: str) -> None:
    memory.save_context({"input": user}, {"output": ai})