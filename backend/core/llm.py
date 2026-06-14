# Location: backend/core/llm.py
"""
LLM factory — one instance per mode at the right temperature.

Chat:        0.7  creative, conversational
RAG:         0.3  accurate, grounded in retrieved docs
Postmortem:  0.1  near-deterministic, reproducible incident analysis
"""
from langchain_groq import ChatGroq
from config import GROQ_API_KEY, MODEL_NAME, CHAT_TEMPERATURE, RAG_TEMPERATURE, PM_TEMPERATURE


def _make_llm(temperature: float) -> ChatGroq:
    return ChatGroq(
        api_key    =GROQ_API_KEY,
        model      =MODEL_NAME,       # FIX: ChatGroq uses 'model', not 'model_name'
        temperature=temperature,
        streaming  =True,
    )


def get_llm()     -> ChatGroq: return _make_llm(CHAT_TEMPERATURE)
def get_rag_llm() -> ChatGroq: return _make_llm(RAG_TEMPERATURE)
def get_pm_llm()  -> ChatGroq: return _make_llm(PM_TEMPERATURE)