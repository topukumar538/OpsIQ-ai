# Location: backend/core/llm.py
"""
LLM factory.

Why separate instances per mode:
    All modes previously shared one LLM instance at TEMPERATURE=0.7.
    Temperature controls output randomness:

      0.7  →  creative, varied       →  good for chat
      0.3  →  accurate, readable     →  good for RAG Q&A
      0.1  →  near-deterministic     →  good for postmortem analysis

    Using 0.7 for postmortem means root cause, timeline, and remediation
    nodes can hallucinate failure causes or give different answers for the
    same log file on repeated runs. Now each context gets the right temp.

Usage:
    from core.llm import get_llm, get_rag_llm, get_pm_llm

    chat_llm = get_llm()        # temperature = CHAT_TEMPERATURE (0.7)
    rag_llm  = get_rag_llm()    # temperature = RAG_TEMPERATURE  (0.3)
    pm_llm   = get_pm_llm()     # temperature = PM_TEMPERATURE   (0.1)
"""
from langchain_groq import ChatGroq
from config import (
    GROQ_API_KEY,
    MODEL_NAME,
    CHAT_TEMPERATURE,
    RAG_TEMPERATURE,
    PM_TEMPERATURE,
)


def _make_llm(temperature: float) -> ChatGroq:
    """Base factory — creates a ChatGroq instance at the given temperature."""
    return ChatGroq(
        api_key    =GROQ_API_KEY,
        model_name =MODEL_NAME,
        temperature=temperature,
        streaming  =True,
    )


def get_llm() -> ChatGroq:
    """Chat mode LLM — creative, conversational."""
    return _make_llm(CHAT_TEMPERATURE)


def get_rag_llm() -> ChatGroq:
    """RAG mode LLM — accurate answers grounded in retrieved documents."""
    return _make_llm(RAG_TEMPERATURE)


def get_pm_llm() -> ChatGroq:
    """
    Postmortem LLM — near-deterministic for reproducible incident analysis.
    Used by all postmortem pipeline nodes: log_analyzer, timeline,
    root_cause, remediation, report_summarizer.
    """
    return _make_llm(PM_TEMPERATURE)