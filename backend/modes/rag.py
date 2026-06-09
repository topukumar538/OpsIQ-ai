# Location: backend/modes/rag.py
from pathlib import Path

from langchain_groq import ChatGroq
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

from config import RAG_TOKEN_LIMIT
from router import normalize_path
import doc_ingest
from router import normalize_path

TEMPLATE = (
    "You are a document-aware AI assistant. Answer using the retrieved context. "
    "If the answer is not in the context, fall back to your own knowledge and say so.\n"
    "Use conversation history for follow-up continuity.\n\n"
    "Conversation History:\n{history}\n\n"
    "Retrieved Context:\n{context}\n\n"
    "Human: {input}\n"
    "AI:"
)

prompt = PromptTemplate(input_variables=["history", "context", "input"], template=TEMPLATE)


def build_memory(llm: ChatGroq) -> ConversationSummaryBufferMemory:
    return ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=RAG_TOKEN_LIMIT,
        memory_key="history",
        human_prefix="Human",
        ai_prefix="AI",
        return_messages=False,
    )


def format_context(docs) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = Path(doc.metadata.get("source", "unknown")).name
        page   = doc.metadata.get("page", "")
        label  = f"[{i}] {source}" + (f" p.{page + 1}" if page != "" else "")
        parts.append(f"{label}\n{doc.page_content.strip()}")
    return "\n\n".join(parts)


def load_file(filepath: str, store: FAISS | None) -> FAISS:
    # Build new store or merge into existing
    if store is None:
        store = doc_ingest.build_store(filepath)
        print(f"  Done. {store.index.ntotal} vectors in store.\n")
    else:
        doc_ingest.add_to_store(store, filepath)
        print(f"  Added. Store now has {store.index.ntotal} vectors.\n")
    return store


def chat(user_input: str, store: FAISS, llm: ChatGroq, memory: ConversationSummaryBufferMemory) -> str:
    history  = memory.load_memory_variables({})["history"]
    docs     = doc_ingest.get_retriever(store).invoke(user_input)
    context  = format_context(docs)
    response = llm.invoke(prompt.format(history=history, context=context, input=user_input))
    answer   = str(response.content)
    memory.save_context({"input": user_input}, {"output": answer})
    return answer