# Location: backend/modes/postmortem.py
import warnings
from pathlib import Path

from langchain_groq import ChatGroq
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config import POSTMORTEM_TOKEN_LIMIT, EMBED_MODEL, PM_TOP_K

# Report passed as system context in the prompt — not in memory
# This avoids immediate summarization and detail loss
TEMPLATE = (
    "You are an expert SRE assistant helping the user understand a postmortem report.\n"
    "Answer using the report and retrieved context below. If the answer is not there, say so.\n"
    "Use conversation history for follow-up continuity.\n\n"
    "Postmortem Report:\n{report}\n\n"
    "Conversation History:\n{history}\n\n"
    "Retrieved Context:\n{context}\n\n"
    "Human: {input}\n"
    "AI:"
)

prompt = PromptTemplate(
    input_variables=["report", "history", "context", "input"],
    template=TEMPLATE
)

_embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def build_report_store(report: str) -> FAISS:
    # Chunk report into overlapping 15-line sections + full report doc
    lines  = [l.strip() for l in report.splitlines() if l.strip()]
    chunks = []
    i = 0
    while i < len(lines):
        content = "\n".join(lines[i: i + 15])
        chunks.append(Document(page_content=content, metadata={"chunk": len(chunks)}))
        i += 10
    chunks.append(Document(page_content=report, metadata={"chunk": "full"}))
    return FAISS.from_documents(chunks, _embeddings)


def build_memory(llm: ChatGroq) -> ConversationSummaryBufferMemory:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # Memory handles only the conversation turns — report lives in the prompt
    return ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=POSTMORTEM_TOKEN_LIMIT,
        memory_key="history",
        human_prefix="Human",
        ai_prefix="AI",
        return_messages=False,
    )


def chat(user_input: str, report: str, store: FAISS, llm: ChatGroq,
         memory: ConversationSummaryBufferMemory) -> str:
    history = memory.load_memory_variables({})["history"]
    docs    = store.as_retriever(search_kwargs={"k": PM_TOP_K}).invoke(user_input)
    context = "\n\n".join([doc.page_content for doc in docs])

    response = llm.invoke(
        prompt.format(report=report, history=history, context=context, input=user_input)
    )
    answer = str(response.content)
    memory.save_context({"input": user_input}, {"output": answer})
    return answer