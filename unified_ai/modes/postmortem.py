import warnings
from pathlib import Path

from langchain_groq import ChatGroq
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config import POSTMORTEM_TOKEN_LIMIT, EMBED_MODEL, PM_TOP_K

TEMPLATE = (
    "You are an expert SRE assistant helping the user understand a postmortem report.\n"
    "Answer using the postmortem context. If the answer is not there, say so clearly.\n"
    "Use conversation history for follow-up continuity.\n\n"
    "Conversation History:\n{history}\n\n"
    "Postmortem Context:\n{context}\n\n"
    "Human: {input}\n"
    "AI:"
)

prompt = PromptTemplate(input_variables=["history", "context", "input"], template=TEMPLATE)

_embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def build_report_store(report: str) -> FAISS:
    # Chunk the report into overlapping 15-line sections + full report doc
    lines  = [l.strip() for l in report.splitlines() if l.strip()]
    chunks = []
    i = 0
    while i < len(lines):
        content = "\n".join(lines[i: i + 15])
        chunks.append(Document(page_content=content, metadata={"chunk": len(chunks)}))
        i += 10
    chunks.append(Document(page_content=report, metadata={"chunk": "full"}))
    return FAISS.from_documents(chunks, _embeddings)


def build_memory(llm: ChatGroq, report: str) -> ConversationSummaryBufferMemory:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=POSTMORTEM_TOKEN_LIMIT,
        memory_key="history",
        human_prefix="Human",
        ai_prefix="AI",
        return_messages=False,
    )
    # Seed memory with the report so LLM has it from turn 1
    memory.save_context(
        {"input": "Here is the postmortem report for this session."},
        {"output": report},
    )
    return memory


def chat(user_input: str, store: FAISS, llm: ChatGroq, memory: ConversationSummaryBufferMemory) -> str:
    history  = memory.load_memory_variables({})["history"]
    docs     = store.as_retriever(search_kwargs={"k": PM_TOP_K}).get_relevant_documents(user_input)
    context  = "\n\n".join([doc.page_content for doc in docs])
    response = llm.invoke(prompt.format(history=history, context=context, input=user_input))
    answer   = str(response.content)
    memory.save_context({"input": user_input}, {"output": answer})
    return answer