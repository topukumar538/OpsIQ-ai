# Location: backend/postmortem/ingest.py
import re
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from config import PM_CHUNK_LINES, PM_OVERLAP_LINES
from core.retriever import get_embeddings
from router import normalize_path

ERROR_PATTERNS = ["ERROR", "CRITICAL", "FATAL", "EXCEPTION", "TRACEBACK", "FAILURE", "FAILED"]


def read_log(filepath: str) -> str:
    path = normalize_path(filepath)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_by_lines(text: str) -> list[Document]:
    lines = text.splitlines()
    chunks, i = [], 0
    while i < len(lines):
        content = "\n".join(lines[i: i + PM_CHUNK_LINES]).strip()
        if content:
            chunks.append(Document(
                page_content=content,
                metadata={"chunk_index": len(chunks), "start_line": i + 1}
            ))
        i += PM_CHUNK_LINES - PM_OVERLAP_LINES
    return chunks


def extract_errors(text: str) -> dict:
    error_counts = {}
    for line in text.splitlines():
        for pattern in ERROR_PATTERNS:
            if pattern in line.upper():
                match = re.search(r"([A-Za-z]+(?:Error|Exception|Failure|Failed|Critical|Fatal))", line)
                name = match.group(1) if match else pattern
                error_counts[name] = error_counts.get(name, 0) + 1
    return error_counts


def build_store(raw_log: str, llm) -> tuple:
    print("  Chunking log...")
    chunks = chunk_by_lines(raw_log)
    print(f"  {len(chunks)} chunks created")

    error_counts = extract_errors(raw_log)
    error_lines  = "\n".join([f"- {n}: {c} occurrence(s)" for n, c in error_counts.items()])
    error_doc    = Document(
        page_content=f"Major errors found:\n{error_lines or 'None detected'}",
        metadata={"type": "error_summary"}
    )
    print(f"  {len(error_counts)} unique error type(s) detected")

    print("  Generating log summary...")
    summary_response = llm.invoke(
        f"Summarize this log in 5-8 sentences. Focus on services involved, "
        f"what went wrong, and the overall timeline.\n\n{raw_log[:8000]}"
    )
    summary_doc = Document(
        page_content=f"Log summary:\n{summary_response.content}",
        metadata={"type": "llm_summary"}
    )

    all_docs = chunks + [error_doc, summary_doc]
    print(f"  Embedding {len(all_docs)} documents into FAISS...")
    store = FAISS.from_documents(all_docs, get_embeddings())
    print(f"  FAISS store ready. {store.index.ntotal} vectors.\n")

    return store, error_counts


def add_report_to_store(store: FAISS, report_str: str) -> None:
    # Add the full report as a document to the existing store
    doc = Document(page_content=report_str, metadata={"type": "postmortem_report"})
    store.add_documents([doc])