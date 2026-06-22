# Location: backend/postmortem/ingest.py
import re
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import PM_CHUNK_LINES, PM_OVERLAP_LINES




def read_log(filepath: str) -> str:
    return Path(filepath).read_text(encoding="utf-8", errors="ignore")


def chunk_by_lines(text: str) -> list[Document]:
    lines = text.splitlines()
    chunks, i = [], 0
    while i < len(lines):
        content = "\n".join(lines[i: i + PM_CHUNK_LINES]).strip()
        if content:
            chunks.append(Document(
                page_content=content,
                metadata={"chunk_index": len(chunks), "start_line": i + 1},
            ))
        i += PM_CHUNK_LINES - PM_OVERLAP_LINES
    return chunks


# Single regex pattern
ERROR_PATTERN = re.compile(r"([A-Za-z]+(?:Error|Exception|Failure|Failed|Critical|Fatal))", re.IGNORECASE)

def extract_errors(text: str) -> dict:
    """
    Count error occurrences per named error type.
    """
    error_counts = {}
    for line in text.splitlines():
        match = ERROR_PATTERN.search(line)
        if match:
            name = match.group(1)
            error_counts[name] = error_counts.get(name, 0) + 1
    return error_counts



# Accept embeddings as a parameter instead of constructing inside
def build_store(raw_log: str, llm, embeddings) -> tuple:
    print("  Chunking log...")
    chunks = chunk_by_lines(raw_log)
    print(f"  {len(chunks)} chunks created")

    error_counts = extract_errors(raw_log)
    print(f"  {len(error_counts)} unique error type(s) detected")

    # Skip noisy "None detected" doc entirely if no errors found
    extra_docs = []
    if error_counts:
        error_lines = "\n".join([f"- {n}: {c} occurrence(s)" for n, c in error_counts.items()])
        extra_docs.append(Document(
            page_content=f"Major errors found:\n{error_lines}",
            metadata={"type": "error_summary"},
        ))

    print("  Generating log summary...")
    # Slice at a newline boundary to avoid cutting mid-sentence
    safe_slice = raw_log[:8000].rsplit('\n', 1)[0]
    summary_response = llm.invoke(
        f"Summarize this log in 5-8 sentences. Focus on services involved, "
        f"what went wrong, and the overall timeline.\n\n{safe_slice}"
    )
    summary_text = summary_response.content
    extra_docs.append(Document(
        page_content=f"Log summary:\n{summary_text}",
        metadata={"type": "llm_summary"},
    ))

    all_docs = chunks + extra_docs
    print(f"  Embedding {len(all_docs)} documents into FAISS...")
    store = FAISS.from_documents(all_docs, embeddings)
    print(f"  FAISS store ready. {store.index.ntotal} vectors.\n")

    # Return summary_text so callers can surface it without re-querying
    return store, error_counts


def add_report_to_store(store: FAISS, report_str: str) -> None:
    doc = Document(page_content=report_str, metadata={"type": "postmortem_report"})
    store.add_documents([doc])