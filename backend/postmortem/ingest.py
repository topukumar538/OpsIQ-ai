# Location: backend/postmortem/ingest.py
import logging
import re
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import PM_CHUNK_LINES, PM_OVERLAP_LINES

logger = logging.getLogger(__name__)


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


# Two-tier error detection.
#
# Tier 1 — named error classes (DatabaseException, ThreadExhaustionFailure).
# Tier 2 — bare severity keywords (ERROR, FATAL, CRITICAL, TRACEBACK).
#
# The previous single pattern required a letter before the keyword, so a line
# reading "ERROR disk full" matched nothing and the report claimed no errors
# were found. Tier 1 runs first so a line with a real class name is attributed
# to that class instead of the generic ERROR bucket.
#
# "Failed" is deliberately excluded from tier 1 — it captures noise like
# "requestFailed=false" from ordinary INFO lines.
NAMED_ERROR_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z0-9]*(?:Error|Exception|Failure|Fault))\b"
)
SEVERITY_PATTERN = re.compile(
    r"\b(CRITICAL|FATAL|ERROR|SEVERE|PANIC|TRACEBACK|EMERGENCY|ALERT)\b",
    re.IGNORECASE,
)


def extract_errors(text: str) -> dict[str, int]:
    """
    Count error occurrences per type, at most one match per line.

    Named error classes take priority over bare severity keywords, so
    "ERROR DatabaseException: refused" counts as DatabaseException, not both.
    Severity keywords are uppercased so "Error" and "ERROR" don't become two
    separate entries.
    """
    error_counts: dict[str, int] = {}
    for line in text.splitlines():
        match = NAMED_ERROR_PATTERN.search(line)
        if match:
            name = match.group(1)            # keep case: DatabaseException
        else:
            match = SEVERITY_PATTERN.search(line)
            if not match:
                continue
            name = match.group(1).upper()    # normalise: Error -> ERROR
        error_counts[name] = error_counts.get(name, 0) + 1
    return error_counts

# Accept embeddings as a parameter instead of constructing inside
def build_store(raw_log: str, llm, embeddings) -> tuple[FAISS, dict[str, int]]:
    logger.info("Chunking log...")
    chunks = chunk_by_lines(raw_log)
    logger.info("%d chunks created", len(chunks))

    error_counts = extract_errors(raw_log)
    logger.info("%d unique error type(s) detected", len(error_counts))

    # Skip noisy "None detected" doc entirely if no errors found
    extra_docs = []
    if error_counts:
        error_lines = "\n".join([f"- {n}: {c} occurrence(s)" for n, c in error_counts.items()])
        extra_docs.append(Document(
            page_content=f"Major errors found:\n{error_lines}",
            metadata={"type": "error_summary"},
        ))

    logger.info("Generating log summary...")
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
    logger.info("Embedding %d documents into FAISS...", len(all_docs))
    store = FAISS.from_documents(all_docs, embeddings)
    logger.info("FAISS store ready — %d vectors", store.index.ntotal)


    return store, error_counts


def add_report_to_store(store: FAISS, report_str: str) -> None:
    doc = Document(page_content=report_str, metadata={"type": "postmortem_report"})
    store.add_documents([doc])