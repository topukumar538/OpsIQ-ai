# Location: backend/main.py
import warnings
import uuid
import asyncio
from pathlib import Path
from typing import AsyncGenerator

warnings.filterwarnings("ignore", category=DeprecationWarning)

from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_groq import ChatGroq

from config import GROQ_API_KEY, MODEL_NAME, TEMPERATURE
from router import classify_input, normalize_path
from session import get_session, delete_session, CHAT, RAG, POSTMORTEM
import modes.chat as chat_mode
import modes.rag as rag_mode
import modes.postmortem as pm_mode
import postmortem.ingest as pm_ingest
import postmortem.graph as pm_graph
import postmortem.report as pm_report

app = FastAPI(title="OpsIQ")

@app.get("/")
def root():
    return FileResponse("../frontend/index.html")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_llm() -> ChatGroq:
    return ChatGroq(api_key=GROQ_API_KEY, model=MODEL_NAME, temperature=TEMPERATURE) # type: ignore


# ── Request models ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


# ── Helpers ───────────────────────────────────────────────────────────────────

async def stream_text(text: str) -> AsyncGenerator[str, None]:
    # Simulate streaming by yielding word by word
    for word in text.split(" "):
        yield f"data: {word} \n\n"
        await asyncio.sleep(0.02)
    yield "data: [DONE]\n\n"


async def stream_llm(llm: ChatGroq, prompt: str) -> AsyncGenerator[str, None]:
    # Stream tokens from Groq
    async for chunk in llm.astream(prompt):
        token = chunk.content
        if token:
            yield f"data: {token}\n\n"
    yield "data: [DONE]\n\n"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/session")
def create_session():
    session_id = str(uuid.uuid4())
    get_session(session_id)  # initialise
    return {"session_id": session_id}


@app.get("/mode")
def get_mode(x_session_id: str = Header(...)):
    state = get_session(x_session_id)
    return {"mode": state.mode}


@app.get("/memory")
def get_memory(x_session_id: str = Header(...)):
    state = get_session(x_session_id)
    if state.mode == CHAT and state.chat_memory:
        msgs = state.chat_memory.chat_memory.messages
        summary = state.chat_memory.moving_summary_buffer or ""
    elif state.mode == RAG and state.rag_memory:
        msgs = state.rag_memory.chat_memory.messages
        summary = state.rag_memory.moving_summary_buffer or ""
    elif state.mode == POSTMORTEM and state.pm_memory:
        msgs = state.pm_memory.chat_memory.messages
        summary = state.pm_memory.moving_summary_buffer or ""
    else:
        return {"summary": "", "messages": []}

    return {
        "summary": summary,
        "messages": [
            {"role": "human" if m.type == "human" else "ai", "content": str(m.content)[:200]}
            for m in msgs
        ]
    }


@app.post("/chat")
async def chat(req: ChatRequest, x_session_id: str = Header(...)):
    state = get_session(x_session_id)
    llm   = get_llm()
    msg   = req.message.strip()

    # Initialise chat memory if first message
    if state.chat_memory is None:
        state.chat_memory = chat_mode.build_memory(llm)

    if state.mode == POSTMORTEM:
        # Locked — stream answer from postmortem chat
        history = state.pm_memory.load_memory_variables({})["history"]
        docs    = state.pm_store.as_retriever(search_kwargs={"k": 4}).invoke(msg)
        context = "\n\n".join([doc.page_content for doc in docs])
        from modes.postmortem import prompt as pm_prompt
        filled  = pm_prompt.format(
            report=state.report_str, history=history,
            context=context, input=msg
        )

        async def _stream():
            full = ""
            async for chunk in llm.astream(filled):
                token = str(chunk.content)
                if token:
                    full += token
                    yield f"data: {token}\n\n"
            state.pm_memory.save_context({"input": msg}, {"output": full})
            yield "data: [DONE]\n\n"

        return StreamingResponse(_stream(), media_type="text/event-stream")

    elif state.mode == RAG:
        history = state.rag_memory.load_memory_variables({})["history"]
        docs    = state.rag_store.as_retriever(search_kwargs={"k": 4}).invoke(msg)
        context = "\n\n".join([
            f"[{i+1}] {Path(d.metadata.get('source','unknown')).name}\n{d.page_content.strip()}"
            for i, d in enumerate(docs)
        ])
        from modes.rag import prompt as rag_prompt
        filled  = rag_prompt.format(history=history, context=context, input=msg)

        async def _stream():
            full = ""
            async for chunk in llm.astream(filled):
                token = str(chunk.content)
                if token:
                    full += token
                    yield f"data: {token}\n\n"
            state.rag_memory.save_context({"input": msg}, {"output": full})
            yield "data: [DONE]\n\n"

        return StreamingResponse(_stream(), media_type="text/event-stream")

    else:
        # Chat mode
        history = state.chat_memory.load_memory_variables({})["history"]
        from modes.chat import prompt as chat_prompt
        filled  = chat_prompt.format(history=history, input=msg)

        async def _stream():
            full = ""
            async for chunk in llm.astream(filled):
                token = str(chunk.content)
                if token:
                    full += token
                    yield f"data: {token}\n\n"
            state.chat_memory.save_context({"input": msg}, {"output": full})
            yield "data: [DONE]\n\n"

        return StreamingResponse(_stream(), media_type="text/event-stream")


@app.post("/upload")
async def upload(file: UploadFile = File(...), x_session_id: str = Header(...)):
    state = get_session(x_session_id)
    llm   = get_llm()

    if state.chat_memory is None:
        state.chat_memory = chat_mode.build_memory(llm)

    # Save upload to temp file
    fname    = file.filename or 'upload'
    suffix   = Path(fname).suffix.lower()
    tmp_path = Path(f"/tmp/{uuid.uuid4()}{suffix}")
    log_name = fname
    tmp_path.write_bytes(await file.read())

    kind = classify_input(str(tmp_path))

    # ── Postmortem locked ──────────────────────────────────────────────────
    if state.mode == POSTMORTEM:
        tmp_path.unlink(missing_ok=True)
        return {"status": "locked", "message": "Session locked to current report. Open a new session for a different report."}

    # ── Unsupported file ───────────────────────────────────────────────────
    if kind == "bad_path":
        tmp_path.unlink(missing_ok=True)
        return {"status": "error", "message": f"Unsupported file type: {suffix}"}

    # ── Log file → postmortem pipeline (SSE) ──────────────────────────────
    if kind == "log_file":
        async def _run_postmortem():
            yield "data: {\"event\":\"progress\",\"text\":\"Reading log file...\"}\n\n"
            raw_log = tmp_path.read_text(encoding="utf-8", errors="ignore")
            yield f"data: {{\"event\":\"progress\",\"text\":\"{len(raw_log.splitlines())} lines read\"}}\n\n"

            yield "data: {\"event\":\"progress\",\"text\":\"Building knowledge store...\"}\n\n"
            store, error_counts = pm_ingest.build_store(raw_log, llm)
            yield f"data: {{\"event\":\"progress\",\"text\":\"{store.index.ntotal} vectors ready\"}}\n\n"

            yield "data: {\"event\":\"progress\",\"text\":\"Running postmortem pipeline (parallel nodes)...\"}\n\n"
            result     = pm_graph.run(llm, store, error_counts)
            report_str = pm_report.build_report(result, log_name)

            yield f"data: {{\"event\":\"progress\",\"text\":\"Indexing report...\"}}\n\n"
            pm_store = pm_mode.build_report_store(report_str)

            # Update session state
            state.mode       = POSTMORTEM
            state.pm_store   = pm_store
            state.pm_memory  = pm_mode.build_memory(llm)
            state.report_str = report_str

            tmp_path.unlink(missing_ok=True)

            # Send the full report as final event
            import json
            yield f"data: {{\"event\":\"report\",\"text\":{json.dumps(report_str)}}}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_run_postmortem(), media_type="text/event-stream")

    # ── RAG file ───────────────────────────────────────────────────────────
    if kind == "rag_file":
        if state.mode == CHAT:
            # First doc — switch to RAG
            import doc_ingest
            state.rag_store  = doc_ingest.build_store(str(tmp_path))
            state.rag_memory = rag_mode.build_memory(llm)
            state.mode       = RAG
            tmp_path.unlink(missing_ok=True)
            return {
                "status": "switched",
                "message": f"RAG mode activated. {state.rag_store.index.ntotal} vectors loaded from '{log_name}'."
            }
        elif state.mode == RAG:
            # Additional doc — merge
            import doc_ingest
            doc_ingest.add_to_store(state.rag_store, str(tmp_path))
            tmp_path.unlink(missing_ok=True)
            return {
                "status": "merged",
                "message": f"'{log_name}' added. Store now has {state.rag_store.index.ntotal} vectors."
            }

    return {"status": "error", "message": "Unknown file type."}


@app.delete("/session")
def end_session(x_session_id: str = Header(...)):
    delete_session(x_session_id)
    return {"status": "deleted"}