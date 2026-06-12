# Location: backend/main.py
import warnings
import uuid
import json
from pathlib import Path
from typing import AsyncGenerator

warnings.filterwarnings("ignore", category=DeprecationWarning)

from fastapi import FastAPI, UploadFile, File, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, RedirectResponse
from pydantic import BaseModel

from auth.dependencies import get_current_user
from auth.router import router as auth_router
from auth.database import init_db
from auth.models import User
from session import create_session, get_session, delete_session, run_graph
from router import classify_input
from graph.state import POSTMORTEM, RAG

app = FastAPI(title="OpsIQ")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    await init_db()


# ── Auth routes ───────────────────────────────────────────────────────────────

app.include_router(auth_router)


# ── Static pages ──────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return RedirectResponse(url="/login")

@app.get("/login")
def login_page():
    return FileResponse("../frontend/login.html")

@app.get("/signup")
def signup_page():
    return FileResponse("../frontend/signup.html")

@app.get("/app")
def app_page():
    return FileResponse("../frontend/index.html")


# ── Session ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


@app.post("/session")
def new_session(current_user: User = Depends(get_current_user)):
    session_id = str(uuid.uuid4())
    create_session(session_id)
    return {"session_id": session_id}


@app.delete("/session")
def end_session(
    x_session_id: str  = Header(...),
    current_user: User = Depends(get_current_user),
):
    delete_session(x_session_id)
    return {"status": "deleted"}


@app.get("/mode")
def get_mode(
    x_session_id: str  = Header(...),
    current_user: User = Depends(get_current_user),
):
    session = get_session(x_session_id)
    return {"mode": session["state"]["mode"]}


@app.get("/memory")
def get_memory(
    x_session_id: str  = Header(...),
    current_user: User = Depends(get_current_user),
):
    session = get_session(x_session_id)
    state   = session["state"]
    mode    = state["mode"]

    memory = (
        state["chat_memory"] if mode == "chat" else
        state["rag_memory"]  if mode == "rag"  else
        state["pm_memory"]
    )

    if not memory:
        return {"summary": "", "messages": []}

    msgs = memory.chat_memory.messages
    return {
        "summary" : memory.moving_summary_buffer or "",
        "messages": [
            {"role": "human" if m.type == "human" else "ai", "content": str(m.content)[:200]}
            for m in msgs
        ],
    }


# ── Chat ──────────────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(
    req          : ChatRequest,
    x_session_id : str  = Header(...),
    current_user : User = Depends(get_current_user),
):
    session = get_session(x_session_id)
    state   = session["state"]
    llm     = session["llm"]
    msg     = req.message.strip()

    async def _stream() -> AsyncGenerator[str, None]:
        from core.memory import get_history, save_turn
        from core.retriever import retrieve
        from config import RAG_TOP_K, PM_TOP_K
        from graph.nodes.chat import prompt as chat_prompt
        from graph.nodes.rag import prompt as rag_prompt
        from graph.nodes.postmortem import prompt as pm_prompt

        mode = state["mode"]

        if mode == "chat":
            memory  = state["chat_memory"]
            history = get_history(memory)
            filled  = chat_prompt.format(history=history, input=msg)
        elif mode == RAG:
            memory  = state["rag_memory"]
            history = get_history(memory)
            context = retrieve(state["rag_store"], msg, RAG_TOP_K)
            filled  = rag_prompt.format(history=history, context=context, input=msg)
        else:  # POSTMORTEM
            memory  = state["pm_memory"]
            history = get_history(memory)
            context = retrieve(state["pm_store"], msg, PM_TOP_K)
            filled  = pm_prompt.format(
                report=state["report_str"], history=history, context=context, input=msg,
            )

        full = ""
        async for chunk in llm.astream(filled):
            token = str(chunk.content)
            if token:
                full += token
                yield f"data: {token}\n\n"
        save_turn(memory, msg, full)
        yield "data: [DONE]\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


# ── Upload ────────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload(
    file         : UploadFile = File(...),
    x_session_id : str        = Header(...),
    current_user : User       = Depends(get_current_user),
):
    session = get_session(x_session_id)
    state   = session["state"]

    if state["mode"] == POSTMORTEM:
        return {"status": "locked", "message": "Session locked to current report. Open a new session."}

    fname    = file.filename or "upload"
    suffix   = Path(fname).suffix.lower()
    tmp_path = Path(f"/tmp/{uuid.uuid4()}{suffix}")
    tmp_path.write_bytes(await file.read())

    kind = classify_input(str(tmp_path))

    if kind == "bad_path":
        tmp_path.unlink(missing_ok=True)
        return {"status": "error", "message": f"Unsupported file type: {suffix}"}

    if state["mode"] == RAG and kind == "log_file":
        tmp_path.unlink(missing_ok=True)
        return {"status": "locked", "message": "Log files cannot be added in RAG mode. Open a new session to run a PostMortem analysis."}

    if kind == "log_file":
        async def _run():
            yield f"data: {json.dumps({'event':'progress','text':f'Processing {fname}...'})}\n\n"
            result = run_graph(x_session_id, file_path=str(tmp_path))
            tmp_path.unlink(missing_ok=True)
            yield f"data: {json.dumps({'event':'report','text':result['report_str']})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_run(), media_type="text/event-stream")

    if kind in {"rag_file", "log_file"}:
        result  = run_graph(x_session_id, file_path=str(tmp_path))
        tmp_path.unlink(missing_ok=True)
        warning = session["state"].get("rag_warning", "")
        if warning:
            session["state"]["rag_warning"] = ""
            return {"status": "locked", "message": warning}
        store = session["state"]["rag_store"]
        return {"status": "ok", "message": f"'{fname}' loaded. Store has {store.index.ntotal} vectors."}

    return {"status": "error", "message": "Unknown file type."}