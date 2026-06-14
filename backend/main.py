# Location: backend/main.py
import asyncio
import hashlib
import json
import logging
import uuid
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

warnings.filterwarnings("ignore", category=DeprecationWarning)

from fastapi import FastAPI, UploadFile, File, Header, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, RedirectResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from auth.database import init_db, get_db
from auth.dependencies import get_current_user
from auth.models import User
from auth.router import router as auth_router
from config import FAISS_STORE_DIR, RAG_TOP_K, PM_TOP_K
from core.memory import save_message_to_db, save_memory_to_db
from graph.state import POSTMORTEM, RAG
from rag.ingest import hash_file, is_duplicate, record_file, build_rag_store, add_to_store
from router import classify_input
from session import (
    _sessions,
    create_session,
    delete_session,
    get_session,
    list_sessions,
    run_graph_async,
    start_cleanup_task,
    touch_session,
    update_session_faiss_path,
    update_session_mode,
    update_session_name,
)

logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    cleanup_task = asyncio.create_task(start_cleanup_task())
    logger.info("OpsIQ started.")
    yield
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    logger.info("OpsIQ shut down cleanly.")


app = FastAPI(title="OpsIQ", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(auth_router)


# ── Static pages ──────────────────────────────────────────────────────────────

@app.get("/")
def home(): return RedirectResponse(url="/login")

@app.get("/login")
def login_page(): return FileResponse("../frontend/login.html")

@app.get("/signup")
def signup_page(): return FileResponse("../frontend/signup.html")

@app.get("/app")
def app_page(): return FileResponse("../frontend/index.html")


# ── Session dependency ────────────────────────────────────────────────────────

async def get_active_session(
    x_session_token: str  = Header(...),
    current_user   : User = Depends(get_current_user),
    db             : AsyncSession = Depends(get_db),
):
    """
    Dependency that resolves + validates the active session on every request.

    - Checks token exists and belongs to current_user (ownership enforced)
    - Restores from DB + disk if evicted from memory cache
    - Raises 404 if not found, 403 if wrong user
    - Updates last_accessed in DB
    """
    session = await get_session(x_session_token, current_user.id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    await touch_session(x_session_token, current_user.id, db)
    return session


# ── Session routes ────────────────────────────────────────────────────────────

@app.post("/session")
async def new_session(
    current_user: User         = Depends(get_current_user),
    db          : AsyncSession = Depends(get_db),
):
    """Create a new session in DB and return its token."""
    session = await create_session(current_user.id, db)
    return {"token": session["token"]}


@app.get("/sessions")
async def get_sessions(
    current_user: User         = Depends(get_current_user),
    db          : AsyncSession = Depends(get_db),
):
    """Return all sessions for the current user — populates the sidebar."""
    sessions = await list_sessions(current_user.id, db)
    return {"sessions": sessions}


@app.delete("/session")
async def end_session(
    current_user   : User         = Depends(get_current_user),
    db             : AsyncSession = Depends(get_db),
    session        : dict         = Depends(get_active_session),
):
    """
    Permanently delete a session — removes DB row, memory cache, FAISS disk.
    This is the only operation that deletes FAISS files.
    TTL eviction only drops from memory; DB + disk stay for reconnection.
    """
    deleted = await delete_session(session["token"], current_user.id, db)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted"}


@app.get("/session/mode")
async def get_mode(session: dict = Depends(get_active_session)):
    return {"mode": session["state"]["mode"]}


@app.get("/session/memory")
async def get_memory(session: dict = Depends(get_active_session)):
    state  = session["state"]
    mode   = state["mode"]
    memory = (
        state.get("chat_memory") if mode == "chat" else
        state.get("rag_memory")  if mode == RAG    else
        state.get("pm_memory")
    )
    if not memory:
        return {"summary": "", "messages": []}
    return {
        "summary" : memory.moving_summary_buffer or "",
        "messages": [
            {
                "role"   : "human" if m.type == "human" else "ai",
                "content": str(m.content)[:200],
            }
            for m in memory.chat_memory.messages
        ],
    }


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat(
    req         : ChatRequest,
    current_user: User         = Depends(get_current_user),
    db          : AsyncSession = Depends(get_db),
    session     : dict         = Depends(get_active_session),
):
    msg = req.message.strip()

    async def _stream() -> AsyncGenerator[str, None]:
        from core.memory import get_history, save_turn
        from core.retriever import retrieve
        from graph.nodes.chat import prompt as chat_prompt
        from graph.nodes.rag import prompt as rag_prompt
        from graph.nodes.postmortem import prompt as pm_prompt

        async with session["lock"]:
            state  = session["state"]
            mode   = state["mode"]
            db_id  = session["db_id"]
            token  = session["token"]
            uid    = session["user_id"]

            # Pick the LLM instance for this mode.
            # Each has a different temperature tuned for its purpose:
            # chat=0.7 (natural), rag=0.3 (accurate), pm=0.1 (deterministic)
            if mode == "chat":
                llm     = session["chat_llm"]
                memory  = state["chat_memory"]
                history = get_history(memory)
                filled  = chat_prompt.format(history=history, input=msg)
            elif mode == RAG:
                llm     = session["rag_llm"]
                memory  = state["rag_memory"]
                history = get_history(memory)
                context = retrieve(state["rag_store"], msg, RAG_TOP_K)
                filled  = rag_prompt.format(history=history, context=context, input=msg)
            else:
                llm     = session["pm_llm"]
                memory  = state["pm_memory"]
                history = get_history(memory)
                context = retrieve(state["pm_store"], msg, PM_TOP_K)
                filled  = pm_prompt.format(
                    report=state.get("report_str", ""),
                    history=history,
                    context=context,
                    input=msg,
                )

            # Auto-name session from first human message
            if not any(m.type == "human" for m in memory.chat_memory.messages):
                name = msg[:40] + ("..." if len(msg) > 40 else "")
                await update_session_name(token, uid, name, db)

            full = ""
            async for chunk in llm.astream(filled):
                token_text = str(chunk.content)
                if token_text:
                    full += token_text
                    yield f"data: {token_text}\n\n"

            # Save turn to LangChain memory
            save_turn(memory, msg, full)

            # Persist messages + memory summary to DB
            await save_message_to_db(db, db_id, "human", msg, mode)
            await save_message_to_db(db, db_id, "ai", full, mode)
            await save_memory_to_db(
                db, db_id,
                state.get("chat_memory"),
                state.get("rag_memory"),
                state.get("pm_memory"),
            )

            yield "data: [DONE]\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


# ── Upload ────────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload(
    file        : UploadFile   = File(...),
    current_user: User         = Depends(get_current_user),
    db          : AsyncSession = Depends(get_db),
    session     : dict         = Depends(get_active_session),
):
    state   = session["state"]
    token   = session["token"]
    uid     = session["user_id"]
    db_id   = session["db_id"]

    if state.get("is_locked"):
        return {"status": "locked", "message": "Session locked to current postmortem. Open a new session."}

    fname     = file.filename or "upload"
    suffix    = Path(fname).suffix.lower()
    tmp_path  = Path(f"/tmp/{uuid.uuid4()}{suffix}")
    raw_bytes = await file.read()
    tmp_path.write_bytes(raw_bytes)

    # classify_input() only ever sees real temp file paths here — never raw
    # user message strings. Messages go through /chat; files go through /upload.
    # This separation means the classifier only needs to check extension,
    # not apply message-vs-path heuristics that caused false positives.
    file_hash = hash_file(str(tmp_path))
    kind      = classify_input(str(tmp_path))

    if kind == "bad_path":
        tmp_path.unlink(missing_ok=True)
        return {"status": "error", "message": f"Unsupported file type: {suffix}"}

    if state["mode"] == RAG and kind == "log_file":
        tmp_path.unlink(missing_ok=True)
        return {
            "status" : "error",
            "message": "Log files cannot be added in RAG mode. Open a new session for postmortem analysis.",
        }

    # ── Duplicate check ───────────────────────────────────────────────────────
    if await is_duplicate(db, db_id, file_hash):
        tmp_path.unlink(missing_ok=True)
        return {
            "status" : "duplicate",
            "message": f"'{fname}' was already loaded into this session.",
        }

    # ── Log file → postmortem pipeline ───────────────────────────────────────
    if kind == "log_file":
        async def _run():
            yield f"data: {json.dumps({'event':'progress','text':f'Processing {fname}...'})}\n\n"
            try:
                async with session["lock"]:
                    result = await run_graph_async(token, uid, file_path=str(tmp_path))
            finally:
                tmp_path.unlink(missing_ok=True)

            # Sync mode + lock back to DB
            await update_session_mode(token, uid, POSTMORTEM, db, is_locked=True)
            await update_session_faiss_path(
                token, uid,
                str(Path(FAISS_STORE_DIR) / str(uid) / token / "pm"),
                db,
            )
            # Auto-name from log filename
            await update_session_name(token, uid, f"PM: {fname}", db)
            # Record file in DB
            await record_file(db, db_id, fname, file_hash)

            yield f"data: {json.dumps({'event':'report','text':result.get('report_str','')})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_run(), media_type="text/event-stream")

    # ── RAG file → document ingestion ─────────────────────────────────────────
    if kind == "rag_file":
        async with session["lock"]:
            existing_store = state.get("rag_store")
            if existing_store is None:
                store = await asyncio.get_running_loop().run_in_executor(
                    None, build_rag_store, str(tmp_path), uid, token,
                )
                state["rag_store"] = store
                state["mode"]      = RAG
            else:
                store = await asyncio.get_running_loop().run_in_executor(
                    None, add_to_store, existing_store, str(tmp_path), uid, token,
                )

        tmp_path.unlink(missing_ok=True)

        # Sync to DB
        await update_session_mode(token, uid, RAG, db)
        await update_session_faiss_path(
            token, uid,
            str(Path(FAISS_STORE_DIR) / str(uid) / token / "rag"),
            db,
        )
        # Auto-name from first uploaded filename if no files yet
        from sqlalchemy import select as sa_select
        from auth.models import SessionFile as SF
        existing = await db.execute(
            sa_select(SF).where(SF.session_id == db_id)
        )
        if not existing.scalars().all():
            await update_session_name(token, uid, fname, db)

        await record_file(db, db_id, fname, file_hash)

        return {
            "status" : "ok",
            "message": f"'{fname}' loaded. Store has {store.index.ntotal} vectors.",
        }

    return {"status": "error", "message": "Unknown file type."}


# ── Admin ─────────────────────────────────────────────────────────────────────

@app.get("/admin/sessions")
async def admin_sessions(current_user: User = Depends(get_current_user)):
    """Snapshot of all in-memory sessions with idle times."""
    import time as _time
    now  = _time.time()
    rows = []
    for tok, s in _sessions.items():
        idle = int(now - s["last_accessed"])
        rows.append({
            "token"       : tok[:12] + "...",
            "user_id"     : s["user_id"],
            "mode"        : s["state"].get("mode"),
            "idle_seconds": idle,
            "idle_human"  : _fmt(idle),
        })
    rows.sort(key=lambda r: r["idle_seconds"], reverse=True)
    return {"total": len(rows), "sessions": rows}


def _fmt(s: int) -> str:
    if s < 60:   return f"{s}s"
    if s < 3600: return f"{s//60}m {s%60}s"
    return f"{s//3600}h {(s%3600)//60}m"


@app.get("/upload/extensions")
def upload_extensions():
    """
    Return accepted file extensions.
    Frontend uses this to build the file input accept attribute dynamically
    so it always stays in sync with the backend — no hardcoding in two places.
    """
    from router import supported_extensions
    exts = sorted(supported_extensions())
    return {
        "extensions": exts,
        "accept":     ",".join(exts),   # ready to drop into <input accept="...">
    }