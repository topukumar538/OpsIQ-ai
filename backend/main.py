# Location: backend/main.py
import asyncio
import json
import logging
import uuid
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

warnings.filterwarnings("ignore", category=DeprecationWarning)

from fastapi import FastAPI, UploadFile, File, Header, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, RedirectResponse
from pydantic import BaseModel, Field
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy.ext.asyncio import AsyncSession

from auth.database import init_db, get_db
from auth.dependencies import get_current_user
from auth.models import User, SessionFile
from auth.router import router as auth_router, limiter
from config import RAG_TOP_K, PM_TOP_K, ALLOWED_ORIGINS
from core.memory import save_message_to_db, save_memory_to_db, make_memory
from core.retriever import get_embeddings
from graph.state import POSTMORTEM, RAG
from rag.ingest import hash_file, is_duplicate, record_file, build_rag_store, add_to_store
from router import classify_input
from session import (
    _sessions,
    create_session,
    delete_session,
    get_session,
    list_sessions,
    run_postmortem_async,
    save_report_to_db,
    start_cleanup_task,
    touch_session,
    update_session_mode,
    update_session_name,
)

logger = logging.getLogger(__name__)

_BACKEND_DIR  = Path(__file__).resolve().parent
_FRONTEND_DIR = _BACKEND_DIR.parent / "frontend"


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    # Preload the embeddings model at startup so the first upload doesn't pay
    # the loading cost. get_embeddings() is lru_cached, so this warms the
    # cache for every subsequent request.
    logger.info("Preloading embeddings model...")
    await asyncio.get_running_loop().run_in_executor(None, get_embeddings)
    logger.info("Embeddings model ready.")

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

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(auth_router)


# ── Static pages ──────────────────────────────────────────────────────────────

@app.get("/")
def home(): return RedirectResponse(url="/login")

@app.get("/login")
def login_page(): return FileResponse(_FRONTEND_DIR / "login.html")

@app.get("/signup")
def signup_page(): return FileResponse(_FRONTEND_DIR / "signup.html")

@app.get("/app")
def app_page(): return FileResponse(_FRONTEND_DIR / "index.html")


# ── Session dependency ────────────────────────────────────────────────────────

async def get_active_session(
    x_session_token: str          = Header(...),
    current_user   : User         = Depends(get_current_user),
    db             : AsyncSession = Depends(get_db),
):
    """
    Resolve and validate the active session on every request.

    - Checks the token exists and belongs to current_user (ownership enforced)
    - Restores from DB + disk if the session was evicted from the memory cache
    - Raises 404 if not found or owned by another user
    - Records access (throttled to one DB write per minute per session)
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
    """Create a new session in the DB and return its token."""
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
    current_user: User         = Depends(get_current_user),
    db          : AsyncSession = Depends(get_db),
    session     : dict         = Depends(get_active_session),
):
    """
    Permanently delete a session — removes the DB row, memory cache, and
    FAISS files. This is the only operation that deletes FAISS from disk;
    TTL eviction drops the memory copy only, so the session can be restored.
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
        return {
            "mode"    : mode,
            "report"  : state.get("report_str", ""),
            "summary" : "",
            "messages": [],
        }
    return {
        "mode"    : mode,
        "report"  : state.get("report_str", ""),
        "summary" : memory.moving_summary_buffer or "",
        "messages": [
            {
                "role"   : "human" if m.type == "human" else "ai",
                "content": str(m.content),
            }
            for m in memory.chat_memory.messages
        ],
    }


@app.get("/session/messages")
async def get_messages(
    before : int          = 0,     # message id to paginate from (0 = latest)
    limit  : int          = 20,
    session: dict         = Depends(get_active_session),
    db     : AsyncSession = Depends(get_db),
):
    """
    Paginated message history — backs the "Load older messages" button.
    Separate from /session/memory, which is for LLM context restoration.

    - before=0  → latest `limit` messages
    - before=N  → `limit` messages older than message id N
    """
    from auth.models import SessionMessage
    from sqlalchemy import select, desc

    db_id = session["db_id"]

    query = select(SessionMessage).where(SessionMessage.session_id == db_id)
    if before > 0:
        query = query.where(SessionMessage.id < before)
    query = query.order_by(desc(SessionMessage.id)).limit(limit)

    result   = await db.execute(query)
    messages = list(reversed(result.scalars().all()))

    has_more = False
    if messages:
        older = await db.execute(
            select(SessionMessage.id)
            .where(
                SessionMessage.session_id == db_id,
                SessionMessage.id < messages[0].id,
            )
            .limit(1)
        )
        has_more = older.scalar_one_or_none() is not None

    return {
        "messages": [
            {"id": m.id, "role": m.role, "content": m.content, "mode": m.mode}
            for m in messages
        ],
        "has_more" : has_more,
        "oldest_id": messages[0].id if messages else None,
    }


@app.get("/session/state")
async def get_session_state(
    session: dict         = Depends(get_active_session),
    db     : AsyncSession = Depends(get_db),
):
    """
    Everything the frontend needs to open a session, in one request.

    switchSession() used to call /session/mode, /session/memory and
    /session/messages in sequence — three round trips, three auth lookups
    and three session resolutions for one user action.
    """
    from auth.models import SessionMessage
    from sqlalchemy import select, desc

    state = session["state"]
    db_id = session["db_id"]

    result = await db.execute(
        select(SessionMessage)
        .where(SessionMessage.session_id == db_id)
        .order_by(desc(SessionMessage.id))
        .limit(20)
    )
    messages = list(reversed(result.scalars().all()))

    has_more = False
    if messages:
        older = await db.execute(
            select(SessionMessage.id)
            .where(
                SessionMessage.session_id == db_id,
                SessionMessage.id < messages[0].id,
            )
            .limit(1)
        )
        has_more = older.scalar_one_or_none() is not None

    return {
        "mode"    : state["mode"],
        "report"  : state.get("report_str", ""),
        "messages": [
            {"id": m.id, "role": m.role, "content": m.content, "mode": m.mode}
            for m in messages
        ],
        "has_more" : has_more,
        "oldest_id": messages[0].id if messages else None,
    }


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=32_000)


@app.post("/chat")
@limiter.limit("20/minute")
@limiter.limit("100/day")
# Per-minute stops a runaway loop; per-day stops one user draining the shared
# Groq quota and leaving the app broken for everyone else. Counted by IP, so
# users behind the same network share a budget — an accepted trade for not
# needing per-user token accounting.
async def chat(
    request     : Request,      # must be first — slowapi reads the IP from it
    req         : ChatRequest,
    current_user: User         = Depends(get_current_user),
    db          : AsyncSession = Depends(get_db),
    session     : dict         = Depends(get_active_session),
):
    msg = req.message.strip()

    async def _stream() -> AsyncGenerator[str, None]:
        from core.memory import get_history, save_turn
        from core.retriever import retrieve
        from prompts import CHAT_PROMPT as chat_prompt
        from prompts import RAG_PROMPT as rag_prompt
        from prompts import PM_PROMPT as pm_prompt

        async with session["lock"]:
            state = session["state"]
            mode  = state["mode"]
            db_id = session["db_id"]
            token = session["token"]
            uid   = session["user_id"]

            if mode == "chat":
                llm = session["chat_llm"]
                if state.get("chat_memory") is None:
                    state["chat_memory"] = make_memory(llm)
                memory  = state["chat_memory"]
                history = get_history(memory)
                filled  = chat_prompt.format(history=history, input=msg)
            elif mode == RAG:
                llm = session["rag_llm"]
                if state.get("rag_memory") is None:
                    state["rag_memory"] = make_memory(llm)
                memory  = state["rag_memory"]
                history = get_history(memory)
                context = retrieve(state["rag_store"], msg, RAG_TOP_K)
                filled  = rag_prompt.format(history=history, context=context, input=msg)
            else:
                llm = session["pm_llm"]
                if state.get("pm_memory") is None:
                    state["pm_memory"] = make_memory(llm)
                memory  = state["pm_memory"]
                history = get_history(memory)
                context = retrieve(state["pm_store"], msg, PM_TOP_K)
                filled  = pm_prompt.format(
                    report  = state.get("report_str", ""),
                    history = history,
                    context = context,
                    input   = msg,
                )

            # Auto-name the session from the first human message
            if not session.get("named"):
                session["named"] = True
                name = msg[:40] + ("..." if len(msg) > 40 else "")
                await update_session_name(token, uid, name, db)

            full       = ""
            error_text = None

            try:
                async for chunk in llm.astream(filled):
                    token_text = str(chunk.content)
                    if token_text:
                        full += token_text
                        # JSON-encoded so newlines survive. A raw
                        # "data: {text}\n\n" frame splits apart at every blank
                        # line in the model's markdown, which is why the
                        # frontend used to discard the stream entirely.
                        yield f"data: {json.dumps({'event': 'token', 'text': token_text})}\n\n"

            except Exception as e:
                logger.exception("Chat stream failed token=%s mode=%s", token, mode)
                error_text = f"Response failed: {e}"

            # Persist whatever we got — this runs on the error path too, so a
            # partial answer the user already saw isn't lost from history.
            if full:
                try:
                    save_turn(memory, msg, full)
                    await save_message_to_db(db, db_id, "human", msg, mode)
                    await save_message_to_db(db, db_id, "ai", full, mode)
                    await save_memory_to_db(
                        db, db_id,
                        state.get("chat_memory"),
                        state.get("rag_memory"),
                        state.get("pm_memory"),
                    )
                except Exception:
                    logger.exception("Failed to persist turn token=%s", token)

            if error_text:
                yield f"data: {json.dumps({'event': 'error', 'text': error_text})}\n\n"

            # Always last — the frontend reader stops on this.
            yield f"data: {json.dumps({'event': 'done'})}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


# ── Upload ────────────────────────────────────────────────────────────────────

@app.post("/upload")
@limiter.limit("3/minute")
@limiter.limit("20/day")
# Each upload runs embedding plus five LLM calls — by far the most expensive
# endpoint. Twenty a day is well beyond any genuine use.
async def upload(
    request     : Request,      # must be first — slowapi reads the IP from it
    file        : UploadFile   = File(...),
    current_user: User         = Depends(get_current_user),
    db          : AsyncSession = Depends(get_db),
    session     : dict         = Depends(get_active_session),
):
    state = session["state"]
    token = session["token"]
    uid   = session["user_id"]
    db_id = session["db_id"]

    # One incident per session, deliberately. A postmortem report has a single
    # root cause, timeline, and remediation — merging two unrelated incidents
    # into one report produces incoherent analysis.
    if state.get("is_locked"):
        return {
            "status" : "locked",
            "message": "This session already has a postmortem. Open a new session to analyse a different incident.",
        }

    from config import MAX_UPLOAD_SIZE_MB

    fname     = file.filename or "upload"
    suffix    = Path(fname).suffix.lower()
    tmp_path  = Path(f"/tmp/{uuid.uuid4()}{suffix}")
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024

    # Cheap pre-check from the multipart headers. A crafted request can lie
    # about this, so it's an optimisation rather than the real guard — but it
    # rejects the obvious cases without reading a byte of the body.
    if file.size is not None and file.size > max_bytes:
        return {
            "status" : "error",
            "message": f"File too large ({file.size // (1024*1024)}MB). Maximum is {MAX_UPLOAD_SIZE_MB}MB.",
        }

    # Stream to disk 1MB at a time, aborting as soon as the running total
    # crosses the limit. The previous version did `await file.read()` — the
    # entire body into RAM — and only checked the size afterwards, so the
    # guard could never prevent the OOM it existed to prevent.
    written = 0
    try:
        with tmp_path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise ValueError("too_large")
                out.write(chunk)
    except ValueError:
        tmp_path.unlink(missing_ok=True)
        return {
            "status" : "error",
            "message": f"File too large. Maximum is {MAX_UPLOAD_SIZE_MB}MB.",
        }
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        logger.exception("Failed to save upload file=%s token=%s", fname, token)
        return {"status": "error", "message": f"Could not read file: {e}"}

    if written == 0:
        tmp_path.unlink(missing_ok=True)
        return {"status": "error", "message": "File is empty."}

    # classify_input() only ever sees real temp file paths here — never raw
    # user message strings. Messages go through /chat; files go through /upload.
    # That separation means the classifier only checks the extension, rather
    # than applying message-vs-path heuristics that caused false positives.
    file_hash = hash_file(str(tmp_path))
    kind      = classify_input(str(tmp_path))

    if kind == "bad_path":
        tmp_path.unlink(missing_ok=True)
        return {"status": "error", "message": f"Unsupported file type: {suffix}"}

    if state["mode"] == RAG and kind == "log_file":
        tmp_path.unlink(missing_ok=True)

        async def _warn():
            yield f"data: {json.dumps({'event': 'error', 'text': 'Log files cannot be added in RAG mode. Open a new session for postmortem analysis.'})}\n\n"
            yield f"data: {json.dumps({'event': 'done'})}\n\n"

        return StreamingResponse(_warn(), media_type="text/event-stream")

    # ── Duplicate check ───────────────────────────────────────────────────────
    if await is_duplicate(db, db_id, file_hash):
        tmp_path.unlink(missing_ok=True)
        return {
            "status" : "duplicate",
            "message": f"'{fname}' was already loaded into this session.",
        }

    # ── Log file → postmortem pipeline ────────────────────────────────────────
    if kind == "log_file":

        async def _run() -> AsyncGenerator[str, None]:
            try:
                yield f"data: {json.dumps({'event': 'progress', 'text': f'Processing {fname}...'})}\n\n"

                try:
                    async with session["lock"]:
                        result = await run_postmortem_async(token, uid, str(tmp_path))
                finally:
                    # Always clean up the temp file, success or failure —
                    # never leave uploads sitting in /tmp.
                    tmp_path.unlink(missing_ok=True)

                # Pipeline succeeded — persist everything to the DB
                await update_session_mode(token, uid, POSTMORTEM, db, is_locked=True)
                await update_session_name(token, uid, f"PM: {fname}", db)
                await record_file(db, db_id, fname, file_hash)
                await save_report_to_db(token, uid, result.get("report_str", ""), db)
                await save_memory_to_db(
                    db, db_id,
                    state.get("chat_memory"),
                    state.get("rag_memory"),
                    state.get("pm_memory"),
                )

                yield f"data: {json.dumps({'event': 'report', 'text': result.get('report_str', '')})}\n\n"

            except Exception as e:
                # Raw exception text is meaningless to a user — a Groq 429
                # arrives as a wall of JSON. The full traceback still goes to
                # the logs; the user gets something they can act on.
                #
                # No specific wait time is promised: Groq has both a per-minute
                # and a per-day quota, and the error doesn't reliably say which
                # one fired, so "wait a minute" would often be wrong.
                logger.exception("Postmortem pipeline failed for file=%s token=%s", fname, token)

                err = str(e).lower()
                if "rate limit" in err or "429" in err or "quota" in err:
                    text = ("The AI service rate limit was reached. This can reset "
                            "within a minute, or take longer if the daily quota was "
                            "hit. Open a new session and try again once it clears.")
                elif "timeout" in err or "timed out" in err:
                    text = ("The analysis timed out — this usually means the log is "
                            "very large. Try a smaller file in a new session.")
                elif "connection" in err or "network" in err:
                    text = ("Could not reach the AI service. Check your connection "
                            "and try again in a new session.")
                else:
                    text = ("Analysis failed and could not be completed. Please open "
                            "a new session and try again.")

                yield f"data: {json.dumps({'event': 'error', 'text': text})}\n\n"

            finally:
                # The done event is ALWAYS last, success or failure. The
                # frontend SSE reader depends on it to stop looping.
                yield f"data: {json.dumps({'event': 'done'})}\n\n"

        return StreamingResponse(_run(), media_type="text/event-stream")

    # ── RAG file → document ingestion ─────────────────────────────────────────
    if kind == "rag_file":
        try:
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

                # Seed RAG memory with the chat history so switching modes
                # mid-conversation doesn't lose context. Both the summary and
                # the raw messages are copied — the summary alone is empty for
                # short conversations that haven't triggered summarisation yet.
                if state.get("rag_memory") is None:
                    rag_memory  = make_memory(session["rag_llm"])
                    chat_memory = state.get("chat_memory")
                    if chat_memory:
                        if chat_memory.moving_summary_buffer:
                            rag_memory.moving_summary_buffer = chat_memory.moving_summary_buffer
                        for m in chat_memory.chat_memory.messages:
                            rag_memory.chat_memory.add_message(m)
                    state["rag_memory"] = rag_memory

        except Exception as e:
            logger.exception("RAG ingestion failed for file=%s token=%s", fname, token)
            return {"status": "error", "message": f"Failed to process file: {e}"}
        finally:
            tmp_path.unlink(missing_ok=True)   # always runs, success or failure

        await update_session_mode(token, uid, RAG, db)

        # Auto-name from the first uploaded filename if no files yet
        from sqlalchemy import select as sa_select
        existing = await db.execute(
            sa_select(SessionFile).where(SessionFile.session_id == db_id)
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
    """
    Snapshot of in-memory sessions with idle times. Debug aid only.

    There is no admin role, so this is gated on DEBUG rather than on the user.
    404 rather than 403 so the endpoint's existence isn't advertised.
    """
    from config import DEBUG
    if not DEBUG:
        raise HTTPException(status_code=404, detail="Not found")

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
    Accepted extensions and the upload size limit.

    The frontend reads this so extension filtering and size checking stay in
    sync with the backend rather than being hardcoded in two places.
    """
    from router import supported_extensions
    from config import MAX_UPLOAD_SIZE_MB
    exts = sorted(supported_extensions())
    return {
        "extensions" : exts,
        "accept"     : ",".join(exts),
        "max_size_mb": MAX_UPLOAD_SIZE_MB,
    }