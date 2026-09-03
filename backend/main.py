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

from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from auth.database import get_db, init_db
from auth.dependencies import get_current_user
from auth.models import SessionFile, SessionMessage, User
from auth.router import limiter, router as auth_router
from config import ALLOWED_ORIGINS, PM_TOP_K, RAG_TOP_K
from core.memory import (
    make_memory,
    save_memory_to_db,
    save_message_to_db,
)
from core.retriever import get_embeddings, retrieve
from graph.state import POSTMORTEM, RAG
from prompts import CHAT_PROMPT, PM_PROMPT, RAG_PROMPT
from rag.ingest import (
    add_to_store,
    build_rag_store,
    hash_file,
    is_duplicate,
    record_file,
)
from router import classify_input
from session import (
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

BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"


# Proxies such as nginx normally buffer responses.
# These headers make sure SSE responses are streamed immediately.
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


# Small helpers

def sse_event(event: str, text: str | None = None) -> str:
    """Create one Server-Sent Event message."""
    data = {"event": event}

    if text is not None:
        data["text"] = text

    return f"data: {json.dumps(data)}\n\n"


def get_memory_for_mode(state: dict):
    """Return the memory belonging to the current session mode."""
    mode = state["mode"]

    if mode == "chat":
        return state.get("chat_memory")

    if mode == RAG:
        return state.get("rag_memory")

    return state.get("pm_memory")


async def get_recent_messages(
    db: AsyncSession,
    session_id: int,
    limit: int = 20,
):
    """
    Get the newest messages in chronological order.

    The database returns newest-first for efficient pagination,
    then we reverse them before sending them to the frontend.
    """
    result = await db.execute(
        select(SessionMessage)
        .where(SessionMessage.session_id == session_id)
        .order_by(desc(SessionMessage.id))
        .limit(limit)
    )

    messages = list(reversed(result.scalars().all()))

    if not messages:
        return [], False

    # Check whether older messages exist.
    older = await db.execute(
        select(SessionMessage.id)
        .where(
            SessionMessage.session_id == session_id,
            SessionMessage.id < messages[0].id,
        )
        .limit(1)
    )

    has_more = older.scalar_one_or_none() is not None

    return messages, has_more


def format_messages(messages):
    """Convert database messages to the API response format."""
    return [
        {
            "id": message.id,
            "role": message.role,
            "content": message.content,
            "mode": message.mode,
        }
        for message in messages
    ]


def format_memory_response(state: dict, memory):
    """Build the response used by /session/memory."""
    mode = state["mode"]

    if not memory:
        return {
            "mode": mode,
            "report": state.get("report_str", ""),
            "summary": "",
            "messages": [],
        }

    return {
        "mode": mode,
        "report": state.get("report_str", ""),
        "summary": memory.moving_summary_buffer or "",
        "messages": [
            {
                "role": "human" if message.type == "human" else "ai",
                "content": str(message.content),
            }
            for message in memory.chat_memory.messages
        ],
    }


# Lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()

    # Load the embedding model once at startup so the first request
    # does not have to pay the model-loading cost.
    logger.info("Preloading embeddings model...")
    await asyncio.get_running_loop().run_in_executor(
        None,
        get_embeddings,
    )
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


app = FastAPI(
    title="OpsIQ",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(auth_router)


# Static pages

@app.get("/")
def home():
    return RedirectResponse(url="/login")


@app.get("/login")
def login_page():
    return FileResponse(FRONTEND_DIR / "login.html")


@app.get("/signup")
def signup_page():
    return FileResponse(FRONTEND_DIR / "signup.html")


@app.get("/app")
def app_page():
    return FileResponse(FRONTEND_DIR / "index.html")


# Session dependency

async def get_active_session(
    x_session_token: str = Header(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Find the user's session and update its last-accessed time.

    get_session() also restores sessions that were removed from memory
    but still exist in the database/disk.
    """
    session = await get_session(
        x_session_token,
        current_user.id,
    )

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found",
        )

    await touch_session(
        x_session_token,
        current_user.id,
        db,
    )

    return session


# Session routes

@app.post("/session")
async def new_session(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new session."""
    session = await create_session(current_user.id, db)
    return {"token": session["token"]}


@app.get("/sessions")
async def get_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return all sessions belonging to the current user."""
    sessions = await list_sessions(current_user.id, db)
    return {"sessions": sessions}


@app.delete("/session")
async def end_session(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    session: dict = Depends(get_active_session),
):
    """Permanently delete the current session."""
    deleted = await delete_session(
        session["token"],
        current_user.id,
        db,
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Session not found",
        )

    return {"status": "deleted"}


@app.get("/session/mode")
async def get_mode(
    session: dict = Depends(get_active_session),
):
    return {"mode": session["state"]["mode"]}


@app.get("/session/memory")
async def get_memory(
    session: dict = Depends(get_active_session),
):
    state = session["state"]
    memory = get_memory_for_mode(state)

    return format_memory_response(state, memory)


@app.get("/session/messages")
async def get_messages(
    before: int = 0,
    limit: int = 20,
    session: dict = Depends(get_active_session),
    db: AsyncSession = Depends(get_db),
):
    """
    Return paginated message history.

    before=0 → newest messages
    before=N → messages older than N
    """
    query = (
        select(SessionMessage)
        .where(SessionMessage.session_id == session["db_id"])
    )

    if before > 0:
        query = query.where(SessionMessage.id < before)

    query = (
        query
        .order_by(desc(SessionMessage.id))
        .limit(limit)
    )

    result = await db.execute(query)
    messages = list(reversed(result.scalars().all()))

    has_more = False

    if messages:
        older = await db.execute(
            select(SessionMessage.id)
            .where(
                SessionMessage.session_id == session["db_id"],
                SessionMessage.id < messages[0].id,
            )
            .limit(1)
        )

        has_more = older.scalar_one_or_none() is not None

    return {
        "messages": format_messages(messages),
        "has_more": has_more,
        "oldest_id": messages[0].id if messages else None,
    }


@app.get("/session/state")
async def get_session_state(
    session: dict = Depends(get_active_session),
    db: AsyncSession = Depends(get_db),
):
    """
    Return everything needed to open a session.

    This replaces three separate frontend requests:
    /session/mode
    /session/memory
    /session/messages
    """
    state = session["state"]

    messages, has_more = await get_recent_messages(
        db,
        session["db_id"],
    )

    return {
        "mode": state["mode"],
        "report": state.get("report_str", ""),
        "messages": format_messages(messages),
        "has_more": has_more,
        "oldest_id": messages[0].id if messages else None,
    }


# Chat

class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=32_000,
    )


@app.post("/chat")
@limiter.limit("20/minute")
@limiter.limit("100/day")
async def chat(
    request: Request,
    req: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    session: dict = Depends(get_active_session),
):
    """
    Stream an AI response using Server-Sent Events.

    The session lock prevents two requests from modifying the same
    conversation memory at the same time.
    """
    msg = req.message.strip()

    async def stream() -> AsyncGenerator[str, None]:
        from core.memory import get_history, save_turn

        async with session["lock"]:
            state = session["state"]

            mode = state["mode"]
            db_id = session["db_id"]
            token = session["token"]
            user_id = session["user_id"]

            # Select the correct model, memory, retriever and prompt.
            if mode == "chat":
                llm = session["chat_llm"]

                if state.get("chat_memory") is None:
                    state["chat_memory"] = make_memory(llm)

                memory = state["chat_memory"]
                history = get_history(memory)

                prompt = CHAT_PROMPT.format(
                    history=history,
                    input=msg,
                )

            elif mode == RAG:
                llm = session["rag_llm"]

                if state.get("rag_memory") is None:
                    state["rag_memory"] = make_memory(llm)

                memory = state["rag_memory"]
                history = get_history(memory)

                context = retrieve(
                    state["rag_store"],
                    msg,
                    RAG_TOP_K,
                )

                prompt = RAG_PROMPT.format(
                    history=history,
                    context=context,
                    input=msg,
                )

            else:
                llm = session["pm_llm"]

                if state.get("pm_memory") is None:
                    state["pm_memory"] = make_memory(llm)

                memory = state["pm_memory"]
                history = get_history(memory)

                context = retrieve(
                    state["pm_store"],
                    msg,
                    PM_TOP_K,
                )

                prompt = PM_PROMPT.format(
                    report=state.get("report_str", ""),
                    history=history,
                    context=context,
                    input=msg,
                )

            # Give the session a name based on its first message.
            if not session.get("named"):
                session["named"] = True

                name = msg[:40]
                if len(msg) > 40:
                    name += "..."

                await update_session_name(
                    token,
                    user_id,
                    name,
                    db,
                )

            full_response = ""
            error_text = None

            # Stream the model response.
            try:
                async for chunk in llm.astream(prompt):
                    text = str(chunk.content)

                    if not text:
                        continue

                    full_response += text

                    # JSON keeps newlines and markdown safe inside SSE.
                    yield sse_event(
                        "token",
                        text,
                    )

            except Exception as error:
                logger.exception(
                    "Chat stream failed token=%s mode=%s",
                    token,
                    mode,
                )

                error_text = f"Response failed: {error}"

            # Save the conversation.
            try:
                if full_response:
                    save_turn(
                        memory,
                        msg,
                        full_response,
                    )

                    await save_message_to_db(
                        db,
                        db_id,
                        "human",
                        msg,
                        mode,
                    )

                    await save_message_to_db(
                        db,
                        db_id,
                        "ai",
                        full_response,
                        mode,
                    )

                    await save_memory_to_db(
                        db,
                        db_id,
                        state.get("chat_memory"),
                        state.get("rag_memory"),
                        state.get("pm_memory"),
                    )

                else:
                    # Save the question even if the model failed.
                    # Don't add an empty AI answer to LangChain memory.
                    await save_message_to_db(
                        db,
                        db_id,
                        "human",
                        msg,
                        mode,
                    )

            except Exception:
                logger.exception(
                    "Failed to persist turn token=%s",
                    token,
                )

            if error_text:
                yield sse_event(
                    "error",
                    error_text,
                )

            # The frontend waits for this event before stopping.
            yield sse_event("done")

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


# Upload

@app.post("/upload")
@limiter.limit("3/minute")
@limiter.limit("20/day")
async def upload(
    request: Request,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    session: dict = Depends(get_active_session),
):
    """
    Upload a document or log file.

    Documents → RAG
    Log files → PostMortem
    """
    state = session["state"]

    token = session["token"]
    user_id = session["user_id"]
    db_id = session["db_id"]

    # A session can contain only one PostMortem incident.
    if state.get("is_locked"):
        return {
            "status": "locked",
            "message": (
                "This session already has a postmortem. "
                "Open a new session to analyse a different incident."
            ),
        }

    from config import MAX_UPLOAD_SIZE_MB

    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()

    temp_path = Path(
        f"/tmp/{uuid.uuid4()}{suffix}"
    )

    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024

    # Check the multipart size first when available.
    if file.size is not None and file.size > max_bytes:
        return {
            "status": "error",
            "message": (
                f"File too large "
                f"({file.size // (1024 * 1024)}MB). "
                f"Maximum is {MAX_UPLOAD_SIZE_MB}MB."
            ),
        }

    # Read in 1 MB chunks instead of loading the entire file into RAM.
    written = 0

    try:
        with temp_path.open("wb") as output:
            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                written += len(chunk)

                if written > max_bytes:
                    raise ValueError("too_large")

                output.write(chunk)

    except ValueError:
        temp_path.unlink(missing_ok=True)

        return {
            "status": "error",
            "message": (
                f"File too large. "
                f"Maximum is {MAX_UPLOAD_SIZE_MB}MB."
            ),
        }

    except Exception as error:
        temp_path.unlink(missing_ok=True)

        logger.exception(
            "Failed to save upload file=%s token=%s",
            filename,
            token,
        )

        return {
            "status": "error",
            "message": f"Could not read file: {error}",
        }

    if written == 0:
        temp_path.unlink(missing_ok=True)

        return {
            "status": "error",
            "message": "File is empty.",
        }

    # Only the file path reaches the classifier.
    # User messages are handled separately by /chat.
    file_hash = hash_file(str(temp_path))
    kind = classify_input(str(temp_path))

    if kind == "bad_path":
        temp_path.unlink(missing_ok=True)

        return {
            "status": "error",
            "message": f"Unsupported file type: {suffix}",
        }

    # Log files belong to PostMortem, not RAG.
    if state["mode"] == RAG and kind == "log_file":
        temp_path.unlink(missing_ok=True)

        async def warn():
            yield sse_event(
                "error",
                "Log files cannot be added in RAG mode. "
                "Open a new session for postmortem analysis.",
            )
            yield sse_event("done")

        return StreamingResponse(
            warn(),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    # Don't process the same file twice in one session.
    if await is_duplicate(db, db_id, file_hash):
        temp_path.unlink(missing_ok=True)

        return {
            "status": "duplicate",
            "message": (
                f"'{filename}' was already loaded into this session."
            ),
        }

    # Log file → PostMortem
    if kind == "log_file":

        async def run_postmortem() -> AsyncGenerator[str, None]:
            try:
                yield sse_event(
                    "progress",
                    f"Processing {filename}...",
                )

                try:
                    async with session["lock"]:
                        from config import POSTMORTEM_TIMEOUT_SECONDS

                        result = await asyncio.wait_for(
                            run_postmortem_async(
                                token,
                                user_id,
                                str(temp_path),
                            ),
                            timeout=POSTMORTEM_TIMEOUT_SECONDS,
                        )

                finally:
                    # Never leave uploaded files in /tmp.
                    temp_path.unlink(missing_ok=True)

                # Save the completed PostMortem.
                await update_session_mode(
                    token,
                    user_id,
                    POSTMORTEM,
                    db,
                    is_locked=True,
                )

                await update_session_name(
                    token,
                    user_id,
                    f"PM: {filename}",
                    db,
                )

                await record_file(
                    db,
                    db_id,
                    filename,
                    file_hash,
                )

                report = result.get("report_str", "")

                await save_report_to_db(
                    token,
                    user_id,
                    report,
                    db,
                )

                await save_memory_to_db(
                    db,
                    db_id,
                    state.get("chat_memory"),
                    state.get("rag_memory"),
                    state.get("pm_memory"),
                )

                yield sse_event(
                    "report",
                    report,
                )

            except Exception as error:
                logger.exception(
                    "Postmortem pipeline failed "
                    "for file=%s token=%s",
                    filename,
                    token,
                )

                error_message = str(error).lower()

                if isinstance(error, asyncio.TimeoutError):
                    text = (
                        "The analysis took too long and was stopped. "
                        "This usually means the log is very large or "
                        "the AI service is slow. Try a smaller file "
                        "in a new session."
                    )

                elif (
                    "rate limit" in error_message
                    or "429" in error_message
                    or "quota" in error_message
                ):
                    text = (
                        "The AI service rate limit was reached. "
                        "This can reset within a minute, or take longer "
                        "if the daily quota was hit. Open a new session "
                        "and try again once it clears."
                    )

                elif (
                    "timeout" in error_message
                    or "timed out" in error_message
                ):
                    text = (
                        "The analysis timed out — this usually means "
                        "the log is very large. Try a smaller file "
                        "in a new session."
                    )

                elif (
                    "connection" in error_message
                    or "network" in error_message
                ):
                    text = (
                        "Could not reach the AI service. "
                        "Check your connection and try again "
                        "in a new session."
                    )

                else:
                    text = (
                        "Analysis failed and could not be completed. "
                        "Please open a new session and try again."
                    )

                yield sse_event(
                    "error",
                    text,
                )

            finally:
                # Always send done as the final SSE event.
                yield sse_event("done")

        return StreamingResponse(
            run_postmortem(),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    # RAG file → Document ingestion
    if kind == "rag_file":
        try:
            async with session["lock"]:
                existing_store = state.get("rag_store")

                if existing_store is None:
                    store = await asyncio.get_running_loop().run_in_executor(
                        None,
                        build_rag_store,
                        str(temp_path),
                        user_id,
                        token,
                        filename,
                    )

                    state["rag_store"] = store
                    state["mode"] = RAG

                else:
                    store = await asyncio.get_running_loop().run_in_executor(
                        None,
                        add_to_store,
                        existing_store,
                        str(temp_path),
                        user_id,
                        token,
                        filename,
                    )

                # When switching from Chat → RAG, preserve the conversation.
                if state.get("rag_memory") is None:
                    rag_memory = make_memory(session["rag_llm"])
                    chat_memory = state.get("chat_memory")

                    if chat_memory:
                        if chat_memory.moving_summary_buffer:
                            rag_memory.moving_summary_buffer = (
                                chat_memory.moving_summary_buffer
                            )

                        for message in chat_memory.chat_memory.messages:
                            rag_memory.chat_memory.add_message(message)

                    state["rag_memory"] = rag_memory

        except ValueError as error:
            # ValueError represents an expected user input problem,
            # such as a scanned PDF with no readable text.
            logger.info(
                "RAG ingestion rejected file=%s token=%s: %s",
                filename,
                token,
                error,
            )

            return {
                "status": "error",
                "message": str(error),
            }

        except Exception as error:
            logger.exception(
                "RAG ingestion failed for file=%s token=%s",
                filename,
                token,
            )

            return {
                "status": "error",
                "message": f"Failed to process file: {error}",
            }

        finally:
            # Always delete the temporary upload.
            temp_path.unlink(missing_ok=True)

        await update_session_mode(
            token,
            user_id,
            RAG,
            db,
        )

        # Name the session after the first uploaded file.
        result = await db.execute(
            select(SessionFile)
            .where(SessionFile.session_id == db_id)
        )

        if not result.scalars().all():
            await update_session_name(
                token,
                user_id,
                filename,
                db,
            )

        await record_file(
            db,
            db_id,
            filename,
            file_hash,
        )

        return {
            "status": "ok",
            "message": (
                f"'{filename}' loaded. "
                f"Store has {store.index.ntotal} vectors."
            ),
        }

    return {
        "status": "error",
        "message": "Unknown file type.",
    }


# Upload configuration

@app.get("/upload/extensions")
def upload_extensions():
    """Return supported file types and the upload size limit."""
    from config import MAX_UPLOAD_SIZE_MB
    from router import supported_extensions

    extensions = sorted(supported_extensions())

    return {
        "extensions": extensions,
        "accept": ",".join(extensions),
        "max_size_mb": MAX_UPLOAD_SIZE_MB,
    }