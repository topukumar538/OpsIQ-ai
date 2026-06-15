# Location: backend/core/memory.py
"""
LangChain memory helpers — creation, save, restore.

Each session has THREE memory objects (one per mode):
    chat_memory  — plain conversation
    rag_memory   — document Q&A
    pm_memory    — postmortem analysis

They are kept separate so switching modes doesn't pollute each other's
context. The LLM sees only the memory for the active mode.

Persistence strategy:
    - moving_summary_buffer  → saved to session_memory table (full summary)
    - recent raw messages    → saved to session_messages table (last 20)

On restore:
    - summary   → loaded into moving_summary_buffer directly
    - messages  → replayed into chat_memory.messages in order
    - Result: AI sees full context as if session never ended
"""
import logging
from typing import Optional

from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import HumanMessage, AIMessage
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from auth.models import SessionMemory, SessionMessage
from config import MAX_TOKEN_LIMIT

logger = logging.getLogger(__name__)

# How many recent raw messages to load back into memory on session restore.
# Older messages are covered by the summary. 20 is enough for immediate
# context without blowing up the prompt on long sessions.
RESTORE_MESSAGE_LIMIT = 20


def make_memory(llm) -> ConversationSummaryBufferMemory:
    """Create a fresh LangChain memory object."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=MAX_TOKEN_LIMIT,
            return_messages=True,
        )


def get_history(memory: Optional[ConversationSummaryBufferMemory]) -> str:
    """
    Format memory into a single history string for prompt injection.
    Returns empty string if memory is None or empty.
    """
    if not memory:
        return ""
    parts = []
    if memory.moving_summary_buffer:
        parts.append(f"[Summary of earlier conversation]\n{memory.moving_summary_buffer}")
    for msg in memory.chat_memory.messages:
        role    = "User" if isinstance(msg, HumanMessage) else "Assistant"
        parts.append(f"{role}: {msg.content}")
    return "\n".join(parts)


def save_turn(memory: ConversationSummaryBufferMemory, human: str, ai: str) -> None:
    """Add a completed turn to LangChain memory (triggers summarisation if needed)."""
    memory.save_context({"input": human}, {"output": ai})


# ── Postgres persistence ───────────────────────────────────────────────────────

async def save_message_to_db(
    db         : AsyncSession,
    session_id : int,
    role       : str,
    content    : str,
    mode       : str,
) -> None:
    """Persist a single message turn to session_messages."""
    db.add(SessionMessage(
        session_id=session_id,
        role=role,
        content=content,
        mode=mode,
    ))
    await db.commit()


async def save_memory_to_db(
    db         : AsyncSession,
    session_id : int,
    chat_memory: Optional[ConversationSummaryBufferMemory],
    rag_memory : Optional[ConversationSummaryBufferMemory],
    pm_memory  : Optional[ConversationSummaryBufferMemory],
) -> None:
    """
    Upsert the moving_summary_buffer for all three memory objects.

    Why upsert: we don't want to INSERT a new row every message — one row
    per session that gets updated in place is cleaner and avoids table bloat.
    """
    def _summary(mem):
        return mem.moving_summary_buffer if mem else ""

    result = await db.execute(
        select(SessionMemory).where(SessionMemory.session_id == session_id)
    )
    row = result.scalar_one_or_none()

    if row:
        row.chat_summary = _summary(chat_memory)
        row.rag_summary  = _summary(rag_memory)
        row.pm_summary   = _summary(pm_memory)
    else:
        db.add(SessionMemory(
            session_id   = session_id,
            chat_summary = _summary(chat_memory),
            rag_summary  = _summary(rag_memory),
            pm_summary   = _summary(pm_memory),
        ))
    await db.commit()


async def restore_memory_from_db(
    db         : AsyncSession,
    session_id : int,
    chat_llm,
    rag_llm,
    pm_llm,
) -> dict:
    """
    Restore all three memory objects from Postgres.

    Strategy:
        1. Load full summaries from session_memory → set as moving_summary_buffer
        2. Load last 20 messages from session_messages → replay into chat_memory
        3. Return dict with chat_memory, rag_memory, pm_memory

    The AI then sees:
        [Full summary of everything before the last 20 messages]
        [Last 20 raw messages]
        [New message]

    This matches exactly how Claude and ChatGPT handle long conversations —
    compress the old, keep the recent raw.
    """
    chat_mem = make_memory(chat_llm)
    rag_mem  = make_memory(rag_llm)
    pm_mem   = make_memory(pm_llm)

    # ── Restore summaries ─────────────────────────────────────────────────────
    result = await db.execute(
        select(SessionMemory).where(SessionMemory.session_id == session_id)
    )
    mem_row = result.scalar_one_or_none()

    if mem_row:
        chat_mem.moving_summary_buffer = mem_row.chat_summary or ""
        rag_mem.moving_summary_buffer  = mem_row.rag_summary  or ""
        pm_mem.moving_summary_buffer   = mem_row.pm_summary   or ""
        logger.debug(
            "Restored memory summaries for session %d — "
            "chat=%d chars, rag=%d chars, pm=%d chars",
            session_id,
            len(mem_row.chat_summary or ""),
            len(mem_row.rag_summary  or ""),
            len(mem_row.pm_summary   or ""),
        )

    # ── Restore last N raw messages ───────────────────────────────────────────
    result = await db.execute(
        select(SessionMessage)
        .where(SessionMessage.session_id == session_id)
        .order_by(desc(SessionMessage.created_at))
        .limit(RESTORE_MESSAGE_LIMIT)
    )
    # Reverse so oldest-first order (we queried newest-first for the LIMIT)
    recent_messages = list(reversed(result.scalars().all()))

    for msg in recent_messages:
        # Replay each message into the correct mode's memory object
        target_mem = {"chat": chat_mem, "rag": rag_mem, "postmortem": pm_mem}.get(
            msg.mode, chat_mem
        )
        if msg.role == "human":
            target_mem.chat_memory.add_message(HumanMessage(content=msg.content))
        else:
            target_mem.chat_memory.add_message(AIMessage(content=msg.content))

    logger.info(
        "Restored %d recent messages for session %d",
        len(recent_messages), session_id,
    )

    return {
        "chat_memory": chat_mem,
        "rag_memory":  rag_mem,
        "pm_memory":   pm_mem,
    }