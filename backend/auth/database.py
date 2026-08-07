import logging
from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from auth.models import Base
from config import DATABASE_URL, DB_SCHEMA, DB_SSL

logger = logging.getLogger(__name__)

_is_postgres = DATABASE_URL.startswith("postgresql")

# SSL is set explicitly via DB_SSL, never inferred from the hostname.
# The previous version turned SSL on whenever the host wasn't localhost,
# assuming "not local => cloud database". That broke under Docker, where
# the host is a container name like "db" but the server has SSL disabled —
# asyncpg fails with "rejected SSL upgrade" instead of falling back.
#   DB_SSL=false -> local Postgres, Docker Compose
#   DB_SSL=true  -> hosted providers (Neon, Supabase, Render, RDS)
_connect_args: dict = {"ssl": True} if DB_SSL else {}

# Route every connection to our named schema instead of public, so the app
# never depends on the session's default search_path.
if _is_postgres and DB_SCHEMA:
    _connect_args["server_settings"] = {"search_path": DB_SCHEMA}

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,   # verifies a pooled connection is alive before reuse
    connect_args=_connect_args,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,   # keeps ORM objects usable after commit()
    class_=AsyncSession,
)


async def init_db() -> None:
    """Create the schema and all tables. Called once from the app lifespan."""
    async with engine.begin() as conn:
        if _is_postgres and DB_SCHEMA:
            await conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{DB_SCHEMA}"'))
            await conn.execute(text(f'SET search_path TO "{DB_SCHEMA}", public'))
            logger.info("Using Postgres schema: %s", DB_SCHEMA)
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields a session and closes it after the request."""
    async with AsyncSessionLocal() as session:
        yield session