import logging
from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from auth.models import Base
from config import DATABASE_URL, DB_SCHEMA

logger = logging.getLogger(__name__)

_is_postgres = DATABASE_URL.startswith("postgresql")

_engine_kwargs: dict = {"echo": False, "pool_pre_ping": True}
if _is_postgres and DB_SCHEMA:
    # Ensure every connection uses the app schema (Postgres 15+ public schema fix).
    _engine_kwargs["connect_args"] = {"server_settings": {"search_path": DB_SCHEMA}}

engine = create_async_engine(DATABASE_URL, **_engine_kwargs)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def init_db() -> None:
    async with engine.begin() as conn:
        if _is_postgres and DB_SCHEMA:
            # Regular users cannot CREATE in schema public on Postgres 15+.
            # Create and use a dedicated schema owned by the connected user.
            await conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{DB_SCHEMA}"'))
            await conn.execute(text(f'SET search_path TO "{DB_SCHEMA}", public'))
            logger.info("Using Postgres schema: %s", DB_SCHEMA)
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
