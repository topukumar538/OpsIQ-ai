from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

import sys
import os

# Make sure backend/ is on the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth.models import Base
from config import DATABASE_URL

# Alembic needs a sync driver — asyncpg is async-only and won't work here.
# Replace asyncpg with psycopg2 just for Alembic migrations.
sync_url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql+psycopg2://")

# Alembic config object
config = context.config
config.set_main_option("sqlalchemy.url", sync_url)

# Set up logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# This is what tells Alembic which tables to track
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_schemas=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()