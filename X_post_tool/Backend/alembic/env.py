import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context
from dotenv import load_dotenv

# Make sure Backend/ is on the path so we can import tables and database
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

# Alembic config object
config = context.config

# Set the DB URL from environment (overrides alembic.ini)
database_url = os.getenv("DATABASE_URL", "")
# Alembic needs the sync driver (psycopg2), not asyncpg
database_url = (
    database_url
    .replace("postgresql+asyncpg://", "postgresql+psycopg2://")
    .replace("postgres://", "postgresql+psycopg2://")
)
config.set_main_option("sqlalchemy.url", database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import all models so Alembic can detect schema changes
from database import Base
import tables  # noqa: F401 — registers all ORM tables on Base.metadata

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
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
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
