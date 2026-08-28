import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set.")

import urllib.parse

# Ensure the URL uses the asyncpg driver and clean libpq-specific parameters
parsed = urllib.parse.urlparse(DATABASE_URL)
has_ssl = "ssl" in parsed.query or "sslmode" in parsed.query

DATABASE_URL = urllib.parse.urlunparse((
    "postgresql+asyncpg",
    parsed.netloc,
    parsed.path,
    "",
    "ssl=require" if has_ssl else "",
    ""
))

engine = create_async_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    echo=False,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    class_=AsyncSession,
)

Base = declarative_base()


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    import tables  # noqa: F401
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Database tables initialized.")
