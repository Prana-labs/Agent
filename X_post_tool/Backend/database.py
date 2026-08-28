import asyncio
import os
import urllib.parse
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set.")

# Robust URL parsing: strip all query parameters to prevent asyncpg keyword errors,
# and pass SSL configuration explicitly via connect_args.
parsed = urllib.parse.urlparse(DATABASE_URL)
has_ssl = "ssl" in parsed.query or "sslmode" in parsed.query or "neon.tech" in parsed.netloc

clean_url = urllib.parse.urlunparse((
    "postgresql+asyncpg",
    parsed.netloc,
    parsed.path,
    "",
    "",
    ""
))

connect_args = {"ssl": True} if has_ssl else {}

engine = create_async_engine(
    clean_url,
    connect_args=connect_args,
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


async def init_db(max_retries: int = 3, retry_delay: float = 2.0):
    """
    Initializes database tables on startup.
    Includes retry logic to handle serverless database wakeups (e.g. Neon cold-starts).
    """
    import tables  # noqa: F401
    for attempt in range(1, max_retries + 1):
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            print(" Database tables initialized.")
            return
        except Exception as e:
            print(f"⚠️ Database initialization attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                await asyncio.sleep(retry_delay)
            else:
                print("❌ Warning: Database tables could not be verified on startup. Will connect on request.")

