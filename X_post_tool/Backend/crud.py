from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from tables import User, Session, Message, UploadedFile


# =========================================================
# USER CRUD
# =========================================================

async def create_user(db: AsyncSession, name: str, email: Optional[str] = None) -> User:
    """Create a new user record."""
    user = User(name=name, email=email)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def get_user_by_id(db: AsyncSession, user_id: str) -> Optional[User]:
    """Fetch a user by their UUID."""
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    """Fetch a user by email (for auth lookup)."""
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


# =========================================================
# SESSION CRUD
# =========================================================

async def create_session_record(
    db: AsyncSession,
    thread_id: str,
    filenames: List[str],
    user_id: Optional[str] = None,
) -> Session:
    """
    Create a session record for a PDF upload.
    Links to a user if user_id is provided (optional for now, required after auth).
    """
    session = Session(thread_id=thread_id, filenames=filenames, user_id=user_id)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def get_session_by_thread_id(db: AsyncSession, thread_id: str) -> Optional[Session]:
    """Fetch a session by its LangGraph thread_id."""
    result = await db.execute(
        select(Session)
        .where(Session.thread_id == thread_id)
        .options(selectinload(Session.messages))
    )
    return result.scalar_one_or_none()


async def list_sessions_for_user(
    db: AsyncSession,
    user_id: str,
    limit: int = 20
) -> List[Session]:
    """Return the most recent sessions for a user."""
    result = await db.execute(
        select(Session)
        .where(Session.user_id == user_id)
        .order_by(Session.created_at.desc())
        .limit(limit)
    )
    return result.scalars().all()


async def delete_session_record(db: AsyncSession, thread_id: str) -> bool:
    """Delete a session (and its messages via cascade) by thread_id."""
    session = await get_session_by_thread_id(db, thread_id)
    if session:
        await db.delete(session)
        await db.commit()
        return True
    return False


# =========================================================
# MESSAGE CRUD  (chat history)
# =========================================================

async def save_message(
    db: AsyncSession,
    session_id: str,
    role: str,   # "human" or "ai"
    content: str,
) -> Message:
    """Persist a single chat message (human or ai) to the database."""
    message = Message(session_id=session_id, role=role, content=content)
    db.add(message)
    await db.commit()
    await db.refresh(message)
    return message


async def get_messages_for_session(
    db: AsyncSession,
    thread_id: str
) -> List[Message]:
    """
    Fetch the full chat history for a session, ordered by time.
    Joins sessions → messages so we only need the thread_id externally.
    """
    result = await db.execute(
        select(Message)
        .join(Session, Message.session_id == Session.id)
        .where(Session.thread_id == thread_id)
        .order_by(Message.created_at.asc())
    )
    return result.scalars().all()


# =========================================================
# UPLOADED FILES
# =========================================================

async def log_uploaded_file(
    db: AsyncSession,
    session_id: str,
    filename: str,
    file_size_bytes: Optional[int] = None,
) -> UploadedFile:
    """Log a PDF file that was uploaded into a session."""
    uploaded = UploadedFile(
        session_id=session_id,
        filename=filename,
        file_size_bytes=file_size_bytes
    )
    db.add(uploaded)
    await db.commit()
    await db.refresh(uploaded)
    return uploaded
