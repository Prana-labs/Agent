import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from contextlib import asynccontextmanager
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

import tempfile

from rag import setup_pipeline
from state import chat, create_session, sessions

from database import get_db, init_db
import crud

load_dotenv()


# =========================================================
# LIFESPAN — DB INIT ON STARTUP
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# PYDANTIC REQUEST/RESPONSE SCHEMAS
# =========================================================

class CreateUserRequest(BaseModel):
    name: str
    email: Optional[str] = None


class UserResponse(BaseModel):
    user_id: str
    name: str
    email: Optional[str]
    plan: Optional[str]
    created_at: str


class MessageResponse(BaseModel):
    role: str
    content: str
    created_at: str


class SessionSummary(BaseModel):
    thread_id: str
    filenames: list
    created_at: str


# =========================================================
# USER ENDPOINTS
# =========================================================

@app.post("/users", response_model=UserResponse)
async def create_user_endpoint(
    body: CreateUserRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new user. Returns user_id to be stored client-side.
    Pass user_id in subsequent /chat requests to link sessions to the user.
    """
    # Check if user with email already exists
    if body.email:
        existing = await crud.get_user_by_email(db, body.email)
        if existing:
            return UserResponse(
                user_id=existing.id,
                name=existing.name,
                email=existing.email,
                plan=existing.plan,
                created_at=existing.created_at.isoformat(),
            )

    user = await crud.create_user(db, name=body.name, email=body.email)
    return UserResponse(
        user_id=user.id,
        name=user.name,
        email=user.email,
        plan=user.plan,
        created_at=user.created_at.isoformat(),
    )


@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user_endpoint(
    user_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Fetch a user's info by their user_id."""
    user = await crud.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return UserResponse(
        user_id=user.id,
        name=user.name,
        email=user.email,
        plan=user.plan,
        created_at=user.created_at.isoformat(),
    )


@app.get("/users/{user_id}/sessions", response_model=List[SessionSummary])
async def list_user_sessions(
    user_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Return all sessions (PDF uploads) for a user, newest first."""
    user = await crud.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    user_sessions = await crud.list_sessions_for_user(db, user_id)
    return [
        SessionSummary(
            thread_id=s.thread_id,
            filenames=s.filenames or [],
            created_at=s.created_at.isoformat(),
        )
        for s in user_sessions
    ]


# =========================================================
# CHAT ENDPOINT (modified to persist to DB)
# =========================================================

@app.post("/chat")
async def chat_endpoint(
    question: str = Form(None),
    thread_id: str = Form(None),
    user_id: str = Form(None),
    files: Optional[List[UploadFile]] = File(None),
    file: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db),
):
    upload_files = []
    if files:
        upload_files.extend(files)
    if file:
        upload_files.append(file)

    # ── PDF upload: create new session ──────────────────
    if upload_files:
        temp_paths = []
        filenames = []
        try:
            for uf in upload_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
                    temp.write(await uf.read())
                    temp_paths.append(temp.name)
                filenames.append(uf.filename or "document.pdf")

            file_tuples = list(zip(temp_paths, filenames))
            pipeline = setup_pipeline(file_tuples)
            filename_str = ", ".join(filenames)

            if thread_id:
                sessions[thread_id] = {
                    "pipeline": pipeline,
                    "filename": filename_str,
                    "filenames": filenames,
                }
                tid = thread_id
            else:
                tid = create_session(pipeline, filename_str)
                if tid in sessions:
                    sessions[tid]["filenames"] = filenames

            # Persist session to PostgreSQL
            db_session = await crud.get_session_by_thread_id(db, tid)
            if not db_session:
                db_session = await crud.create_session_record(
                    db,
                    thread_id=tid,
                    filenames=filenames,
                    user_id=user_id or None,
                )
                # Log each uploaded file
                for fname in filenames:
                    await crud.log_uploaded_file(db, db_session.id, fname)

            count = len(filenames)
            doc_label = "document" if count == 1 else f"{count} documents"
            return {
                "thread_id": tid,
                "question": None,
                "answer": f"I have successfully loaded and indexed {doc_label} ({filename_str}). Ask me anything across your documents!",
                "filenames": filenames,
            }
        finally:
            for p in temp_paths:
                if os.path.exists(p):
                    os.remove(p)

    elif not thread_id:
        raise HTTPException(
            status_code=400,
            detail="Either thread_id or PDF file(s) must be provided to start a chat session.",
        )

    if not question:
        return {
            "thread_id": thread_id,
            "question": None,
            "answer": "Session ready. Ask me anything about the document contents!",
        }

    # ── Chat message: invoke agent + persist messages ──
    try:
        # Persist the human message
        db_session = await crud.get_session_by_thread_id(db, thread_id)
        if db_session:
            await crud.save_message(db, db_session.id, role="human", content=question)

        answer = chat(thread_id, question)

        # Persist the AI response
        if db_session:
            await crud.save_message(db, db_session.id, role="ai", content=answer)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "thread_id": thread_id,
        "question": question,
        "answer": answer,
    }


# =========================================================
# HISTORY & SESSION MANAGEMENT ENDPOINTS
# =========================================================

@app.get("/sessions/{thread_id}/history", response_model=List[MessageResponse])
async def get_session_history(
    thread_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Fetch the full chat history for a session from PostgreSQL.
    Works even after a server restart (FAISS gone, but messages are in DB).
    """
    messages = await crud.get_messages_for_session(db, thread_id)
    return [
        MessageResponse(
            role=m.role,
            content=m.content,
            created_at=m.created_at.isoformat(),
        )
        for m in messages
    ]


@app.delete("/sessions/{thread_id}")
async def delete_session_endpoint(
    thread_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a session from both in-memory FAISS store and PostgreSQL."""
    # Remove from in-memory sessions (FAISS pipeline)
    if thread_id in sessions:
        del sessions[thread_id]

    # Remove from DB (cascades to messages + uploaded_files)
    deleted = await crud.delete_session_record(db, thread_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found in database.")

    return {"message": f"Session {thread_id} deleted successfully."}


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/")
def home():
    return {"message": "PDF RAG API is running"}
