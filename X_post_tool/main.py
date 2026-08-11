from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Form

import tempfile
import os

from rag import setup_pipeline
from state import create_session, chat

from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

# FASTAPI
@app.post("/chat")
async def chat_endpoint(
    question: str = Form(None),
    thread_id: str = Form(None),
    file: UploadFile = File(None)
):
    if file:
        # Save uploaded PDF
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp:
            temp.write(await file.read())
            pdf_path = temp.name

        try:
            # Build FAISS + BM25
            vectorstore, bm25 = setup_pipeline(pdf_path)
            # Create session
            thread_id = create_session(vectorstore, bm25, file.filename)
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
    elif not thread_id:
        raise HTTPException(
            status_code=400,
            detail="Either thread_id or file must be provided to start a chat session."
        )

    # If initializing session without a question, return successfully without invoking graph
    if not question:
        filename = file.filename if file else "document"
        return {
            "thread_id": thread_id,
            "question": None,
            "answer": f"I have successfully loaded and indexed '{filename}'. Ask me anything about its contents!"
        }

    try:
        answer = chat(thread_id, question)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "thread_id": thread_id,
        "question": question,
        "answer": answer
    }


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "message": "PDF RAG API is running"
    }