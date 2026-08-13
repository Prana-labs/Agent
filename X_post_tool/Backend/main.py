import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Form

import tempfile

from rag import setup_pipeline
from state import chat, create_session, sessions

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

@app.post("/chat")
async def chat_endpoint(
    question: str = Form(None),
    thread_id: str = Form(None),
    file: UploadFile = File(None),
):
    if file:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp:
            temp.write(await file.read())
            pdf_path = temp.name

        try:
            pipeline = setup_pipeline(pdf_path)
            filename = file.filename or "document"

            if thread_id:
                sessions[thread_id] = {
                    "pipeline": pipeline,
                    "filename": filename,
                }
                tid = thread_id
            else:
                tid = create_session(pipeline, filename)

            return {
                "thread_id": tid,
                "question": None,
                "answer": f"I have successfully loaded and indexed '{filename}'. Ask me anything about its contents!",
            }
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
    elif not thread_id:
        raise HTTPException(
            status_code=400,
            detail="Either thread_id or file must be provided to start a chat session."
        )

    if not question:
        return {
            "thread_id": thread_id,
            "question": None,
            "answer": "Session ready. Ask me anything about the document contents!",
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


@app.get("/")
def home():

    return {
        "message": "PDF RAG API is running"
    }
