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

from typing import List, Optional

@app.post("/chat")
async def chat_endpoint(
    question: str = Form(None),
    thread_id: str = Form(None),
    files: Optional[List[UploadFile]] = File(None),
    file: Optional[UploadFile] = File(None),
):
    upload_files = []
    if files:
        upload_files.extend(files)
    if file:
        upload_files.append(file)

    if upload_files:
        temp_paths = []
        filenames = []
        try:
            for uf in upload_files:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp:
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
            detail="Either thread_id or PDF file(s) must be provided to start a chat session."
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
