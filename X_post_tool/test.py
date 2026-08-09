import logging
import traceback
import os
import uuid

from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from rag import (
    run_rag,
    get_cached_answer,
    cache_answer
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


# ============================================================
# UPLOAD + ASK  (single endpoint)
# ============================================================

@app.post("/upload")
async def upload_and_ask(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    """
    Upload a PDF and ask a question about it in one shot.

    Form fields:
      - file     : the PDF file
      - question : the question you want answered
    """

    document_id = str(uuid.uuid4())

    pdf_path = f"temp_{document_id}.pdf"

    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    try:

        # Return cached answer if available
        cached = get_cached_answer(document_id, question)
        if cached:
            return {
                "document_id": document_id,
                "filename": file.filename,
                "question": question,
                "answer": cached,
                "cached": True
            }

        # Run full RAG pipeline (in-memory FAISS + BM25)
        answer = run_rag(pdf_path, question)

        cache_answer(document_id, question, answer)

        return {
            "document_id": document_id,
            "filename": file.filename,
            "question": question,
            "answer": answer,
            "cached": False
        }

    except Exception as e:
        logger.error("Pipeline failed:\n%s", traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {str(e)}"
        )

    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Hybrid RAG API is running"
    }