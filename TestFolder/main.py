from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Form

import tempfile
import os

from rag import setup_pipeline, retrieve
from LLMs import chain

app = FastAPI()

load_dotenv()
# FASTAPI
@app.post("/ask")
async def ask_question(
    file: UploadFile = File(...),
    question: str = Form(...)
):
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
        # Hybrid retrieval + reranking
        docs = retrieve(vectorstore, bm25, question)
        # Build context
        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )
        # LLM
        result = chain.invoke({
            "context": context,
            "question": question
        })
        return {
            "filename": file.filename,
            "question": question,
            "answer": result
        }


    finally:

        os.remove(pdf_path)


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "message": "PDF RAG API is running"
    }