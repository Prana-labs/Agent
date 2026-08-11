from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from fastapi import FastAPI, UploadFile, File, Form

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

from sentence_transformers import CrossEncoder

import tempfile
import os

from langsmith import traceable


app = FastAPI()

load_dotenv()


# =========================================================
# MODELS
# =========================================================

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# =========================================================
# 1. LOAD PDF
# =========================================================

@traceable(name="load_pdf")
def load_pdf(path: str):

    loader = PyPDFLoader(path)

    return loader.load()


# =========================================================
# 2. SPLIT DOCUMENTS
# =========================================================

@traceable(name="split_documents")
def split_documents(
    docs,
    chunk_size=1000,
    chunk_overlap=150
):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return splitter.split_documents(docs)


# =========================================================
# 3. BUILD DENSE + SPARSE RETRIEVERS
# =========================================================

@traceable(name="build_retrievers")
def build_retrievers(splits):

    # Dense
    vectorstore = FAISS.from_documents(
        splits,
        embeddings
    )

    # Sparse
    bm25 = BM25Retriever.from_documents(
        splits
    )

    bm25.k = 20

    return vectorstore, bm25


# =========================================================
# 4. SETUP PIPELINE
# =========================================================

@traceable(name="setup_pipeline")
def setup_pipeline(pdf_path: str):

    docs = load_pdf(pdf_path)

    splits = split_documents(docs)

    vectorstore, bm25 = build_retrievers(splits)

    return vectorstore, bm25


# =========================================================
# 5. DENSE RETRIEVAL
# =========================================================

@traceable(name="dense_retrieval")
def dense_retrieval(vectorstore, question):

    return vectorstore.similarity_search(
        question,
        k=20
    )


# =========================================================
# 6. SPARSE RETRIEVAL
# =========================================================

@traceable(name="sparse_retrieval")
def sparse_retrieval(bm25, question):

    return bm25.invoke(question)


# =========================================================
# 7. HYBRID RETRIEVAL
# =========================================================

@traceable(name="hybrid_retrieval")
def hybrid_retrieval(
    dense_docs,
    sparse_docs
):

    docs = {
        doc.page_content: doc
        for doc in dense_docs + sparse_docs
    }

    return list(docs.values())


# =========================================================
# 8. CROSS ENCODER RERANKING
# =========================================================

@traceable(name="rerank_documents")
def rerank_documents(
    question,
    documents,
    top_k=5
):

    scores = reranker.predict([
        (question, doc.page_content)
        for doc in documents
    ])

    ranked = sorted(
        zip(scores, documents),
        key=lambda x: x[0],
        reverse=True
    )

    return [
        doc
        for _, doc in ranked[:top_k]
    ]


# =========================================================
# 9. COMPLETE RETRIEVAL
# =========================================================

@traceable(name="retrieve")
def retrieve(
    vectorstore,
    bm25,
    question
):

    dense_docs = dense_retrieval(
        vectorstore,
        question
    )

    sparse_docs = sparse_retrieval(
        bm25,
        question
    )

    candidates = hybrid_retrieval(
        dense_docs,
        sparse_docs
    )

    final_docs = rerank_documents(
        question,
        candidates,
        top_k=5
    )

    return final_docs


# =========================================================
# LLM
# =========================================================

prompt = PromptTemplate.from_template(
    """
    Answer the question using the following context.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)

model = ChatOpenAI()

parser = StrOutputParser()

chain = prompt | model | parser


# =========================================================
# FASTAPI
# =========================================================

@app.post("/ask")
async def ask_question(
    file: UploadFile = File(...),
    question: str = Form(...)
):

    # ---------------------------------
    # Save uploaded PDF
    # ---------------------------------

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp:

        temp.write(await file.read())

        pdf_path = temp.name


    try:

        # ---------------------------------
        # Build FAISS + BM25
        # ---------------------------------

        vectorstore, bm25 = setup_pipeline(
            pdf_path
        )


        # ---------------------------------
        # Hybrid retrieval + reranking
        # ---------------------------------

        docs = retrieve(
            vectorstore,
            bm25,
            question
        )


        # ---------------------------------
        # Build context
        # ---------------------------------

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )


        # ---------------------------------
        # LLM
        # ---------------------------------

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