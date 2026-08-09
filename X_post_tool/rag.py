# rag.py

import os
import re

from langsmith import traceable

from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from dotenv import load_dotenv
load_dotenv()


# ============================================================
# MODELS
# ============================================================

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

model = ChatOpenAI(
    temperature=0
)


# ============================================================
# CACHE
# ============================================================

answer_cache = {}


def make_cache_key(document_id, question):

    question = re.sub(
        r"\s+",
        " ",
        question.strip().lower()
    )

    return f"{document_id}:{question}"


def get_cached_answer(document_id, question):

    key = make_cache_key(document_id, question)

    return answer_cache.get(key)


def cache_answer(document_id, question, answer):

    key = make_cache_key(document_id, question)

    answer_cache[key] = answer


# ============================================================
# 1. LOAD PDF
# ============================================================

@traceable(name="load_pdf")
def load_pdf(path: str):

    loader = PyPDFLoader(path)

    return loader.load()


# ============================================================
# 2. SPLIT DOCUMENTS
# ============================================================

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


# ============================================================
# 3. BUILD FAISS  (dense retrieval)
# ============================================================

@traceable(name="build_faiss")
def build_faiss(splits):

    return FAISS.from_documents(splits, embeddings)


# ============================================================
# 4. BUILD BM25  (sparse / lexical retrieval)
# ============================================================

@traceable(name="build_bm25")
def build_bm25(splits):

    tokenized_corpus = [
        doc.page_content.lower().split()
        for doc in splits
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    return bm25


# ============================================================
# 5. HYBRID RETRIEVAL  (dense + sparse, deduplicated)
# ============================================================

@traceable(name="hybrid_retrieval")
def hybrid_retrieval(
    faiss_store,
    bm25,
    splits,
    question,
    k=10
):

    # --- Dense ---
    dense_docs = faiss_store.similarity_search(question, k=k)

    # --- Sparse BM25 ---
    tokenized_query = question.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k]

    sparse_docs = [splits[i] for i in top_indices]

    # --- Deduplicate ---
    seen = set()
    combined = []

    for doc in dense_docs + sparse_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            combined.append(doc)

    return combined


# ============================================================
# 6. CROSS-ENCODER RERANKING
# ============================================================

@traceable(name="rerank_documents")
def rerank_documents(
    question,
    documents,
    top_k=5
):

    pairs = [
        (question, doc.page_content)
        for doc in documents
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, documents),
        key=lambda x: x[0],
        reverse=True
    )

    return [doc for _, doc in ranked[:top_k]]


# ============================================================
# 7. RETRIEVE
# ============================================================

@traceable(name="retrieve")
def retrieve(
    faiss_store,
    bm25,
    splits,
    question
):

    candidates = hybrid_retrieval(
        faiss_store,
        bm25,
        splits,
        question
    )

    return rerank_documents(question, candidates, top_k=5)


# ============================================================
# 8. LLM
# ============================================================

prompt = PromptTemplate.from_template(
    """
    You are a factual document question-answering system.

    Answer the question using ONLY the provided context.

    Do not invent information.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)

parser = StrOutputParser()

chain = prompt | model | parser


@traceable(name="generate_answer")
def generate_answer(question, documents):

    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    return chain.invoke({
        "context": context,
        "question": question
    })


# ============================================================
# 9. FULL ONE-SHOT PIPELINE
# ============================================================

@traceable(name="run_rag")
def run_rag(pdf_path: str, question: str) -> str:
    """
    Load a PDF, build FAISS + BM25 in memory,
    do hybrid retrieval + reranking, and return the LLM answer.
    No disk writes (except the temp PDF handled by the caller).
    """

    docs = load_pdf(pdf_path)

    splits = split_documents(docs)

    faiss_store = build_faiss(splits)

    bm25 = build_bm25(splits)

    documents = retrieve(faiss_store, bm25, splits, question)

    return generate_answer(question, documents)