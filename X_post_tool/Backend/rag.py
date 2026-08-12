import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

import tempfile
import os

from langsmith import traceable

load_dotenv()


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


@traceable(name="load_pdf")
def load_pdf(path: str):

    loader = PyPDFLoader(path)

    return loader.load()


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

@traceable(name="setup_pipeline")
def setup_pipeline(pdf_path: str):

    docs = load_pdf(pdf_path)

    splits = split_documents(docs)

    vectorstore, bm25 = build_retrievers(splits)

    return vectorstore, bm25

@traceable(name="dense_retrieval")
def dense_retrieval(vectorstore, question):

    return vectorstore.similarity_search(
        question,
        k=20
    )

@traceable(name="sparse_retrieval")
def sparse_retrieval(bm25, question):

    return bm25.invoke(question)

@traceable(name="reciprocal_rank_fusion")
def reciprocal_rank_fusion(
    dense_docs,
    sparse_docs,
    c=60,
    top_k=5
):
    """
    Applies Reciprocal Rank Fusion (RRF) on dense and sparse retrieval results.
    RRF score formula: RRF_Score(d) = sum(1 / (c + rank_i(d)))
    """
    rrf_scores = {}
    doc_map = {}

    # Dense scoring
    for rank, doc in enumerate(dense_docs):
        content = doc.page_content
        doc_map[content] = doc
        rrf_scores[content] = rrf_scores.get(content, 0.0) + 1.0 / (c + rank + 1)

    # Sparse scoring
    for rank, doc in enumerate(sparse_docs):
        content = doc.page_content
        doc_map[content] = doc
        rrf_scores[content] = rrf_scores.get(content, 0.0) + 1.0 / (c + rank + 1)

    # Sort documents by RRF score descending
    sorted_contents = sorted(
        rrf_scores.keys(),
        key=lambda content: rrf_scores[content],
        reverse=True
    )

    return [doc_map[content] for content in sorted_contents[:top_k]]

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

    # Perform Reciprocal Rank Fusion instead of CrossEncoder reranking
    final_docs = reciprocal_rank_fusion(
        dense_docs,
        sparse_docs,
        top_k=5
    )

    return final_docs
