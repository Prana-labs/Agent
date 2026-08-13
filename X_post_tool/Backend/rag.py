import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dataclasses import dataclass
from typing import List

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from langsmith import traceable

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


@traceable(name="load_pdf")
def load_pdf(path: str) -> List[Document]:
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
    vectorstore = FAISS.from_documents(
        splits,
        embeddings
    )

    bm25 = BM25Retriever.from_documents(
        splits
    )
    bm25.k = 20

    return vectorstore, bm25


@traceable(name="dense_retrieval")
def dense_retrieval(vectorstore, question: str):
    return vectorstore.similarity_search(
        question,
        k=20
    )


@traceable(name="sparse_retrieval")
def sparse_retrieval(bm25, question: str):
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

    for rank, doc in enumerate(dense_docs):
        content = doc.page_content
        doc_map[content] = doc
        rrf_scores[content] = rrf_scores.get(content, 0.0) + 1.0 / (c + rank + 1)

    for rank, doc in enumerate(sparse_docs):
        content = doc.page_content
        doc_map[content] = doc
        rrf_scores[content] = rrf_scores.get(content, 0.0) + 1.0 / (c + rank + 1)

    sorted_contents = sorted(
        rrf_scores.keys(),
        key=lambda content: rrf_scores[content],
        reverse=True
    )

    return [doc_map[content] for content in sorted_contents[:top_k]]


@dataclass
class RAGPipeline:
    """
    Full RAG pipeline: indexing (load/split/build) + hybrid retrieval (dense/sparse/RRF).
    """
    vectorstore: FAISS
    bm25: BM25Retriever

    @traceable(name="retrieve")
    def retrieve(self, question: str, top_k: int = 5) -> List[Document]:
        dense_docs = dense_retrieval(self.vectorstore, question)
        sparse_docs = sparse_retrieval(self.bm25, question)
        return reciprocal_rank_fusion(
            dense_docs,
            sparse_docs,
            top_k=top_k
        )

    def get_context(self, question: str, top_k: int = 5) -> str:
        docs = self.retrieve(question, top_k=top_k)
        return "\n\n".join(doc.page_content for doc in docs)


@traceable(name="setup_pipeline")
def setup_pipeline(pdf_path: str) -> RAGPipeline:
    """
    Indexing pipeline: load PDF -> split -> build retrievers -> return RAGPipeline.
    Query-time retrieval uses dense_retrieval + sparse_retrieval + reciprocal_rank_fusion.
    """
    docs = load_pdf(pdf_path)
    splits = split_documents(docs)
    vectorstore, bm25 = build_retrievers(splits)
    return RAGPipeline(vectorstore=vectorstore, bm25=bm25)


def retrieve(vectorstore, bm25, question: str, top_k: int = 5):
    """Backward-compatible wrapper around the full retrieval pipeline."""
    pipeline = RAGPipeline(vectorstore=vectorstore, bm25=bm25)
    return pipeline.retrieve(question, top_k=top_k)
