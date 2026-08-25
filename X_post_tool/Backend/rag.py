import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dataclasses import dataclass, field
from typing import List, Dict, Union, Tuple, Optional

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


# =========================================================
# DOCUMENT LOADING & CHUNKING WITH SOURCE METADATA
# =========================================================

@traceable(name="load_pdf")
def load_pdf(path: str, filename: Optional[str] = None) -> List[Document]:
    """
    Loads a single PDF and tags every page with its filename source.
    """
    loader = PyPDFLoader(path)
    docs = loader.load()
    doc_name = filename or os.path.basename(path)
    for doc in docs:
        doc.metadata["source_name"] = doc_name
        if "page" in doc.metadata:
            # 1-indexed page for human readability
            doc.metadata["page_number"] = doc.metadata["page"] + 1
        else:
            doc.metadata["page_number"] = 1
    return docs


@traceable(name="load_pdfs")
def load_pdfs(files: Union[List[Tuple[str, str]], List[str], str]) -> List[Document]:
    """
    Loads multiple PDFs, preserving source metadata for each document.
    Accepts:
      - List of (file_path, file_name) tuples
      - List of file_path strings
      - Single file_path string
    """
    if isinstance(files, str):
        return load_pdf(files)

    docs = []
    for item in files:
        if isinstance(item, tuple):
            path, fname = item
            docs.extend(load_pdf(path, fname))
        else:
            docs.extend(load_pdf(item))
    return docs


@traceable(name="split_documents")
def split_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


# =========================================================
# RETRIEVAL UTILITIES: DENSE, SPARSE & RECIPROCAL RANK FUSION
# =========================================================

@traceable(name="dense_retrieval")
def dense_retrieval(vectorstore: FAISS, question: str, k: int = 5) -> List[Document]:
    return vectorstore.similarity_search(question, k=k)


@traceable(name="sparse_retrieval")
def sparse_retrieval(bm25: BM25Retriever, question: str) -> List[Document]:
    return bm25.invoke(question)


@traceable(name="reciprocal_rank_fusion")
def reciprocal_rank_fusion(
    dense_docs: List[Document],
    sparse_docs: List[Document],
    c: int = 60,
    top_k: int = 5
) -> List[Document]:
    """
    Applies Reciprocal Rank Fusion (RRF) on dense and sparse retrieval results.
    RRF score formula: RRF_Score(d) = sum(1 / (c + rank_i(d)))
    """
    rrf_scores = {}
    doc_map = {}

    for rank, doc in enumerate(dense_docs):
        # Key on source_name + content to distinguish identical chunks across different docs
        key = f"{doc.metadata.get('source_name', '')}_{doc.page_content}"
        doc_map[key] = doc
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (c + rank + 1)

    for rank, doc in enumerate(sparse_docs):
        key = f"{doc.metadata.get('source_name', '')}_{doc.page_content}"
        doc_map[key] = doc
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (c + rank + 1)

    sorted_keys = sorted(
        rrf_scores.keys(),
        key=lambda k: rrf_scores[k],
        reverse=True
    )

    return [doc_map[k] for k in sorted_keys[:top_k]]


# =========================================================
# 1. STANDARD RAG PIPELINE (Dense Vector Only)
# =========================================================

@dataclass
class StandardRAGPipeline:
    """
    Standard RAG pipeline:
    Chunking -> Embedding -> FAISS Vector Store -> Dense Similarity Retrieval.
    """
    vectorstore: FAISS
    doc_name: str = "document"

    @traceable(name="standard_retrieve")
    def retrieve(self, question: str, top_k: int = 5) -> List[Document]:
        return dense_retrieval(self.vectorstore, question, k=top_k)


# =========================================================
# 2. HYBRID RAG PIPELINE (Dense + Sparse + RRF)
# =========================================================

@dataclass
class HybridRAGPipeline:
    """
    Hybrid RAG pipeline:
    Chunking -> FAISS VectorStore (Dense) + BM25 (Sparse) -> Reciprocal Rank Fusion (RRF).
    """
    vectorstore: FAISS
    bm25: BM25Retriever
    doc_name: str = "document"

    @traceable(name="hybrid_retrieve")
    def retrieve(self, question: str, top_k: int = 5) -> List[Document]:
        dense_docs = dense_retrieval(self.vectorstore, question, k=max(top_k * 2, 10))
        sparse_docs = sparse_retrieval(self.bm25, question)
        return reciprocal_rank_fusion(dense_docs, sparse_docs, top_k=top_k)


# =========================================================
# 3. MULTI-DOCUMENT COLLECTION (Per-PDF Balanced Retrieval)
# =========================================================

@dataclass
class DocumentCollection:
    """
    Manages multiple PDF documents with individual pipelines per PDF.
    
    If multiple PDFs are loaded:
      - Runs retrieval on each PDF pipeline independently to ensure fair representation (no document starvation).
      - Merges the retrieved chunks and tags each chunk with [Document: <name> | Page: <num>].
    """
    doc_names: List[str] = field(default_factory=list)
    standard_pipelines: Dict[str, StandardRAGPipeline] = field(default_factory=dict)
    hybrid_pipelines: Dict[str, HybridRAGPipeline] = field(default_factory=dict)
    
    # Unified global pipelines (for broad fallback search)
    global_standard: Optional[StandardRAGPipeline] = None
    global_hybrid: Optional[HybridRAGPipeline] = None

    @property
    def document_count(self) -> int:
        return len(self.doc_names)

    @traceable(name="collection_retrieve")
    def retrieve(
        self,
        question: str,
        mode: str = "hybrid",
        top_k_per_doc: int = 3,
        global_top_k: int = 8
    ) -> List[Document]:
        """
        Multi-PDF Aware Retrieval:
        - If 1 document: retrieves top_k from the single document pipeline.
        - If >1 documents: retrieves top_k_per_doc from EACH document pipeline separately to guarantee balanced context.
        """
        mode = mode.lower()
        use_hybrid = mode in ["hybrid", "hybrid_rag"]

        if self.document_count <= 1:
            # Single document retrieval
            doc_name = self.doc_names[0] if self.doc_names else "document"
            pipeline = self.hybrid_pipelines.get(doc_name) if use_hybrid else self.standard_pipelines.get(doc_name)
            if pipeline:
                return pipeline.retrieve(question, top_k=global_top_k)
            elif self.global_hybrid and use_hybrid:
                return self.global_hybrid.retrieve(question, top_k=global_top_k)
            elif self.global_standard:
                return self.global_standard.retrieve(question, top_k=global_top_k)
            return []

        # Multiple documents: Per-Document Balanced Retrieval
        merged_docs: List[Document] = []
        for doc_name in self.doc_names:
            pipeline = (
                self.hybrid_pipelines.get(doc_name)
                if use_hybrid
                else self.standard_pipelines.get(doc_name)
            )
            if pipeline:
                doc_results = pipeline.retrieve(question, top_k=top_k_per_doc)
                merged_docs.extend(doc_results)

        return merged_docs

    def get_formatted_context(
        self,
        question: str,
        mode: str = "hybrid",
        top_k_per_doc: int = 3
    ) -> str:
        """
        Retrieves relevant documents and formats them with clear document headers and page numbers.
        """
        docs = self.retrieve(question, mode=mode, top_k_per_doc=top_k_per_doc)
        if not docs:
            return "No relevant document context found."

        formatted_snippets = []
        for doc in docs:
            source = doc.metadata.get("source_name", "Unknown Document")
            page = doc.metadata.get("page_number", doc.metadata.get("page", 1))
            formatted_snippets.append(
                f"[Document: {source} | Page: {page}]\n{doc.page_content.strip()}"
            )

        return "\n\n---\n\n".join(formatted_snippets)


# =========================================================
# FACTORY / PIPELINE SETUP
# =========================================================

@traceable(name="setup_pipeline")
def setup_pipeline(
    files: Union[List[Tuple[str, str]], List[str], str]
) -> DocumentCollection:
    """
    Ingests 1 or more PDF files into a DocumentCollection containing:
    - Per-PDF StandardRAGPipelines (Dense FAISS)
    - Per-PDF HybridRAGPipelines (Dense FAISS + Sparse BM25 + RRF)
    - Global merged pipelines
    """
    file_list: List[Tuple[str, str]] = []

    if isinstance(files, str):
        file_list = [(files, os.path.basename(files))]
    elif isinstance(files, list):
        for item in files:
            if isinstance(item, tuple):
                file_list.append(item)
            else:
                file_list.append((item, os.path.basename(item)))

    doc_names = []
    standard_pipelines: Dict[str, StandardRAGPipeline] = {}
    hybrid_pipelines: Dict[str, HybridRAGPipeline] = {}
    all_splits: List[Document] = []

    for path, filename in file_list:
        raw_docs = load_pdf(path, filename)
        if not raw_docs:
            continue
        
        splits = split_documents(raw_docs)
        if not splits:
            continue

        doc_names.append(filename)
        all_splits.extend(splits)

        # Build Per-PDF FAISS VectorStore
        doc_vectorstore = FAISS.from_documents(splits, embeddings)
        standard_pipelines[filename] = StandardRAGPipeline(
            vectorstore=doc_vectorstore,
            doc_name=filename
        )

        # Build Per-PDF BM25 Retriever
        doc_bm25 = BM25Retriever.from_documents(splits)
        doc_bm25.k = max(len(splits), 10)

        hybrid_pipelines[filename] = HybridRAGPipeline(
            vectorstore=doc_vectorstore,
            bm25=doc_bm25,
            doc_name=filename
        )

    # Build Global merged pipelines if multiple documents
    global_standard = None
    global_hybrid = None
    if len(all_splits) > 0:
        global_vs = FAISS.from_documents(all_splits, embeddings)
        global_standard = StandardRAGPipeline(vectorstore=global_vs, doc_name="all_documents")
        global_bm25 = BM25Retriever.from_documents(all_splits)
        global_bm25.k = 20
        global_hybrid = HybridRAGPipeline(vectorstore=global_vs, bm25=global_bm25, doc_name="all_documents")

    return DocumentCollection(
        doc_names=doc_names,
        standard_pipelines=standard_pipelines,
        hybrid_pipelines=hybrid_pipelines,
        global_standard=global_standard,
        global_hybrid=global_hybrid
    )
