"""RAG subsystem: ingestion, retrieval, subgraph."""

from app.rag.ingestion import (
    ingest_documents,
    iter_documents,
    load_and_split,
)
from app.rag.retriever import get_retriever, get_vector_store
from app.rag.splitter import SplitterConfig, split_legal_text
from app.rag.subgraph import RAGState, build_rag_subgraph

__all__ = [
    "RAGState",
    "SplitterConfig",
    "build_rag_subgraph",
    "get_retriever",
    "get_vector_store",
    "ingest_documents",
    "iter_documents",
    "load_and_split",
    "split_legal_text",
]

