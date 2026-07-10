# This module is intentionally thin.
# All vector store logic lives in retrieval/retriever.py (BGE embeddings + Chroma).
# Import from there directly.
from retrieval.retriever import build_vector_store, query_store

__all__ = ["build_vector_store", "query_store"]
