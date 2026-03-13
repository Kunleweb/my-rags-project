from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Protocol

import numpy as np

from .embeddings import EmbeddingManager
from .vectorstores import ChromaVectorStore, RetrievedChunk, TypesenseVectorStore


class Retriever(Protocol):
    """Defines a standardized interface for document retrieval operations within the RAG framework."""

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        ...


class ChromaRAGRetriever:
    """Retriever implementation utilizing the local Chroma vector database and a dedicated embedding manager."""

    def __init__(self, vector_store: ChromaVectorStore, embedding_manager: EmbeddingManager) -> None:
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        query_emb: np.ndarray = self.embedding_manager.generate_embeddings([query])[0]
        chunks: List[RetrievedChunk] = self.vector_store.query(
            query_emb, top_k=top_k, score_threshold=score_threshold
        )
        return [asdict(chunk) for chunk in chunks]


class TypesenseRAGRetriever:
    """Retriever implementation utilizing the remote Typesense vector search engine."""

    def __init__(self, vector_store: TypesenseVectorStore) -> None:
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,  # kept for API compatibility; Typesense does not expose scores
    ) -> List[Dict[str, Any]]:
        chunks = self.vector_store.similarity_search(query, top_k)
        # score_threshold is ignored because Typesense / LC_Typesense does not expose distances
        return [asdict(chunk) for chunk in chunks]

