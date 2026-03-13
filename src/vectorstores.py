from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import chromadb
import numpy as np
from chromadb import Collection
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Typesense as LC_Typesense

from .config import AppConfig


config = AppConfig()


@dataclass
class RetrievedChunk:
    id: str
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    distance: float
    rank: int


class ChromaVectorStore:
    """Chroma-based vector store (local, persistent)."""

    def __init__(
        self,
        collection_name: str | None = None,
        persist_directory: str | None = None,
    ) -> None:
        collection_name = collection_name or config.chroma.collection_name
        persist_directory = str(
            persist_directory or config.chroma.persist_directory
        )

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "PDF/text document embeddings for RAG"},
        )

    def add_documents(
        self,
        documents: Sequence[Any],
        embeddings: np.ndarray,
    ) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings.")

        ids: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        documents_text: List[str] = []
        embeddings_list: List[list[float]] = []

        for idx, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{idx}"
            ids.append(doc_id)

            metadata = dict(getattr(doc, "metadata", {}) or {})
            metadata["doc_index"] = idx
            metadata["content_length"] = len(getattr(doc, "page_content", ""))
            metadatas.append(metadata)

            documents_text.append(getattr(doc, "page_content", ""))
            embeddings_list.append(embedding.tolist())

        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=documents_text,
        )

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[RetrievedChunk]:
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )

        retrieved: List[RetrievedChunk] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        for rank, (doc_id, text, meta, dist) in enumerate(
            zip(ids, docs, metas, distances), start=1
        ):
            similarity = 1 - dist
            if similarity < score_threshold:
                continue
            retrieved.append(
                RetrievedChunk(
                    id=doc_id,
                    content=text,
                    metadata=meta,
                    similarity_score=similarity,
                    distance=dist,
                    rank=rank,
                )
            )

        return retrieved


class TypesenseVectorStore:
    """
    Typesense-backed vector store using LangChain's Typesense integration.

    This stores embeddings in a Typesense collection and exposes a unified
    similarity search interface compatible with the rest of the RAG pipeline.
    """

    def __init__(
        self,
        collection_name: str | None = None,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.collection_name = collection_name or config.typesense.collection_name
        self.embedding_model_name = embedding_model_name

        if not config.typesense.api_key:
            raise ValueError(
                "TYPESENSE_API_KEY is not set. Please set it in your environment."
            )

        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.vectorstore: LC_Typesense | None = None

    def build_from_documents(self, documents: Sequence[Any]) -> None:
        """Create or replace a Typesense collection from LangChain documents."""
        self.vectorstore = LC_Typesense.from_documents(
            documents,
            self.embeddings,
            typesense_client_params={
                "host": config.typesense.host,
                "port": config.typesense.port,
                "protocol": config.typesense.protocol,
                "typesense_api_key": config.typesense.api_key,
            },
            typesense_collection_name=self.collection_name,
        )

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        if self.vectorstore is None:
            raise ValueError(
                "Typesense vector store not initialized. "
                "Call `build_from_documents` first."
            )

        docs = self.vectorstore.similarity_search(query, k=top_k)
        chunks: List[RetrievedChunk] = []
        for rank, doc in enumerate(docs, start=1):
            meta = getattr(doc, "metadata", {}) or {}
            chunks.append(
                RetrievedChunk(
                    id=str(meta.get("id", rank)),
                    content=getattr(doc, "page_content", ""),
                    metadata=meta,
                    similarity_score=1.0,  # not provided directly by Typesense
                    distance=0.0,
                    rank=rank,
                )
            )
        return chunks

