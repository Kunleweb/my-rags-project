from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

from langchain_groq import ChatGroq

from .config import AppConfig


config = AppConfig()


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        ...


def build_groq_llm() -> ChatGroq:
    """Return a configured Groq LLM client."""
    if not config.groq.api_key:
        raise ValueError("GROQ_API_KEY is not set. Please set it in your environment.")

    return ChatGroq(
        groq_api_key=config.groq.api_key,
        model_name=config.groq.model_name,
        temperature=config.groq.temperature,
        max_tokens=config.groq.max_tokens,
    )


def rag_simple(query: str, retriever: Retriever, llm: ChatGroq, top_k: int = 3) -> str:
    """Minimal RAG: retrieve context, ask Groq to answer."""
    results = retriever.retrieve(query, top_k=top_k)
    context = "\n\n".join(doc["content"] for doc in results) if results else ""
    if not context:
        return "No relevant context found to answer the question."

    prompt = f"""Use the following context to answer the question concisely.
Context:
{context}

Question: {query}

Answer:"""
    response = llm.invoke([prompt])
    return response.content


def rag_advanced(
    query: str,
    retriever: Retriever,
    llm: ChatGroq,
    top_k: int = 5,
    min_score: float = 0.2,
    return_context: bool = False,
) -> Dict[str, Any]:
    """
    RAG pipeline with extra features:
    - Returns answer, sources, and a simple confidence signal when available.
    """
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)
    if not results:
        return {
            "answer": "No relevant context found.",
            "sources": [],
            "confidence": 0.0,
            "context": "" if return_context else None,
        }

    context = "\n\n".join(doc["content"] for doc in results)
    sources = [
        {
            "source": doc["metadata"].get(
                "source_file", doc["metadata"].get("source", "unknown")
            ),
            "page": doc["metadata"].get("page", "unknown"),
            "score": doc.get("similarity_score"),
            "preview": doc["content"][:300] + "...",
        }
        for doc in results
    ]

    scores = [doc.get("similarity_score") for doc in results if doc.get("similarity_score") is not None]
    confidence = max(scores) if scores else 0.0

    prompt = f"""Use the following context to answer the question concisely.
Context:
{context}

Question: {query}

Answer:"""
    response = llm.invoke([prompt])

    out: Dict[str, Any] = {
        "answer": response.content,
        "sources": sources,
        "confidence": confidence,
    }
    if return_context:
        out["context"] = context
    return out


@dataclass
class AdvancedRAGPipeline:
    """Higher-level pipeline with citations and optional summarization."""

    retriever: Retriever
    llm: ChatGroq
    history: List[Dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.history is None:
            self.history = []

    def query(
        self,
        question: str,
        top_k: int = 5,
        min_score: float = 0.2,
        summarize: bool = False,
    ) -> Dict[str, Any]:
        results = self.retriever.retrieve(
            question,
            top_k=top_k,
            score_threshold=min_score,
        )
        if not results:
            answer = "No relevant context found."
            sources: List[Dict[str, Any]] = []
            context = ""
        else:
            context = "\n\n".join(doc["content"] for doc in results)
            sources = [
                {
                    "source": doc["metadata"].get(
                        "source_file", doc["metadata"].get("source", "unknown")
                    ),
                    "page": doc["metadata"].get("page", "unknown"),
                    "score": doc.get("similarity_score"),
                    "preview": doc["content"][:120] + "...",
                }
                for doc in results
            ]

            prompt = f"""Use the following context to answer the question concisely.
Context:
{context}

Question: {question}

Answer:"""
            resp = self.llm.invoke([prompt])
            answer = resp.content

        citations = [
            f"[{idx + 1}] {src['source']} (page {src['page']})"
            for idx, src in enumerate(sources)
        ]
        answer_with_citations = (
            answer + "\n\nCitations:\n" + "\n".join(citations) if citations else answer
        )

        summary = None
        if summarize and answer:
            summary_prompt = f"Summarize the following answer in 2 sentences:\n{answer}"
            summary_resp = self.llm.invoke([summary_prompt])
            summary = summary_resp.content

        record = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "summary": summary,
        }
        self.history.append(record)

        return {
            "question": question,
            "answer": answer_with_citations,
            "sources": sources,
            "summary": summary,
            "history": self.history,
        }

