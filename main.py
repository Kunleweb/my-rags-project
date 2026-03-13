from __future__ import annotations

from typing import Literal

from src.config import AppConfig, get_backend_from_env
from src.embeddings import EmbeddingManager
from src.ingestion import load_text_and_pdfs, split_documents
from src.pipeline import AdvancedRAGPipeline, build_groq_llm
from src.retrieval import ChromaRAGRetriever, TypesenseRAGRetriever
from src.vectorstores import ChromaVectorStore, TypesenseVectorStore


def build_chroma_pipeline() -> AdvancedRAGPipeline:
    cfg = AppConfig()

    # Ingestion + chunking
    raw_docs = load_text_and_pdfs(cfg.paths.data_dir)
    chunks = split_documents(raw_docs)

    # Embeddings + Chroma vector store
    embed_manager = EmbeddingManager()
    embed_manager.load_model()
    texts = [doc.page_content for doc in chunks]
    embeddings = embed_manager.generate_embeddings(texts)

    chroma_store = ChromaVectorStore()
    chroma_store.add_documents(chunks, embeddings)

    retriever = ChromaRAGRetriever(chroma_store, embed_manager)
    llm = build_groq_llm()
    return AdvancedRAGPipeline(retriever=retriever, llm=llm)


def build_typesense_pipeline() -> AdvancedRAGPipeline:
    cfg = AppConfig()

    # Ingestion + chunking
    raw_docs = load_text_and_pdfs(cfg.paths.data_dir)
    chunks = split_documents(raw_docs)

    # Typesense stores its own embeddings via LangChain's Typesense integration
    typesense_store = TypesenseVectorStore()
    typesense_store.build_from_documents(chunks)

    retriever = TypesenseRAGRetriever(typesense_store)
    llm = build_groq_llm()
    return AdvancedRAGPipeline(retriever=retriever, llm=llm)


def run_demo(backend: Literal["chroma", "typesense"] = "chroma") -> None:
    """
    Simple console demo to show:
    - How Groq is used for generation
    - How embeddings are stored either in Chroma or Typesense
    """
    print(f"Selected backend: {backend}")

    if backend == "typesense":
        pipeline = build_typesense_pipeline()
    else:
        pipeline = build_chroma_pipeline()

    question = "What skills is the interviewer looking for?"
    result = pipeline.query(question, top_k=3, min_score=0.1, summarize=True)

    print("\nQuestion:")
    print(result["question"])
    print("\nAnswer (Groq):")
    print(result["answer"])
    print("\nSources:")
    for src in result["sources"]:
        print(f"- {src['source']} (page {src['page']})")


if __name__ == "__main__":
    backend = get_backend_from_env(default="chroma")
    run_demo(backend)

