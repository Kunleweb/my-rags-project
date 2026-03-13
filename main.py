from __future__ import annotations

from typing import Literal

from src.config import AppConfig, get_backend_from_env
from src.embeddings import EmbeddingManager
from src.ingestion import load_text_and_pdfs, split_documents
from src.pipeline import AdvancedRAGPipeline, build_groq_llm
from src.retrieval import ChromaRAGRetriever, TypesenseRAGRetriever
from src.vectorstores import ChromaVectorStore, TypesenseVectorStore


def build_chroma_pipeline(force_ingest: bool = False) -> AdvancedRAGPipeline:
    cfg = AppConfig()
    embed_manager = EmbeddingManager()
    embed_manager.load_model()
    
    chroma_store = ChromaVectorStore()
    
    # Validates existing document count to determine if indexing is required.
    if chroma_store.count() == 0 or force_ingest:
        print("Ingesting and indexing documents...")
        # Ingestion + chunking
        raw_docs = load_text_and_pdfs(cfg.paths.data_dir)
        chunks = split_documents(raw_docs)

        # Embeddings + Chroma vector store
        texts = [doc.page_content for doc in chunks]
        embeddings = embed_manager.generate_embeddings(texts)
        chroma_store.add_documents(chunks, embeddings)
    else:
        print(f"Vector store already contains {chroma_store.count()} documents. Skipping ingestion.")

    retriever = ChromaRAGRetriever(chroma_store, embed_manager)
    llm = build_groq_llm()
    return AdvancedRAGPipeline(retriever=retriever, llm=llm)


def build_typesense_pipeline(force_ingest: bool = False) -> AdvancedRAGPipeline:
    # The Typesense implementation currently rebuilds the collection on each execution,
    # as the remote-first nature of the Typesense wrapper is optimized for hosted environments.
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


def run_demo(backend: Literal["chroma", "typesense"] = "chroma", force_ingest: bool = False) -> None:
    """
    Executes a demonstration of the RAG pipeline, showcasing:
    - LLM-based response generation via Groq.
    - Persistent document storage across different vector backends.
    """
    print(f"Selected backend: {backend}")

    if backend == "typesense":
        pipeline = build_typesense_pipeline(force_ingest=force_ingest)
    else:
        pipeline = build_chroma_pipeline(force_ingest=force_ingest)
    print("\n--- RAG Interactive CLI ---")
    print("Type your questions below. Type 'exit' or 'quit' to stop.")
    
    while True:
        try:
            question = input("\nQuery: ").strip()
            
            if not question:
                continue
            if question.lower() in ["exit", "quit"]:
                print("Exiting CLI...")
                break
                
            result = pipeline.query(question, top_k=3, min_score=0.1, summarize=True)

            print("\nAnswer (Groq):")
            print(result["answer"])
            
            if result["summary"]:
                print("\nExecutive Summary:")
                print(result["summary"])
                
            print("\nSources:")
            for src in result["sources"]:
                source_name = src.get('source', 'Unknown')
                page_num = src.get('page', '?')
                print(f"- {source_name} (page {page_num})")
                
        except KeyboardInterrupt:
            print("\nExiting CLI...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    import sys
    backend = get_backend_from_env(default="chroma")
    force = "--force" in sys.argv
    run_demo(backend, force_ingest=force)

