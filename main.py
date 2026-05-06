from __future__ import annotations

import sys
from typing import Literal

from src import PipelineFactory, get_backend_from_env


def start_interactive_cli(pipeline) -> None:
    """Runs a persistent interactive loop for user queries."""
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
                source_name = src.get("source", "Unknown")
                page_num = src.get("page", "?")
                print(f"- {source_name} (page {page_num})")

        except KeyboardInterrupt:
            print("\nExiting CLI...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")


def main() -> None:
    """Main entry point for the RAG application."""
    backend = get_backend_from_env(default="chroma")
    force_ingest = "--force" in sys.argv

    print(f"Initializing RAG system with backend: {backend}")
    
    try:
        # Using the PipelineFactory to abstract away the construction details
        pipeline = PipelineFactory.create(backend=backend, force_ingest=force_ingest)
        start_interactive_cli(pipeline)
    except Exception as e:
        print(f"Failed to initialize RAG pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
