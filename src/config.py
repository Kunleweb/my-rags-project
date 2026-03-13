import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


load_dotenv()


@dataclass
class PathsConfig:
    """Configuration for local data paths."""

    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = base_dir / "data"
    pdf_dir: Path = data_dir / "pdf"
    text_dir: Path = data_dir / "text_files"
    chroma_dir: Path = data_dir / "vector_store"


@dataclass
class GroqConfig:
    """Configuration for Groq LLM."""

    api_key: Optional[str] = os.getenv("GROQ_API_KEY")
    model_name: str = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")
    temperature: float = float(os.getenv("GROQ_TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("GROQ_MAX_TOKENS", "1024"))


@dataclass
class ChromaConfig:
    """Configuration for Chroma vector store."""

    collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "pdf_documents")
    persist_directory: Path = PathsConfig().chroma_dir


@dataclass
class TypesenseConfig:
    """Configuration for Typesense (for vector store use)."""

    host: str = os.getenv("TYPESENSE_HOST", "tobd21ghuvcmr46fp-1.a2.typesense.net")
    port: str = os.getenv("TYPESENSE_PORT", "443")
    protocol: str = os.getenv("TYPESENSE_PROTOCOL", "https")
    api_key: Optional[str] = os.getenv("TYPESENSE_API_KEY")
    collection_name: str = os.getenv("TYPESENSE_COLLECTION_NAME", "quality-docs")


@dataclass
class AppConfig:
    """Top-level configuration holder."""

    paths: PathsConfig = PathsConfig()
    groq: GroqConfig = GroqConfig()
    chroma: ChromaConfig = ChromaConfig()
    typesense: TypesenseConfig = TypesenseConfig()


def get_backend_from_env(default: str = "chroma") -> str:
    """
    Return selected RAG backend.

    Values: "chroma" (default) or "typesense".
    """
    backend = os.getenv("RAG_BACKEND", default).lower()
    if backend not in {"chroma", "typesense"}:
        return default
    return backend

