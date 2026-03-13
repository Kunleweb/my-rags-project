import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


load_dotenv()


@dataclass
class PathsConfig:
    """Manages the filesystem structure for data and persistent storage."""

    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = base_dir / "data"
    pdf_dir: Path = data_dir / "pdf"
    text_dir: Path = data_dir / "text_files"
    chroma_dir: Path = data_dir / "vector_store"


@dataclass
class GroqConfig:
    """Manages configuration for the Groq Large Language Model integration."""

    api_key: Optional[str] = os.getenv("GROQ_API_KEY")
    model_name: str = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")
    temperature: float = float(os.getenv("GROQ_TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("GROQ_MAX_TOKENS", "1024"))


@dataclass
class ChromaConfig:
    """Manages configuration for the Chroma vector database."""

    collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "pdf_documents")
    persist_directory: Path = PathsConfig().chroma_dir


@dataclass
class TypesenseConfig:
    """Manages configuration for the Typesense vector search engine."""

    host: str = os.getenv("TYPESENSE_HOST", "tobd21ghuvcmr46fp-1.a2.typesense.net")
    port: str = os.getenv("TYPESENSE_PORT", "443")
    protocol: str = os.getenv("TYPESENSE_PROTOCOL", "https")
    api_key: Optional[str] = os.getenv("TYPESENSE_API_KEY")
    collection_name: str = os.getenv("TYPESENSE_COLLECTION_NAME", "quality-docs")


@dataclass
class AppConfig:
    """Top-level configuration holder."""

    # Use default_factory to avoid mutable default instances being shared
    paths: PathsConfig = None  # type: ignore[assignment]
    groq: GroqConfig = None  # type: ignore[assignment]
    chroma: ChromaConfig = None  # type: ignore[assignment]
    typesense: TypesenseConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.paths is None:
            self.paths = PathsConfig()
        if self.groq is None:
            self.groq = GroqConfig()
        if self.chroma is None:
            self.chroma = ChromaConfig()
        if self.typesense is None:
            self.typesense = TypesenseConfig()


def get_backend_from_env(default: str = "chroma") -> str:
    """
    Return selected RAG backend.

    Values: "chroma" (default) or "typesense".
    """
    backend = os.getenv("RAG_BACKEND", default).lower()
    if backend not in {"chroma", "typesense"}:
        return default
    return backend

