from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import AppConfig


config = AppConfig()


def load_text_and_pdfs(
    base_data_dir: Path | None = None,
) -> List[Document]:
    """
    Parses and loads text and PDF documents from the specified directory into LangChain-compatible Document objects.
    """
    base_dir = base_data_dir or config.paths.data_dir
    pdf_dir = base_dir / "pdf"
    text_dir = base_dir / "text_files"

    documents: List[Document] = []

    if text_dir.exists():
        text_loader = DirectoryLoader(
            str(text_dir),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=False,
        )
        documents.extend(text_loader.load())

    if pdf_dir.exists():
        # Processes PDF documents using PyPDFLoader and enriches them with foundational metadata.
        pdf_files = list(Path(pdf_dir).glob("**/*.pdf"))
        for pdf_path in pdf_files:
            loader = PyPDFLoader(str(pdf_path))
            pdf_docs = loader.load()
            for doc in pdf_docs:
                doc.metadata["source_file"] = pdf_path.name
                doc.metadata["file_type"] = "pdf"
            documents.extend(pdf_docs)

    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Document]:
    """Partitions document objects into smaller, overlapping segments to optimize retrieval precision."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)

