# Modular RAG Pipeline for Semantic Document Search using ChromaDB, Typesense and Groq LLMs

The pipeline integrates with Groq’s high-performance LLM APIs, enabling fast response times while grounding answers in retrieved document context. The modular structure also makes it easy to swap components such as embedding models, vector stores, or retrieval strategies.

## What it does

The system follows a standard RAG pipeline:

- **Loading Data**: It scans your folders for PDFs and text files and breaks them into smaller pieces.
- **Creating Embeddings**: It turns those pieces of text into numbers (vectors) so the computer can understand their meaning.
- **Storing Data**: It saves those vectors into a database (Chroma or Typesense) so you can search them later.
- **Asking Questions**: It finds the right pieces of text and uses an LLM to give you a clear, cited answer.

---

## 1. How it's built

The code is split into a few simple parts:

- **Ingestion (`src/ingestion.py`)**: Handles document loading and intelligent chunking.
- **Storage (`src/vectorstores.py`)**: Manages local (Chroma) and remote (Typesense) storage with content-based deduplication.
- **Retrieval (`src/retrieval.py`)**: Standardizes search operations across different backends using a shared `Retriever` protocol.
- **Pipeline (`src/pipeline.py`)**: Orchestrates LLM interaction and includes a `PipelineFactory` for easy backend switching.
- **Public API (`src/__init__.py`)**: Exposes key components for clean imports throughout the project.

---

## 2. Setup

### Configuration

Everything is controlled by a `.env` file in the root folder.

```env
GROQ_API_KEY=your_groq_api_key_here
TYPESENSE_API_KEY=your_typesense_api_key_here

# Choose your backend: 'chroma' (stays local) or 'typesense' (remote)
RAG_BACKEND=chroma
```

### Installation

This project uses `uv` for lightning-fast dependency management.

```bash
# Sync dependencies and create a virtual environment
uv sync
```

---

## 3. Setting up your data

This project expects a specific folder structure for your documents. By default, the `data/` folder is ignored by Git to keep your data private. 

To get started, create the following folders in the root directory:

```text
data/
-----pdf/           <-- Put your .pdf files here
-----text_files/    <-- Put your .txt files here
```

### Smart Indexing
The system automatically finds, chunks, and indexes any new files the next time you run a script or notebook. It uses content-based hashing to skip files it has already processed, making it very fast and efficient during subsequent runs.

---

## 4. Explore with Notebooks

The `notebooks/` directory contains a step-by-step walkthrough of the RAG lifecycle:

- **`notebooks/01_Ingestion_Pipeline.ipynb`**: Demonstrates document loading and vector indexing.
- **`notebooks/02_Retrieval_Architecture.ipynb`**: Explores local (Chroma) and remote (Typesense) retrieval.
- **`notebooks/03_LLM_Generation.ipynb`**: Shows how to synthesize answers with Groq LLMs.

### Interactive CLI

You can also run the whole thing as an interactive script in your terminal:

(might take a bit to run)
```bash
python main.py
```

Once it starts, you can enter questions and you will get a response. Type Ctrl+C** to stop.

