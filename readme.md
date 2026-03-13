## RAG-1: Retrieval-Augmented Generation with Chroma, Typesense, and Groq

This project is a **hands-on RAG playground**. It shows how to:

- Ingest and chunk local PDFs / text files.
- Turn those chunks into embeddings.
- Store embeddings in **either**:
  - A local **Chroma** vector store, or
  - A remote **Typesense** vector store (via LangChain’s `Typesense` integration).
- Use **Groq** LLMs to answer questions over those documents.

You can run it as a small script (`main.py`) or from the Jupyter notebooks.

---

## 1. High-level architecture

- **Data ingestion** (in `src/ingestion.py`):
  - Reads from `data/text_files/**/*.txt` and `data/pdf/**/*.pdf`.
  - Uses LangChain `TextLoader` / `PyPDFLoader` to create `Document` objects.
  - Splits documents into overlapping chunks using `RecursiveCharacterTextSplitter`.

- **Embeddings** (in `src/embeddings.py`):
  - Uses `sentence-transformers/all-MiniLM-L6-v2` via `SentenceTransformer`.
  - `EmbeddingManager` wraps the model and exposes `generate_embeddings(texts)`.

- **Vector stores** (in `src/vectorstores.py`):
  - **`ChromaVectorStore`**:
    - Local, persistent Chroma collection.
    - Stores embeddings and metadata on disk (under `data/vector_store`).
  - **`TypesenseVectorStore`**:
    - Uses LangChain’s `Typesense` vectorstore.
    - Stores embeddings inside a Typesense collection (`quality-docs` by default).

- **Retrievers** (in `src/retrieval.py`):
  - **`ChromaRAGRetriever`**:
    - Uses `EmbeddingManager` + `ChromaVectorStore`.
    - Embeddings are stored in Chroma.
  - **`TypesenseRAGRetriever`**:
    - Uses `TypesenseVectorStore`.
    - Embeddings are stored in Typesense.
  - Both expose a unified `retrieve(query, top_k, score_threshold)` that returns chunks with `content`, `metadata`, and scores (when available).

- **RAG pipeline & Groq LLM** (in `src/pipeline.py`):
  - **Groq usage**:
    - `build_groq_llm()` constructs a `ChatGroq` client from environment variables.
    - All LLM calls go through this (no keys in code).
  - **RAG functions**:
    - `rag_simple(query, retriever, llm, ...)` – basic “retrieve + answer”.
    - `rag_advanced(...)` – returns answer, sources, and a simple confidence signal.
  - **`AdvancedRAGPipeline` class**:
    - Higher-level wrapper that:
      - Runs retrieval.
      - Calls Groq to answer.
      - Adds citations and (optionally) a summary.
      - Keeps a history of queries.

- **Configuration** (in `src/config.py`):
  - Centralizes:
    - Paths (`data/`, `data/pdf`, `data/text_files`, `data/vector_store`).
    - Groq configuration (API key, model, temperature, max tokens).
    - Chroma configuration (collection name, path).
    - Typesense configuration (host, port, protocol, API key, collection name).
  - `get_backend_from_env(default="chroma")` controls whether you use **Chroma** or **Typesense** for embeddings.

---

## 2. Backends: where embeddings live

You can switch between two RAG backends:

- **Chroma backend (`RAG_BACKEND=chroma`)**:
  - Embeddings are computed locally with `sentence-transformers`.
  - Embeddings + metadata are stored in a local Chroma collection.
  - Great for fully local experimentation.

- **Typesense backend (`RAG_BACKEND=typesense`)**:
  - LangChain’s `Typesense` vectorstore handles embeddings + storage.
  - Embeddings are stored in a Typesense collection (e.g. in Typesense Cloud).
  - Good if you already use Typesense or want a remote, hosted vector store.

The rest of the RAG pipeline (retrieval interface + Groq answering) is the same in both cases.

---

## 3. Project layout

Key files/folders:

- `main.py`  
  Script entrypoint that wires everything together and runs a demo query.

- `src/config.py`  
  Configuration for paths, Groq, Chroma, Typesense, and backend selection.

- `src/ingestion.py`  
  Functions to load PDFs/text files and split them into chunks.

- `src/embeddings.py`  
  `EmbeddingManager` around `SentenceTransformer`.

- `src/vectorstores.py`  
  `ChromaVectorStore` and `TypesenseVectorStore` (where embeddings are stored).

- `src/retrieval.py`  
  `ChromaRAGRetriever` and `TypesenseRAGRetriever` (unified retrieval interface).

- `src/pipeline.py`  
  Groq LLM setup, `rag_simple`, `rag_advanced`, and `AdvancedRAGPipeline`.

- `notebook/document.ipynb`  
  A step-by-step notebook that shows:
  - How ingestion, chunking, embeddings, and vector stores work.
  - A final cell that uses the refactored `src/` modules to run the same RAG pipeline.

- `typesense.ipynb`  
  Notebook showing Typesense usage (classic keyword search and RAG via the shared pipeline).

- `data/`  
  - `data/pdf/` – example PDFs (e.g. `cheatsheet.pdf`).  
  - `data/text_files/` – simple text examples created in the notebook.  
  - `data/vector_store/` – Chroma persistence directory.

---

## 4. Environment variables (`.env`)

The project expects a `.env` file in the repo root with at least:

```env
GROQ_API_KEY=your_groq_api_key_here
TYPESENSE_API_KEY=your_typesense_api_key_here

RAG_BACKEND=chroma   # or typesense
```

Optional (with sensible defaults if omitted):

```env
GROQ_MODEL_NAME=llama-3.1-8b-instant
GROQ_TEMPERATURE=0.1
GROQ_MAX_TOKENS=1024
CHROMA_COLLECTION_NAME=pdf_documents
TYPESENSE_HOST=tobd21ghuvcmr46fp-1.a2.typesense.net
TYPESENSE_PORT=443
TYPESENSE_PROTOCOL=https
TYPESENSE_COLLECTION_NAME=quality-docs
```

The code uses `python-dotenv` to load this automatically at import time.

---

## 5. Installing and running

### 5.1. Install dependencies

Using `pip`:

```bash
python -m venv .venv
.venv\Scripts\activate  # on Windows

pip install -e .
```

or:

```bash
pip install -r requirements.txt
```

### 5.2. Prepare data

Place your documents under:

- PDFs: `data/pdf/*.pdf` (the repo already includes a `cheatsheet.pdf` example).  
- Text files: `data/text_files/*.txt` (the notebook can generate a couple of sample files for you).

### 5.3. Choose backend

In your `.env`:

- **Use Chroma (local)**:

```env
RAG_BACKEND=chroma
```

- **Use Typesense (remote)**:

```env
RAG_BACKEND=typesense
```

Make sure `TYPESENSE_API_KEY` and `TYPESENSE_HOST`/`PORT`/`PROTOCOL` are set correctly for your Typesense instance.

---

## 6. How to run the project

### 6.1. Run the script demo (`main.py`)

This is the simplest way to see everything end-to-end.

```bash
.venv\Scripts\activate  # if not already

python main.py
```

What it does:

1. Reads `RAG_BACKEND` from `.env`.
2. Loads and chunks documents from `data/`.
3. Builds either:
   - A Chroma vector store (embeddings in Chroma), or
   - A Typesense vector store (embeddings in Typesense).
4. Builds a `AdvancedRAGPipeline` with a Groq LLM.
5. Asks: **“What skills is the interviewer looking for?”**  
6. Prints the question, Groq‑generated answer, and sources (file + page).

This is where you can clearly see:

- **Groq is used for generation** (via `build_groq_llm()` in `src/pipeline.py`).
- **Embeddings are stored** in Chroma or Typesense, depending on `RAG_BACKEND`.

---

### 6.2. Run from the `document.ipynb` notebook

Open `notebook/document.ipynb` and:

1. Run the earlier cells if you want to see the “from scratch” implementation (data ingestion, chunking, embeddings, manual Chroma usage, etc.).
2. Scroll to the **“Refactored RAG usage (backed by src/ modules)”** cell (at the end).
3. Run that final cell.

That cell:

- Imports from `src.config`, `src.ingestion`, `src.embeddings`, `src.vectorstores`, `src.retrieval`, `src.pipeline`.
- Reads `RAG_BACKEND` to decide:
  - Chroma vs Typesense for embeddings.
- Builds the same `AdvancedRAGPipeline` as `main.py`.
- Runs the demo query and returns the full result dictionary (answer, sources, history).

This is a nice way to:

- Tweak parameters interactively (e.g. chunk sizes, `top_k`, `min_score`).
- Inspect the retrieved chunks, metadata, and raw LLM output.

---

### 6.3. Run the Typesense-focused notebook

Open `typesense.ipynb`.

It shows:

- How to:
  - Set up a Typesense collection.
  - Import `books.jsonl` for classic keyword search.
  - Use the shared RAG pipeline with **Typesense** as the vector store backend.

For the RAG part, it delegates to:

- `TypesenseVectorStore` and `TypesenseRAGRetriever`.
- `build_groq_llm()` + `rag_simple()` from `src/pipeline.py`.

This gives you a clear example of **Groq + Typesense embeddings** end-to-end.

---

## 7. Customizing for your own documents

To adapt this project to your own knowledge base:

1. Drop your PDFs into `data/pdf/` and/or `.txt`/`.md` files into `data/text_files/`.
2. Optionally adjust:
   - Chunk size / overlap in `split_documents()` (`src/ingestion.py`).
   - Embedding model in `EmbeddingManager` (`src/embeddings.py`) or the Typesense embedding model in `TypesenseVectorStore`.
3. Run `python main.py` or the final cell in `document.ipynb`.

The retrieval pipeline and Groq LLM usage stay the same; only the underlying documents change.

---

## 8. Summary

This repo is a **practical, modular RAG lab**:

- Clear separation between:
  - Ingestion/chunking,
  - Embeddings,
  - Vector storage (Chroma vs Typesense),
  - Retrieval,
  - LLM‑powered answer generation (Groq).
- A single environment file (`.env`) controls:
  - Secrets (Groq, Typesense),
  - Backend choice.
- You can run it:
  - As a script (`python main.py`), or
  - Interactively from notebooks, while reusing the same core pipeline.

From here, you can easily extend it into an API, CLI tool, or UI on top of the existing RAG pipeline. 