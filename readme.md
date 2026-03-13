# RAG-1: A Modular RAG Framework

This project implements a modular Retrieval-Augmented Generation (RAG) pipeline designed to make it easier to query and interact with document collections using modern language models. The system handles the full workflow—from loading and preprocessing documents, to generating embeddings, storing them in a vector database, and retrieving relevant context to answer user queries.

It is designed as a lightweight and extensible environment for experimenting with different retrieval strategies and storage backends. The current implementation supports both local vector storage with Chroma and remote search using Typesense, allowing flexible experimentation with different retrieval infrastructures.

For generation, the pipeline integrates with Groq’s high-performance LLM APIs, enabling fast response times while grounding answers in retrieved document context. The modular structure also makes it easy to swap components such as embedding models, vector stores, or retrieval strategies.

## What it does

The system follows a standard RAG pipeline:

- **Loading Data**: It scans your folders for PDFs and text files and breaks them into smaller pieces.
- **Creating Embeddings**: It turns those pieces of text into numbers (vectors) so the computer can understand their meaning.
- **Storing Data**: It saves those vectors into a database (Chroma or Typesense) so you can search them later.
- **Asking Questions**: It finds the right pieces of text and uses an LLM to give you a clear, cited answer.

---

## 1. How it's built

The code is split into a few simple parts:

- **Ingestion (`src/ingestion.py`)**: Finds your files and cuts them into overlapping chunks so no information is lost.
- **Embeddings (`src/embeddings.py`)**: Uses `sentence-transformers` to create high-quality vector representations of your text.
- **Storage (`src/vectorstores.py`)**: A single place to manage both Chroma and Typesense. It uses a hashing trick to make sure it doesn't index the same file twice.
- **Retrieval (`src/retrieval.py`)**: Searches the database for the most relevant text based on what you asked.
- **The Pipeline (`src/pipeline.py`)**: Puts everything together—it grabs the text, talks to the LLM, and gives you the final answer.

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

Install the requirements using `pip`:

```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

---

## 3. Setting up your data

This project expects a specific folder structure for your documents. By default, the `data/` folder is ignored by Git to keep your data private. 

To get started, create the following folders in the root directory:

```text
data/
├── pdf/           <-- Put your .pdf files here
└── text_files/    <-- Put your .txt files here
```

### Smart Indexing
The system automatically finds, chunks, and indexes any new files the next time you run a script or notebook. It uses content-based hashing to skip files it has already processed, making it very fast and efficient during subsequent runs.

---

## 4. Explore with Notebooks

There are two main ways to interact with the system:

- **`RAG_Playground.ipynb`**: A quick place to run the whole pipeline and see how it works.
- **`RAG_System_Deep_Dive.ipynb`**: A detailed walkthrough that explains every step of the process.

### Interactive CLI

You can also run the whole thing as an interactive script in your terminal:

(might take a bit to run)
```bash
python main.py
```

Once it starts, you can enter questions and you will get a response. Type Ctrl+C** to stop.

