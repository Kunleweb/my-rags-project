# RAG-1: A Modular RAG Framework

This project is a simple, modular system for Retrieval-Augmented Generation (RAG). It handles everything from loading your documents to answering questions using Groq's fast LLMs. You can use it as a playground to test different ways of storing and searching data, using either local storage (Chroma) or a remote search engine (Typesense).

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

### Tips for Customizing:
- **Starting Fresh**: If you want to clear your local database and re-index everything, just delete the `data/vector_store` folder.
- **Auto-Indexing**: The system will automatically find, chunk, and index any new files the next time you run a script or notebook. It only processes files that have changed, so it's very efficient.

---

## 4. Explore with Notebooks

There are two main ways to interact with the system:

- **`RAG_Playground.ipynb`**: A quick place to run the whole pipeline and see how it works.
- **`RAG_System_Deep_Dive.ipynb`**: A detailed walkthrough that explains every step of the process in plain English. This is the best place to start if you want to learn how the system is built.

### Running from the command line

You can also run the whole thing as a script:

```bash
python main.py
```

If you want to force it to re-index everything from scratch, use the `--force` flag:

```bash
python main.py --force
```

---

## 5. Summary

RAG-1 is a clean, professional implementation of a RAG system. By keeping everything modular, it's easy to tweak the settings or swap out parts of the pipeline to see what works best for your data.