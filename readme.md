# RAG PIPELINE FOR DOCUMENTATION

## Problem

Developer documentation is often large, fragmented, and difficult to navigate. Developers frequently spend significant time searching through long documentation files or multiple sources to find specific information. Traditional keyword search does not always capture the intent of a query, making it inefficient to locate the exact section needed.

## Solution

This project builds a **Retrieval-Augmented Generation (RAG)** system for developer documentation. The goal is to transform documentation into a **semantically searchable knowledge base** that allows developers to quickly retrieve the most relevant sections of documentation.

## Current Implementation

The **data ingestion pipeline** has been implemented. This pipeline:

* `Loads documentation from PDF files`
* `Extracts text from documents`
* `Splits documents into smaller chunks`
* `Generates embeddings for each chunk`
* `Stores embeddings and metadata in a vector database`

Embeddings are generated using **sentence-transformers** and stored in **ChromaDB** to enable semantic similarity search.

## Work in Progress

The **retrieval layer** is currently being developed. This will handle:

* `Query embedding generation`
* `Similarity search over stored vectors`
* `Returning the most relevant documentation chunks for a given query`

## Tech Stack (@ current implementation)

* langchain
* langchain-core
* langchain-community
* pypdf
* pymupdf
* sentence-transformers
* chromadb
* faiss-cpu (not using right now)
* ipykernel

## Planned Storage Architecture

The project will use Amazon S3 as the document storage layer.

S3 will act as the persistent storage location for raw documentation files (PDFs and other documents). The ingestion pipeline will read documents from S3, process them, and generate embeddings that are stored in the vector database.

This design separates:

Document storage → Amazon S3

Embedding storage and retrieval → ChromaDB

Processing and embedding generation → Python ingestion pipeline

Using S3 allows the system to scale document storage while keeping the retrieval pipeline independent from cloud-based ML services.