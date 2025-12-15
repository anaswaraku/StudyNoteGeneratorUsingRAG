# AI Study Assistant

A local, AI-powered study assistant that uses Retrieval-Augmented Generation (RAG) to answer questions based on personal study materials. Built with Flask, LangChain, ChromaDB, and Groq (Llama 3).

## Features

-  RAG Architecture: Retrieves relevant information from documents.
-  Local Vector Store: Uses ChromaDB to store document embeddings.
-  LLM: Llama 3 (via Groq API).
-  Multi-Format Support: Ingests PDF, DOCX, and TXT files.
  
## Tech Stack

-  Python, Flask
-  Llama 3.1 8B (via Groq API)
-  LangChain
-  Chroma (Local)
-  Embeddings- `sentence-transformers/all-MiniLM-L6-v2` (Local)
