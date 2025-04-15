# HydRO - Hybrid Document RAG Orchestrator

A powerful Retrieval-Augmented Generation (RAG) system that combines PDF document processing, semantic search, and Gemini's language model capabilities.

## Features

- PDF document processing with text chunking
- Semantic search using FAISS and Sentence Transformers
- Integration with Google's Gemini model for answer generation
- Persistent storage of vector indices and metadata
- Interactive query interface

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

3. Place your PDF documents in the `data/` directory.

## Usage

Run the main script:
```bash
python hydro.py
```

The system will:
1. Process PDFs in the `data/` directory if running for the first time
2. Create and save vector embeddings using FAISS
3. Start an interactive query interface

## Directory Structure

- `data/` - Place your PDF documents here
- `vector_store/` - Stores FAISS index and metadata
- `hydro.py` - Main implementation
- `requirements.txt` - Project dependencies
