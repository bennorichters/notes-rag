# Notes RAG

A Retrieval-Augmented Generation (RAG) system for querying your personal markdown notes using AI.

## Overview

This system allows you to:
- Index your markdown notes stored on a VPS
- Query them from your local laptop using natural language
- Get AI-generated answers based on your notes content
- Keep your notes and RAG index on your server, while using local Ollama for LLM inference

## Architecture

- **VPS (Dokku)**: Hosts the FastAPI service and ChromaDB vector database
- **Local Laptop**: Runs queries using Ollama for final answer generation
- **Security**: API key authentication over HTTPS

## Quick Start

### VPS Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

```bash
# Create app
dokku apps:create notes-rag

# Configure storage and environment
dokku storage:mount notes-rag /var/lib/dokku/data/storage/notes:/app/notes:ro
dokku config:set notes-rag API_KEY="your-secret-key" NOTES_PATH="/app/notes"

# Deploy
git push dokku main

# Build the index
dokku run notes-rag python index.py
```

### Local Usage

```bash
# Install dependencies
uv sync --extra local

# Configure
export API_URL=https://notes-rag.yourdomain.com
export API_KEY=your-secret-key

# Query your notes
python local_query.py "What are my notes about Python?"
```

## Features

- **Smart Chunking**: Intelligently splits markdown by sections, code blocks, and lists
- **Tag Support**: Extracts and indexes tags from notes (format: `:tag1:tag2:`)
- **Efficient Embedding**: Uses sentence-transformers for fast, local embeddings
- **Secure API**: API key authentication with HTTPS
- **Local LLM**: Uses your local Ollama for privacy and speed

## Files

- `index.py`: Builds the RAG index from markdown notes
- `api.py`: FastAPI service for querying the RAG
- `local_query.py`: Local script to query from your laptop
- `Procfile`: Dokku deployment configuration
- `DEPLOYMENT.md`: Comprehensive deployment guide

## Documentation

See [DEPLOYMENT.md](DEPLOYMENT.md) for:
- Complete deployment instructions
- Environment configuration
- Usage examples
- Troubleshooting
- Security best practices

## Requirements

### VPS
- Python 3.10+
- Dokku
- Notes in markdown format

### Local Laptop
- Python 3.10+
- Ollama

## License

MIT
