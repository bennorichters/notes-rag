# notes-rag

A Retrieval-Augmented Generation (RAG) system for querying markdown notes using Ollama and ChromaDB.

## Overview

This system indexes markdown notes stored in a directory structure and enables semantic search with LLM-generated answers. It uses vector embeddings for initial retrieval and LLM re-ranking for improved accuracy.

## Features

- **Markdown indexing**: Processes notes with smart chunking that preserves document structure, code blocks, and lists
- **Multi-language support**: Uses BGE-M3 embedding model for cross-lingual retrieval
- **Tag extraction**: Automatically extracts and indexes tags from notes (format: `:tag1:tag2:`)
- **Two-stage retrieval**: Combines embedding-based search with LLM re-ranking
- **Document-level answers**: Retrieves full documents and generates answers from complete context

## Prerequisites

- Python >= 3.11
- [uv](https://github.com/astral-sh/uv) package manager
- [Ollama](https://ollama.ai/) with llama3.1 model

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd notes-rag
```

2. Install uv (if not installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Configure environment variables:
```bash
cp .env.example .env
```

Edit `.env` and set:
- `NOTES_PATH`: Path to your notes directory
- `CHROMA_PATH`: Path for ChromaDB storage (default: `./chroma_data`)

4. Install dependencies:
```bash
uv sync
```

5. Pull the Ollama model:
```bash
ollama pull llama3.1
```

## Usage

### Indexing Notes

Run the indexer to build the RAG database:

```bash
uv run index.py
```

This script:
- Scans the notes directory for `.md` files
- Extracts titles, tags, and content
- Chunks documents based on markdown structure
- Generates embeddings using BGE-M3
- Stores chunks in ChromaDB

### Querying Notes

Query the indexed notes:

```bash
uv run ask.py
```

The query process:
1. Prompts for a question
2. Embeds the question using BGE-M3
3. Retrieves top 8 candidate chunks from ChromaDB
4. Uses Ollama to re-rank candidates and select the most relevant document
5. Retrieves the full document content
6. Generates an answer using Ollama based on the document

## Note Format

### Structure

Notes are markdown files with optional elements:

- **Title** (optional): First line starting with `#`
- **Content**: Standard markdown with support for headings, lists, code blocks
- **Tags** (optional): Last line in format `:tag1:tag2:tag3:`

### Example

```markdown
# Example Note

## Section 1

Content here...

## Section 2

- List item 1
- List item 2

:python:tutorial:web:
```

## How It Works

### Indexing Pipeline

1. **Load notes**: Recursively scans directory for `.md` files (skips hidden folders)
2. **Extract metadata**: Pulls title from first `#` heading and tags from last line
3. **Chunk content**: Splits documents at `##` headings, then handles:
   - Code blocks: Preserved intact or split with proper fence markers
   - Lists: Chunked at list item boundaries
   - Plain text: Chunked with 500-character windows and 50-character overlap
4. **Embed chunks**: Generates embeddings using BGE-M3 (567M parameter model)
5. **Store in ChromaDB**: Saves embeddings with metadata (title, source, tags)

### Query Pipeline

1. **Embed question**: Converts user question to vector using same BGE-M3 model
2. **Vector search**: Retrieves top 8 chunks from ChromaDB by cosine similarity
3. **Confidence check**: Evaluates match quality based on distance scores
4. **LLM re-ranking**: Sends candidates to Ollama for semantic re-ranking
5. **Document retrieval**: Loads the full source document of the selected chunk
6. **Answer generation**: Uses Ollama to extract answer from document content

## Project Structure

```
notes-rag/
├── index.py           # Indexing script
├── ask.py             # Query script
├── pyproject.toml     # Project dependencies
├── .env.example       # Environment template
└── CLAUDE.md          # Project instructions
```

## Dependencies

- `chromadb==1.4.0`: Vector database for embeddings
- `sentence-transformers==5.2.0`: BGE-M3 embedding model
- `ollama==0.6.1`: LLM interface for llama3.1
- `python-dotenv==1.2.1`: Environment configuration
- `numpy<2.0`: Array operations

## License

See LICENSE file for details.
