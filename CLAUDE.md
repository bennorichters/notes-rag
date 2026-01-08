# Overview
- This project aims to connect Ollama to a set of notes while leveraging a RAG.
- Each note is written in a separate file with the format `[IRRELEVANT-NAME].md`.
- All notes are stored in one folder and its sub folders.
- All notes are written in Markdown format.
- Notes can be written in different languages.

# Setup
## Prerequisites
-  Use `uv` for dependency management
- See `pyproject.toml` for the dependencies

## Configuration
# Architecture
- The project has two scripts (`index.py` and `ask.py`) than can be ran independently.
- See `pyproject.toml` for the dependencies

# Usage
## Indexing Notes
Indexing notes is done with `index.py`:  
`uv run index.py`

The script `index.py`:  
- Builds a RAG in ChromaDB based on the available notes.
- Indexing takes markdown format, code blocks, document titles and tags into account.

## Querying
The user can ask Ollama a question about the notes using `ask.py`:  
`uv run ask.py`

The script `ask.py`:   
- Can only be ran if `index.py` has been ran before.
- Asks the user for a question.
- Asks Ollama for query input to send to ChromaDB.
- Queries ChromaDB.
- Sends the resulting document to Ollama.
- Ollama gives a final answer.

# Note Format
## Title
Most notes, but not all, start with a single pound (`#`) header. That is the title of the document.

## Tags
- Some notes end with a tag line. Tags are colon separated. The line starts and ends with a colon.
