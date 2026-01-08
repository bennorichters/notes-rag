# Claude

## General
- This project aims to connect Ollama to a set of notes.
- Each note is written in a separate
- All notes are stored in one folder and sub folders
- All notes are written in Markdown format
- Some notes end with a tag line. Tags are colon separated. The line starts and ends with a colon.
- Notes can be written in different languages.

## index.py
- Builds a RAG in ChromaDB based on the available notes.
- Indexing takes markdown format, code blocks, document titles and tags into account.

## ask.pu
- Asks the user for a question
- Asks Ollama for query input to send to ChromaDB
- Queries ChromaDB
- Sends the resulting document to Ollama
- Ollama gives a final answer

