#!/usr/bin/env python3
"""
Local query script for notes RAG.
Run this on your laptop to query the VPS API and use local Ollama for answers.

Usage:
    python local_query.py "Your question here"

Configuration:
    Set these environment variables:
    - API_URL: URL of your VPS API (e.g., https://notes-rag.yourdomain.com)
    - API_KEY: Your secret API key
"""

import os
import sys
import requests
import ollama

# Configuration
API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")

if not API_URL:
    print("Error: API_URL environment variable not set")
    print("Example: export API_URL=https://notes-rag.yourdomain.com")
    sys.exit(1)

if not API_KEY:
    print("Error: API_KEY environment variable not set")
    print("Example: export API_KEY=your-secret-key")
    sys.exit(1)

def query_rag(question: str, n_results: int = 3) -> str:
    """Query the VPS RAG API and use local Ollama to generate answer"""

    # Call VPS API to get relevant chunks
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"question": question, "n_results": n_results},
            headers={"X-API-Key": API_KEY},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        return f"Error querying API: {str(e)}"

    # Extract chunks
    chunks = data.get("chunks", [])
    if not chunks:
        return "No relevant notes found."

    # Build context from chunks
    context = "\n\n---\n\n".join(chunks)

    # Use local Ollama to generate answer
    prompt = f"""Based on the following notes, answer the question.

Notes:
{context}

Question: {question}"""

    try:
        ollama_response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}]
        )
        return ollama_response["message"]["content"]
    except Exception as e:
        return f"Error calling Ollama: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python local_query.py \"Your question here\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"Question: {question}\n")
    print("Querying RAG...")
    answer = query_rag(question)
    print(f"\nAnswer:\n{answer}")
