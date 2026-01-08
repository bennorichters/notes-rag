import os
from pydantic import BaseModel
from chromadb.config import Settings
import chromadb
from sentence_transformers import SentenceTransformer
import json
import ollama
from dotenv import load_dotenv

# Configuration
load_dotenv()
NOTES_PATH = os.getenv("NOTES_PATH", "./notes")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_data")

# Initialize
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=Settings(anonymized_telemetry=False),
)


# Request/Response models
class QueryRequest(BaseModel):
    question: str
    n_results: int = 3


class QueryItem(BaseModel):
    chunk: str
    title: str
    source: str
    tags: list[str]


# Helper functions
def strip_surrounding_quotes(s: str) -> str:
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s


def strip_leading_slash(s: str) -> str:
    if len(s) >= 1 and s[0] == "/":
        return s[1:]
    return s


def full_notes_path(source: str) -> str:
    return os.path.join(NOTES_PATH, source)


def read_file_contents(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def print_sources_and_titles(data):
    for item in data:
        print(item.get("source") + ": " + item.get("title"))


# Main workflow functions
def query_chromadb(keywords: str, n_results: int = 3) -> list[dict]:
    """Query ChromaDB with keywords and return results."""
    collection = client.get_collection("notes")

    # Generate embedding for keywords
    q_embedding = model.encode([keywords], convert_to_tensor=False)[0].tolist()

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[q_embedding], n_results=n_results
    )

    # Extract results
    docs = results.get("documents")
    if not isinstance(docs, (list, tuple)):
        docs = []
    documents = docs[0] if docs and isinstance(docs[0], (list, tuple)) else list(docs)

    metas = results.get("metadatas")
    if not isinstance(metas, (list, tuple)):
        metas = []
    metadatas = (
        metas[0] if metas and isinstance(metas[0], (list, tuple)) else list(metas)
    )

    def _as_tags(value) -> list[str]:
        if isinstance(value, list):
            return [str(t).strip() for t in value if str(t).strip()]
        if isinstance(value, str):
            return [t.strip() for t in value.split(",") if t.strip()]
        return []

    items = []
    for doc, meta in zip(documents, metadatas):
        items.append(
            {
                "chunk": doc,
                "title": str(meta.get("title", "")),
                "source": str(meta.get("source", "")),
                "tags": _as_tags(meta.get("tags", [])),
            }
        )

    return items


def extract_keywords(question: str) -> str:
    """Use Ollama to extract keywords from the question."""
    prompt = f"""You are given a question.

Task:
Determine which keywords are used to query a RAG to retrieve documents with the most relevant information.

Rules:
- Select words from the question that are most likely to give relevant results from the RAG.
- You are allowed to fix user typos in your response.
- You are allowed to add words to your response that are not used in the question.

Output:
- Return a list of key words
- Your result is a space separated list of words
- Do not respond with anything else but the list of keywords
- Do not mention that you fixed typos

<QUESTION>
{question}
</QUESTION>
"""

    ollama_response = ollama.chat(
        model="llama3.1",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    return ollama_response["message"]["content"]


def get_document_content(source: str) -> str:
    """Retrieve the full content of a document from its source path."""
    cleaned_source = strip_leading_slash(strip_surrounding_quotes(source))
    notes_file = full_notes_path(cleaned_source)
    return read_file_contents(notes_file)


def get_final_answer(question: str, document_content: str) -> str:
    """Use Ollama to extract the answer from the document."""
    prompt = f"""Task: High-Fidelity Information Extraction

- Role: You are an objective and precise Research Assistant.
- Goal: Answer the `<QUESTION>` using **only** the content provided in the `<DOCUMENT>`.

---

Strict Guidelines:
1. **Groundedness:** Treat the `<DOCUMENT>` as the absolute source of truth. Do not use prior knowledge, external facts, or assumptions. If the document does not contain the answer, respond with: "The provided document does not contain sufficient information to answer this question."
2. **Output Structure:** - Use **Markdown** for clarity (headers, bullet points, or tables where appropriate).
   - If the information is a process, use a numbered list.
   - If the information is a list of items or facts, use bullet points.
3. **No Meta-Talk:** Do not include introductory phrases (e.g., "According to the document...") or concluding remarks. Provide only the extracted data.
4. **Verbatim Accuracy:** Retain specific terminology, technical names, dates, and figures exactly as they appear in the source text.
5. **Conflict Resolution:** If the document contains internal contradictions, report exactly what the text states without trying to resolve the discrepancy.

---

<QUESTION>
{question}
</QUESTION>

<DOCUMENT>
{document_content}
<DOCUMENT>
"""

    ollama_response = ollama.chat(
        model="llama3.1",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    return ollama_response["message"]["content"]


def main():
    """Main execution flow."""
    # Get question from user
    print("Ask a question about your notes")
    question = input()

    # Extract keywords using Ollama
    keywords = extract_keywords(question)
    print("Keywords: " + keywords)

    # Query RAG with keywords
    results = query_chromadb(keywords)
    print("---RAG result")
    print_sources_and_titles(results)
    print("---RAG result")

    # Get the first result's source
    source = results[0]["source"]
    print("source: " + source)

    # Retrieve full document content
    document_content = get_document_content(source)

    # Get final answer from Ollama
    print("Generating answer...")
    answer = get_final_answer(question, document_content)

    print("Final answer:")
    print(answer)


if __name__ == "__main__":
    main()
