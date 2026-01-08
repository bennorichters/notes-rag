import os
import re
from chromadb.config import Settings
import chromadb
from sentence_transformers import SentenceTransformer
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
def query_chromadb(question: str, n_results: int = 8) -> list[dict]:
    """Query ChromaDB with the question directly and return results with distances."""
    collection = client.get_collection("notes")

    # Embed the question directly (no keyword extraction)
    q_embedding = model.encode([question], convert_to_tensor=False)[0].tolist()

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
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

    dists = results.get("distances")
    if not isinstance(dists, (list, tuple)):
        dists = []
    distances = (
        dists[0] if dists and isinstance(dists[0], (list, tuple)) else list(dists)
    )

    def _as_tags(value) -> list[str]:
        if isinstance(value, list):
            return [str(t).strip() for t in value if str(t).strip()]
        if isinstance(value, str):
            return [t.strip() for t in value.split(",") if t.strip()]
        return []

    items = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        items.append(
            {
                "chunk": doc,
                "title": str(meta.get("title", "")),
                "source": str(meta.get("source", "")),
                "tags": _as_tags(meta.get("tags", [])),
                "distance": float(dist),
            }
        )

    return items


def rerank_with_llm(question: str, candidates: list[dict]) -> int:
    """Use Ollama to pick the most relevant document from candidates.

    Returns the index of the best matching candidate.
    """
    # Build candidate list for the prompt
    candidate_descriptions = []
    for i, c in enumerate(candidates):
        tags_str = ", ".join(c["tags"]) if c["tags"] else "none"
        # Show a preview of the chunk (first 200 chars)
        chunk_preview = c["chunk"][:200].replace("\n", " ")
        if len(c["chunk"]) > 200:
            chunk_preview += "..."
        candidate_descriptions.append(
            f"[{i}] Title: {c['title']}\n    Tags: {tags_str}\n    Preview: {chunk_preview}"
        )

    candidates_text = "\n\n".join(candidate_descriptions)

    prompt = f"""You are a document retrieval assistant. Given a question and a list of candidate documents, determine which document is most likely to contain the answer.

<QUESTION>
{question}
</QUESTION>

<CANDIDATES>
{candidates_text}
</CANDIDATES>

Task:
- Analyze each candidate's title, tags, and content preview
- Select the ONE document most likely to answer the question
- Consider semantic relevance, not just keyword matching

Output:
- Return ONLY the number in brackets (e.g., 0, 1, 2) of the best matching document
- Do not explain your reasoning
- If none seem relevant, return 0"""

    ollama_response = ollama.chat(
        model="llama3.1",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )

    response_text = ollama_response["message"]["content"].strip()

    # Parse the response - extract the number
    try:
        # Handle responses like "0", "[0]", "Document 0", etc.
        match = re.search(r'\d+', response_text)
        if match:
            idx = int(match.group())
            if 0 <= idx < len(candidates):
                return idx
    except (ValueError, AttributeError):
        pass

    # Default to first result if parsing fails
    return 0


def check_confidence(results: list[dict], threshold: float = 1.0) -> tuple[bool, float]:
    """Check if the top result has sufficient confidence based on distance.

    Lower distance = better match. Returns (is_confident, best_distance).
    """
    if not results:
        return False, float('inf')

    best_distance = results[0]["distance"]
    return best_distance <= threshold, best_distance


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

    # Query RAG directly with the question (no keyword extraction)
    print("Searching notes...")
    results = query_chromadb(question)

    if not results:
        print("No results found in your notes.")
        return

    # Check confidence based on distance scores
    is_confident, best_distance = check_confidence(results)
    print(f"Best match distance: {best_distance:.3f}", end="")
    if not is_confident:
        print(" (low confidence - answer may not be in your notes)")
    else:
        print()

    # Show candidates
    print("\n--- Top candidates ---")
    for i, r in enumerate(results[:5]):
        print(f"[{i}] {r['source']}: {r['title']} (dist: {r['distance']:.3f})")
    print("----------------------\n")

    # LLM re-ranking: let Ollama pick the best document
    print("Re-ranking candidates...")
    best_idx = rerank_with_llm(question, results)
    selected = results[best_idx]
    print(f"Selected: [{best_idx}] {selected['source']}: {selected['title']}")

    # Retrieve full document content
    document_content = get_document_content(selected["source"])

    # Get final answer from Ollama
    print("\nGenerating answer...")
    answer = get_final_answer(question, document_content)

    print("\n" + "=" * 50)
    print("Answer:")
    print("=" * 50)
    print(answer)


if __name__ == "__main__":
    main()
