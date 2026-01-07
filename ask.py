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
model = SentenceTransformer("all-MiniLM-L6-v2")

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
    source: str
    tags: list[str]


def query(request: QueryRequest):
    """Query the notes RAG"""
    collection = client.get_collection("notes")

    # Generate embedding for question
    q_embedding = model.encode([request.question], convert_to_tensor=False)[0].tolist()

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[q_embedding], n_results=request.n_results
    )

    # Extract results
    # Safely handle missing/empty keys and list-or-list-of-lists results
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

    items: list[QueryItem] = []
    for doc, meta in zip(documents, metadatas):
        items.append(
            QueryItem(
                chunk=doc,
                source=str(meta.get("source", "")),
                tags=_as_tags(meta.get("tags", [])),
            )
        )

    # return items
    return json.dumps(
        [{"chunk": it.chunk, "source": it.source, "tags": it.tags} for it in items],
        ensure_ascii=False,
    )


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


question = "Give a recipe for hachee"
qr = QueryRequest(
    question=question,
)
ans = query(qr)
print("---RAG result")
print(ans)
print("---RAG result")
prompt = f"""You are given:
- A question.
- A JSON array of items.  
  Each item contains:
  - `chunk`: an excerpt that may contain relevant information.
  - `source`: the identifier of the full document from which the chunk was taken.

Task:
Determine which single item’s `chunk` is most useful for answering the question.

Rules:
- Select the item whose `chunk` most directly and substantially helps answer the question.
- If a `chunk` only contains a title or metadata but clearly indicates the full document is about the question topic, select that item.
- If none of the chunks are relevant, answer with [NONE].
- Do not combine multiple items.
- Do not infer beyond the given chunks.

Output:
- Return only the exact `source` value of the selected item, or [NONE].
- Do not wrap your reponse in quotes, braces, brackets or anything else.
- Do not include explanations or any other text.

<QUESTTION>
{question}
</QUESTION>

<JSON>
{ans}
</JSON>
"""

print("---- sending 1st prompt")
print(prompt)
print("---- sending 1st prompt")

ollama_response = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": prompt}],
    options={"temperature": 0},
)
ol_resp = ollama_response["message"]["content"]

print("1st response: " + ol_resp)

source = strip_leading_slash(strip_surrounding_quotes(ol_resp))
notes_file = full_notes_path(source)
contents = read_file_contents(notes_file)

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
{contents}
<DOCUMENT>
"""

print("sending 2nd prompt")

ollama_response = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": prompt}],
    options={"temperature": 0},
)
ol_resp = ollama_response["message"]["content"]

print("Final answer:")
print(ol_resp)
