import asyncio
import os
from pydantic import BaseModel
from chromadb.config import Settings
import chromadb
from sentence_transformers import SentenceTransformer
import json
import ollama

# Configuration
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_data")

# Initialize
model = SentenceTransformer('all-MiniLM-L6-v2')

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

async def query(request: QueryRequest):
    """Query the notes RAG"""
    collection = client.get_collection("notes")

    # Generate embedding for question
    q_embedding = model.encode([request.question], convert_to_tensor=False)[0].tolist()


    # Query ChromaDB
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=request.n_results
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
    metadatas = metas[0] if metas and isinstance(metas[0], (list, tuple)) else list(metas)

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

question="what is a recipe for hachee?"
qr = QueryRequest(
    question=question,
)
ans = asyncio.run(query(qr))
print(ans)
print("---------------------")

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
- Do not include explanations or any other text.

<QUESTTION>
{question}
</QUESTION>

<JSON>
{ans}
</JSON>
"""

print(prompt)
print("---------------------")

ollama_response = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": prompt}]
)
ol_resp = ollama_response["message"]["content"]
print(ol_resp)
