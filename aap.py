import asyncio
import os
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from chromadb.config import Settings
import chromadb
from sentence_transformers import SentenceTransformer

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
    try:
        collection = client.get_collection("notes")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Collection not found. Has indexing been run? Error: {str(e)}")

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

    return items

qr = QueryRequest(
    question="what is a recipe for hachee?",
)
ans = asyncio.run(query(qr))
print(ans)
