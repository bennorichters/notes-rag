import os
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer

# Configuration
API_KEY = os.getenv("API_KEY")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_data")

if not API_KEY:
    raise ValueError("API_KEY environment variable must be set")

# Initialize
app = FastAPI(title="Notes RAG API", version="1.0.0")
model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path=CHROMA_PATH)

# API Key authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Request/Response models
class QueryRequest(BaseModel):
    question: str
    n_results: int = 3

class QueryItem(BaseModel):
    chunk: str
    source: str
    tags: list[str]

@app.get("/health")
async def health():
    """Health check endpoint (no auth required)"""
    return {"status": "healthy"}

@app.post("/query", response_model=list[QueryItem])
async def query(request: QueryRequest, api_key: str = Depends(verify_api_key)):
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

#if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
