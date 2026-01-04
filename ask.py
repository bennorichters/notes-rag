import ollama
import chromadb

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_collection("notes")

def ask(question: str, n_results: int = 3) -> str:
    q_embedding = ollama.embed(model="nomic-embed-text", input=question)["embeddings"][0]
    results = collection.query(query_embeddings=[q_embedding], n_results=n_results)

    docs = results["documents"]
    if docs:
        context = "\n\n---\n\n".join(docs[0])
    else:
        context = ""
        
    prompt = f"""Based on the following notes, answer the question.

Notes:
{context}

Question: {question}"""
    
    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

print(ask("Ask you question here"))
