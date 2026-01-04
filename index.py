import ollama
import chromadb
import re
import json
from pathlib import Path

with open("config.json") as f:
    config = json.load(f)

def load_notes(notes_dir: str) -> list[dict]:
    notes = []
    for path in Path(notes_dir).rglob("*.md"):
        text = path.read_text()
        notes.append({"path": str(path), "content": text})
    return notes

def extract_tags(text: str) -> list[str]:
    """Extract tags from the last line if it matches :tag:tag: format."""
    lines = text.strip().split('\n')
    if not lines:
        return []
    last_line = lines[-1].strip()
    if re.match(r'^(:[a-zA-Z0-9À-ÿ-]+)+:$', last_line):
        return [t for t in last_line.split(':') if t]
    return []

def remove_tag_line(text: str) -> str:
    """Remove the tag line from text."""
    lines = text.strip().split('\n')
    if not lines:
        return text
    last_line = lines[-1].strip()
    if re.match(r'^(:[a-zA-Z0-9À-ÿ-]+)+:$', last_line):
        return '\n'.join(lines[:-1])
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def chunk_list(text: str, max_size: int = 1500) -> list[str]:
    lines = text.split('\n')
    chunks = []
    current = ""
    
    for line in lines:
        is_item = line.strip().startswith('- ')
        if is_item and len(current) + len(line) > max_size and current:
            chunks.append(current.strip())
            current = ""
        current += line + '\n'
    
    if current.strip():
        chunks.append(current.strip())
    return chunks

def chunk_code_block(text: str, max_size: int = 1500) -> list[str]:
    if len(text) <= max_size:
        return [text]
    
    lines = text.split('\n')
    header = lines[0]
    chunks = []
    current = header + '\n'
    
    for line in lines[1:-1]:
        if len(current) + len(line) > max_size and current != header + '\n':
            chunks.append(current.rstrip() + '\n```')
            current = header + '\n'
        current += line + '\n'
    
    if current != header + '\n':
        chunks.append(current.rstrip() + '\n```')
    
    return chunks

def is_list_block(text: str) -> bool:
    lines = [l for l in text.strip().split('\n') if l.strip()]
    if not lines:
        return False
    list_lines = sum(1 for l in lines if l.strip().startswith('- '))
    return list_lines / len(lines) > 0.5

def chunk_section(text: str, max_size: int = 1500) -> list[str]:
    parts = re.split(r'(```[\s\S]*?```)', text)
    
    chunks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        if part.startswith('```'):
            chunks.extend(chunk_code_block(part, max_size))
        elif len(part) <= max_size:
            chunks.append(part)
        elif is_list_block(part):
            chunks.extend(chunk_list(part, max_size))
        else:
            chunks.extend(chunk_text(part, chunk_size=500, overlap=50))
    
    return chunks

def chunk_markdown(text: str, max_size: int = 1500) -> list[str]:
    sections = re.split(r'(?=^## )', text, flags=re.MULTILINE)
    
    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        
        if len(section) <= max_size:
            chunks.append(section)
        else:
            header_match = re.match(r'^(## .+)$', section, flags=re.MULTILINE)
            header = header_match.group(1) + '\n\n' if header_match else ""
            content = section[len(header):].strip() if header else section
            
            for sub_chunk in chunk_section(content, max_size):
                chunks.append(header + sub_chunk if header else sub_chunk)
    
    return chunks

def chunk_notes(notes: list[dict]) -> list[dict]:
    chunks = []
    for note in notes:
        tags = extract_tags(note["content"])
        content = remove_tag_line(note["content"])
        tag_suffix = "\n\nTags: " + ", ".join(tags) if tags else ""
        
        for i, chunk in enumerate(chunk_markdown(content)):
            chunks.append({
                "id": f"{note['path']}_{i}",
                "text": chunk + tag_suffix,
                "source": note["path"],
                "tags": tags
            })
    return chunks

# Main
notes = load_notes(config["notes_path"])
chunks = chunk_notes(notes)

client = chromadb.PersistentClient(path="./chroma_data")
try:
    client.delete_collection("notes")
except ValueError:
    pass
collection = client.create_collection("notes")

for chunk in chunks:
    response = ollama.embed(model="nomic-embed-text", input=chunk["text"])
    collection.add(
        ids=[chunk["id"]],
        embeddings=[response["embeddings"][0]],
        documents=[chunk["text"]],
        metadatas=[{"source": chunk["source"], "tags": ",".join(chunk["tags"])}]
    )

print(f"Indexed {len(chunks)} chunks from {len(notes)} notes")
