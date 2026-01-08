import chromadb
import re
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from chromadb.errors import NotFoundError
from dotenv import load_dotenv

# Load embedding model (BGE-M3: best cross-lingual alignment, 567M params)
model = SentenceTransformer("BAAI/bge-m3")

load_dotenv()

# load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

# Configuration from environment variables
NOTES_PATH = os.getenv("NOTES_PATH", "./notes")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_data")


def load_notes(notes_dir: str) -> list[dict]:
    notes = []
    for path in Path(notes_dir).rglob("*.md"):
        # Skip hidden folders (those starting with .)
        if any(part.startswith(".") for part in path.parts):
            continue

        text = path.read_text()
        notes.append({"path": str(path), "content": text})
    return notes


def extract_title(text: str) -> str:
    if not text:
        return ""
    line = text.split("\n", 1)[0]
    return line.lstrip(" #")


def extract_tags(text: str) -> list[str]:
    """Extract tags from the last line if it matches :tag:tag: format."""
    lines = text.strip().split("\n")
    if not lines:
        return []
    last_line = lines[-1].strip()
    if re.match(r"^(:[\w-]+)+:$", last_line):
        return [t for t in last_line.split(":") if t]
    return []


def remove_tag_line(text: str) -> str:
    """Remove the tag line from text."""
    lines = text.strip().split("\n")
    if not lines:
        return text
    last_line = lines[-1].strip()
    if re.match(r"^(:[\w-]+)+:$", last_line):
        return "\n".join(lines[:-1])
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
    lines = text.split("\n")
    chunks = []
    current = ""

    for line in lines:
        is_item = line.strip().startswith("- ")
        if is_item and len(current) + len(line) > max_size and current:
            chunks.append(current.strip())
            current = ""
        current += line + "\n"

    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_code_block(text: str, max_size: int = 1500) -> list[str]:
    if len(text) <= max_size:
        return [text]

    lines = text.split("\n")
    header = lines[0]
    chunks = []
    current = header + "\n"

    for line in lines[1:-1]:
        if len(current) + len(line) > max_size and current != header + "\n":
            chunks.append(current.rstrip() + "\n```")
            current = header + "\n"
        current += line + "\n"

    if current != header + "\n":
        chunks.append(current.rstrip() + "\n```")

    return chunks


def is_list_block(text: str) -> bool:
    lines = [l for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return False
    list_lines = sum(1 for l in lines if l.strip().startswith("- "))
    return list_lines / len(lines) > 0.5


def chunk_section(text: str, max_size: int = 1500) -> list[str]:
    parts = re.split(r"(```[\s\S]*?```)", text)

    chunks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.startswith("```"):
            chunks.extend(chunk_code_block(part, max_size))
        elif len(part) <= max_size:
            chunks.append(part)
        elif is_list_block(part):
            chunks.extend(chunk_list(part, max_size))
        else:
            chunks.extend(chunk_text(part, chunk_size=500, overlap=50))

    return chunks


def chunk_markdown(text: str, max_size: int = 1500) -> list[str]:
    sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)

    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue

        if len(section) <= max_size:
            chunks.append(section)
        else:
            header_match = re.match(r"^(## .+)$", section, flags=re.MULTILINE)
            header = header_match.group(1) + "\n\n" if header_match else ""
            content = section[len(header) :].strip() if header else section

            for sub_chunk in chunk_section(content, max_size):
                chunks.append(header + sub_chunk if header else sub_chunk)

    return chunks


def chunk_notes(notes: list[dict]) -> list[dict]:
    chunks = []
    for note in notes:
        title = extract_title(note["content"])
        tags = extract_tags(note["content"])
        content = remove_tag_line(note["content"])
        tag_suffix = "\n\nTags: " + ", ".join(tags) if tags else ""
        stripped_path = note["path"][len(NOTES_PATH) :]

        for i, chunk in enumerate(chunk_markdown(content)):
            chunks.append(
                {
                    "id": f"{note['path']}_{i}",
                    "title": title,
                    "text": chunk + tag_suffix,
                    "source": stripped_path,
                    "tags": tags,
                }
            )
    return chunks


# Main
notes = load_notes(NOTES_PATH)
chunks = chunk_notes(notes)

client = chromadb.PersistentClient(path=CHROMA_PATH)
try:
    client.delete_collection("notes")
except NotFoundError:
    pass

collection = client.create_collection("notes")

# Batch embedding for efficiency
texts = [chunk["text"] for chunk in chunks]
embeddings = model.encode(texts, show_progress_bar=True)

for i, chunk in enumerate(chunks):
    collection.add(
        ids=[chunk["id"]],
        embeddings=[embeddings[i].tolist()],
        documents=[chunk["text"]],
        metadatas=[
            {
                "title": chunk["title"],
                "source": chunk["source"],
                "tags": ",".join(chunk["tags"]),
            }
        ],
    )

print(f"Indexed {len(chunks)} chunks from {len(notes)} notes")
