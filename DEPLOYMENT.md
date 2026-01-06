# Notes RAG Deployment Guide

This guide covers deploying your Notes RAG system to Dokku on your VPS and querying it from your local laptop.

## Architecture Overview

- **VPS (Dokku)**: Runs the FastAPI service that queries ChromaDB and returns relevant chunks
- **Local Laptop**: Runs queries using local Ollama for final answer generation
- **Security**: API key authentication over HTTPS

## Prerequisites

- Dokku installed on your VPS
- Notes directory at `/var/lib/dokku/data/storage/notes/` on VPS
- Ollama installed on your local laptop
- Python 3.10+ on both VPS and local laptop

## VPS Setup (Dokku)

### 1. Create the Dokku App

```bash
# On your VPS
dokku apps:create notes-rag
```

### 2. Configure Storage Mounts

Mount your notes directory into the container:

```bash
# Create persistent storage for ChromaDB data
dokku storage:ensure-directory notes-rag-chroma

# Mount notes directory (read-only)
dokku storage:mount notes-rag /var/lib/dokku/data/storage/notes:/app/notes:ro

# Mount ChromaDB data directory (read-write)
dokku storage:mount notes-rag /var/lib/dokku/data/storage/notes-rag-chroma:/app/chroma_data
```

### 3. Set Environment Variables

Generate a strong API key and configure the app:

```bash
# Generate a random API key (or use your own)
API_KEY=$(openssl rand -hex 32)

# Set environment variables
dokku config:set notes-rag \
  API_KEY="$API_KEY" \
  NOTES_PATH="/app/notes" \
  CHROMA_PATH="/app/chroma_data"

# Save your API key somewhere safe - you'll need it for local queries!
echo "Your API Key: $API_KEY"
```

### 4. Enable HTTPS (Let's Encrypt)

```bash
# Install Let's Encrypt plugin (if not already installed)
sudo dokku plugin:install https://github.com/dokku/dokku-letsencrypt.git

# Set your email for Let's Encrypt
dokku letsencrypt:set notes-rag email your-email@example.com

# Set your domain
dokku domains:set notes-rag notes-rag.yourdomain.com

# Enable HTTPS
dokku letsencrypt:enable notes-rag

# Auto-renew certificates
dokku letsencrypt:cron-job --add notes-rag
```

### 5. Deploy the Application

```bash
# On your local machine (where you cloned this repo)
git remote add dokku dokku@your-vps-ip:notes-rag

# Push to deploy
git push dokku claude/rag-markdown-notes-nIONE:main
```

### 6. Build the RAG Index

After deployment, SSH into your VPS and run the indexing script:

```bash
# SSH to VPS
ssh your-vps

# Run the indexing script in the Dokku container
dokku run notes-rag python index.py
```

This will:
- Read all `.md` files from your notes directory
- Chunk them intelligently
- Generate embeddings using sentence-transformers
- Store everything in ChromaDB

**Schedule periodic re-indexing** (optional):

```bash
# Add a cron job to rebuild the index daily at 2 AM
dokku cron:set notes-rag "0 2 * * * dokku run notes-rag python index.py"
```

Or use Dokku's built-in scheduler:

```bash
# Create a scheduled task
dokku scheduler-docker-local:set notes-rag DOKKU_SCHEDULER_RUN="0 2 * * * python index.py"
```

### 7. Verify Deployment

```bash
# Check if the API is running
curl https://notes-rag.yourdomain.com/health

# Should return: {"status":"healthy"}
```

## Local Laptop Setup

### 1. Install Dependencies

```bash
# On your local laptop
pip install -r requirements-local.txt
```

### 2. Configure Environment Variables

Create a `.env` file in your local project directory:

```bash
# Copy the example
cp .env.local.example .env

# Edit with your values
API_URL=https://notes-rag.yourdomain.com
API_KEY=your-secret-api-key-from-step-3
```

Or export them directly:

```bash
export API_URL=https://notes-rag.yourdomain.com
export API_KEY=your-secret-api-key-from-step-3
```

### 3. Ensure Ollama is Running

Make sure Ollama is installed and running on your laptop:

```bash
# Check Ollama is running
ollama list

# Pull the model if needed
ollama pull llama3.2
```

### 4. Query Your Notes

```bash
# Run a query
python local_query.py "What are my notes about Python?"

# Or make it executable
chmod +x local_query.py
./local_query.py "What are my notes about Docker?"
```

## Usage

### Querying from Python

You can also import and use the query function in your own scripts:

```python
from local_query import query_rag

answer = query_rag("What are my notes about AI?", n_results=5)
print(answer)
```

### Rebuilding the Index

Whenever you add, modify, or delete notes on your VPS:

```bash
# SSH to VPS
ssh your-vps

# Rebuild the index
dokku run notes-rag python index.py
```

Or wait for the scheduled cron job to run (if configured).

## Troubleshooting

### API returns "Collection not found"

The index hasn't been built yet. Run:

```bash
dokku run notes-rag python index.py
```

### "Invalid API Key" error

Double-check your API key matches on both server and local:

```bash
# On VPS - check server API key
dokku config:get notes-rag API_KEY

# On laptop - check your .env or environment variable
echo $API_KEY
```

### Timeouts or connection errors

- Verify the API is running: `dokku ps:report notes-rag`
- Check logs: `dokku logs notes-rag`
- Ensure your domain DNS is pointing to your VPS
- Verify HTTPS is working: `dokku letsencrypt:list`

### No relevant notes found

- Verify notes are being indexed: Check the output of `python index.py`
- Ensure notes directory is properly mounted
- Check notes are in `.md` format

## Security Notes

- **API Key**: Keep your API key secret. Don't commit it to git.
- **HTTPS**: Always use HTTPS in production. Let's Encrypt is free and automatic.
- **Firewall**: Ensure your VPS firewall allows HTTPS traffic (port 443)
- **Rotation**: To rotate your API key:
  ```bash
  # Generate new key
  NEW_KEY=$(openssl rand -hex 32)

  # Update on server
  dokku config:set notes-rag API_KEY="$NEW_KEY"

  # Update on local laptop (in .env file)
  ```

## Architecture Details

### How It Works

1. **Indexing** (on VPS):
   - `index.py` reads your markdown notes
   - Chunks them intelligently (respecting sections, code blocks, lists)
   - Generates embeddings using `sentence-transformers` (all-MiniLM-L6-v2)
   - Stores embeddings + text in ChromaDB

2. **Querying** (from laptop):
   - `local_query.py` sends your question to VPS API
   - VPS API embeds the question using the same model
   - VPS queries ChromaDB for most relevant chunks
   - VPS returns chunks to your laptop
   - Your local Ollama generates the final answer using the chunks as context

### Why This Architecture?

- **Notes on VPS**: Your notes live where you edit them
- **RAG on VPS**: Build index where notes are, no syncing needed
- **Ollama local**: Keep your powerful LLM local, no need to run on VPS
- **API-based**: Flexible, can add web UI or mobile app later
- **Secure**: API key + HTTPS protects your notes

## Customization

### Change the Embedding Model

Edit both `index.py` and `api.py`:

```python
# Replace 'all-MiniLM-L6-v2' with your preferred model
model = SentenceTransformer('your-model-name')
```

**Important**: Both must use the same model!

### Change the LLM Model

Edit `local_query.py`:

```python
ollama_response = ollama.chat(
    model="llama3.2",  # Change to your preferred model
    messages=[{"role": "user", "content": prompt}]
)
```

### Adjust Chunk Sizes

Edit `index.py` - see `chunk_markdown()`, `chunk_section()`, etc.

Default max chunk size is 1500 characters. Adjust the `max_size` parameter.

## Next Steps

- Set up automatic backups of `/var/lib/dokku/data/storage/notes-rag-chroma`
- Create a web UI for querying (optional)
- Add more endpoints (list sources, filter by tags, etc.)
- Set up monitoring/alerts for the API

## Support

For issues with:
- Dokku: https://dokku.com/docs/
- ChromaDB: https://docs.trychroma.com/
- Sentence Transformers: https://www.sbert.net/
- Ollama: https://ollama.ai/
