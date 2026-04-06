"""
serve_rag.py
────────────
Lightweight local web server that lets the browser app query the ChromaDB
vector store built by build_rag.py. Must be running in a terminal whenever
you use crc_dialogue.html with RAG enabled.

Usage
─────
  py serve_rag.py

  Then open crc_dialogue.html and click the RAG Off button to connect.
  Server listens on http://127.0.0.1:8000 (localhost only — not exposed to internet)

Endpoints
─────────
  GET  /health   → confirms server is alive and returns paper count
  POST /search   → accepts a query string, returns top-k relevant abstracts
"""

# ── Imports ───────────────────────────────────────────────────────────────────
# pathlib      — cross-platform file path construction
from pathlib import Path

# chromadb     — reads the vector store from disk built by build_rag.py
import chromadb

# os — reads environment variables
import os

# FastAPI      — the web framework that handles HTTP requests from the browser
# HTTPException — used to return clean error messages with proper HTTP status codes
from fastapi import FastAPI, HTTPException

# CORSMiddleware — allows the browser to call this local server from an HTML file
# opened directly from disk (file:// protocol), which browsers normally block
from fastapi.middleware.cors import CORSMiddleware

# BaseModel    — Pydantic class that validates and types request/response JSON
from pydantic import BaseModel

# SentenceTransformer — same embedding model used in build_rag.py
# Must match exactly so query vectors are in the same vector space as stored vectors
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
# Must match the paths and model name used in build_rag.py

CHROMA_DIR  = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION  = "crc_abstracts"
EMBED_MODEL = "all-MiniLM-L6-v2"  # Must be identical to build_rag.py — mismatched models
                                    # produce incompatible vectors and garbage search results
DEFAULT_K   = 5  # Number of abstracts returned per search query


# ── Load .env file ───────────────────────────────────────────────────────────
# Reads Colo/.env and loads each KEY=VALUE line into environment variables.
# This keeps secrets out of source code and out of git.
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="CRC RAG Server")

# CORS (Cross-Origin Resource Sharing) — browsers block JavaScript from calling
# a different origin than the page itself. Since crc_dialogue.html is opened as
# a local file (file://) and this server is at http://127.0.0.1:8000, they are
# different origins. This middleware tells the browser to allow those requests.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Allow any origin — safe here because server only
    allow_methods=["*"],   # listens on localhost and is not exposed to the internet
    allow_headers=["*"],
)

# ── Load model and database at startup ───────────────────────────────────────
# These are loaded once when the server starts and stay in RAM the entire time
# the server is running. This is why you keep the terminal open — closing it
# unloads both the model and the database connection.

print(f"Loading embedding model: {EMBED_MODEL}")
_model = SentenceTransformer(EMBED_MODEL)  # ~90MB neural network loaded into RAM

# Verify the ChromaDB folder exists before trying to open it
if not CHROMA_DIR.exists():
    raise RuntimeError(
        f"ChromaDB not found at {CHROMA_DIR}\n"
        "Run build_rag.py first to build the vector store."
    )

# PersistentClient reads from disk — the actual vectors stay on disk and are
# paged into RAM as needed during queries, not all at once
_client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection = _client.get_collection(COLLECTION)
print(f"Loaded collection '{COLLECTION}' with {_collection.count():,} records.")


# ── Request / response models ─────────────────────────────────────────────────
# Pydantic models define the shape of JSON going in and out of each endpoint.
# FastAPI automatically validates incoming requests against these and returns
# a 422 error if the shape doesn't match.

class SearchRequest(BaseModel):
    query:     str        # Plain English search query derived from the agent conversation
    n_results: int = DEFAULT_K  # How many abstracts to return (defaults to 5)


class SearchResult(BaseModel):
    pmid:     str    # PubMed ID — agents cite this in their responses
    title:    str
    abstract: str    # Full abstract text — injected into the agent's context window
    authors:  str
    year:     str
    journal:  str
    score:    float  # Cosine distance — lower means more semantically similar to the query


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """
    Simple liveness check. The HTML app calls this when you click RAG Off
    to confirm the server is reachable before enabling RAG mode. Returns
    the paper count so the button can display "RAG On (9,489 papers)".
    """
    return {"status": "ok", "count": _collection.count()}


@app.get("/config")
def config():
    """
    Serves the Anthropic API key from the local .env file to the browser app.
    Only accessible from localhost — never exposed to the internet.
    The HTML fetches this once on load so the key never needs to be hardcoded
    in the HTML file or entered manually each session.
    """
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise HTTPException(status_code=404, detail="ANTHROPIC_API_KEY not found in .env")
    return {"anthropic_api_key": key}


@app.post("/search", response_model=list[SearchResult])
def search(req: SearchRequest):
    """
    Core retrieval endpoint. Called once per agent turn before the Anthropic
    API call is made. Pipeline:
      1. Convert the query string into a 384-number vector using the same
         embedding model that was used to embed all abstracts at build time
      2. ChromaDB compares that vector against every stored abstract vector
         using cosine distance (angle between vectors in 384-dimensional space)
      3. Returns the n_results closest matches — their original abstract text
         and metadata — which get injected into the agent's system prompt
    The whole round trip takes ~200-400ms.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Step 1: embed the query — produces a single 384-number vector
    embedding = _model.encode([req.query]).tolist()

    # Step 2: vector similarity search across all stored abstracts
    # include= specifies what to return alongside the IDs:
    #   documents = the original abstract text stored at embed time
    #   metadatas = title, authors, year, journal
    #   distances = cosine distance scores (0 = identical, 2 = opposite)
    results = _collection.query(
        query_embeddings=embedding,
        n_results=min(req.n_results, _collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    # Step 3: package results into clean response objects
    output = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        output.append(SearchResult(
            pmid     = meta.get("pmid", ""),
            title    = meta.get("title", ""),
            abstract = results["documents"][0][i],  # Original text, not the vector
            authors  = meta.get("authors", ""),
            year     = meta.get("year", ""),
            journal  = meta.get("journal", ""),
            score    = round(results["distances"][0][i], 4),
        ))

    return output


# ── Entry point ───────────────────────────────────────────────────────────────
# uvicorn is the ASGI server that actually runs FastAPI and listens for HTTP
# requests. host="127.0.0.1" means localhost only — the server cannot be
# reached from outside your machine.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
