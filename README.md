# CRC Literature Dialogue

## What This Project Does

This tool runs a structured, evidence-grounded dialogue between two AI agents who debate colorectal cancer research literature. The goal is to act as a **literature triage engine** — surfacing the strongest findings, filtering out low-quality evidence, and producing actionable research hypotheses from a large body of publications.

Rather than reading thousands of papers manually, you provide a research question and the two agents deliberate across the literature, challenging each other's claims, flagging weak evidence, and converging on testable hypotheses through a structured consensus mechanism.

The long-term goal is to run this against the full PubMed colorectal cancer corpus (107,000+ papers, 2015–2025) so that every agent response is grounded in retrieved, citable literature — not just the model's pre-trained knowledge.

---

## The Two Agents

**Agent A — Dr. Alex Chen (Clinical Oncologist)**
Reasons from clinical trial data, OS/PFS endpoints, response rates, and real-world applicability. Challenges mechanistic claims that don't translate to patient outcomes.

**Agent B — Dr. Sam Rivera (Translational Researcher)**
Reasons from tumor microenvironment biology, signaling pathways, and resistance mechanisms. Pushes back when clinical framing oversimplifies the underlying biology.

Both agents are constrained to four research lanes:
1. MSI-H/dMMR and immunotherapy response and resistance
2. RAS/RAF mutation status and anti-EGFR or BRAF-targeted therapy selection
3. Tumor microenvironment and mechanisms of treatment resistance
4. ctDNA and liquid biopsy for MRD detection and surveillance

Every claim must be labeled with an evidence level (`[RCT]`, `[META]`, `[COHORT]`, `[PRECLINICAL]`, `[EXPERT]`). Uncited claims are flagged as `INFERENCE:` and cannot support a final verdict. Every 4 turns, a consensus checkpoint fires and forces both agents to produce a `VERDICT` — a single testable hypothesis — before the conversation continues.

---

## How It Works (Pipeline Overview)

```
PubMed API
    └── build_rag.py fetches abstracts → saves to pubmed_cache.json
            └── embeds abstracts as vectors → saves to chroma_db/
                    └── serve_rag.py loads vectors into RAM, runs local server
                            └── crc_dialogue.html queries server each agent turn
                                    └── relevant abstracts injected into agent context
                                            └── agents respond citing real PMIDs
```

At query time, the search is semantic — not keyword-based. The agent's recent conversation is converted into a vector and compared against all stored abstract vectors. The 5 most similar abstracts are retrieved and injected into the agent's context window before it generates a response.

---

## File Structure

```
Colo/
│
├── README.md                  ← this file
│
├── crc_dialogue.html          ← the app; open directly in Chrome or Edge
│
├── rag/
│   ├── requirements.txt       ← Python package dependencies
│   ├── build_rag.py           ← scrapes PubMed, embeds abstracts, builds vector store
│   └── serve_rag.py           ← local API server; must run while using the app
│
└── data/
    ├── pubmed_cache.json      ← raw abstract cache (checkpoint file for resuming)
    └── chroma_db/             ← ChromaDB vector store (vectors + metadata on disk)
```

---

## File Reference

### `crc_dialogue.html`
The entire front-end application. Open directly in a browser — no installation required. Contains:
- API key field (pre-filled) for the Anthropic API
- Two editable persona panels side by side (Agent A and Agent B system prompts)
- Controls bar: opening topic input, turn count, Start / Next Turn / Stop, RAG toggle, Literature drawer, Export, Clear
- Chat window where the agent dialogue streams in real time
- Inject bar at the bottom to steer the conversation mid-session
- Export button that saves the full transcript as a `.txt` file

The RAG toggle connects to `serve_rag.py` running locally. When enabled, each agent turn triggers a semantic search of the vector store before the Anthropic API call is made, injecting the top 5 most relevant abstracts into the agent's context.

---

### `rag/requirements.txt`
List of Python packages required to run the RAG pipeline. Install once with:
```
py -m pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| `biopython` | Python wrapper for NCBI's PubMed API |
| `chromadb` | Local vector database — stores and searches embeddings |
| `sentence-transformers` | Downloads and runs the text embedding model |
| `fastapi` | Web framework for the local query server |
| `uvicorn` | ASGI server that runs FastAPI |
| `tqdm` | Progress bars in the terminal |

---

### `rag/build_rag.py`
Runs once (or overnight) to build the vector store. Does three things in sequence:

1. **Search** — calls NCBI's E-utilities API with the MeSH term `"colorectal neoplasms"` scoped to 2015–2025. Registers the query on NCBI's server-side history to enable unlimited paging (bypasses the 10,000 ID cap).

2. **Fetch** — downloads abstracts from PubMed in batches of 500. Saves progress to `data/pubmed_cache.json` after every batch so interrupted runs can resume. With an NCBI API key, fetches at 10 requests/sec; without one, 3 requests/sec.

3. **Embed + Store** — converts each abstract (title + abstract text concatenated) into a 384-number vector using `all-MiniLM-L6-v2`, then stores the vector alongside the original text and metadata (PMID, title, authors, year, journal) in ChromaDB.

**Commands:**
```bash
# Test with 500 papers first
py build_rag.py --limit 500

# Full build (run overnight — 107,000+ papers)
py build_rag.py

# Resume after interruption
py build_rag.py --resume
```

---

### `rag/serve_rag.py`
Lightweight FastAPI server that must be running in a terminal whenever the HTML app is open with RAG enabled. Loads the embedding model and ChromaDB collection into RAM at startup, then exposes two endpoints:

- `GET /health` — liveness check; returns paper count. Called when you click the RAG button in the browser.
- `POST /search` — accepts a query string, converts it to a vector, finds the top-k nearest abstracts in ChromaDB by cosine similarity, and returns them as JSON. Called automatically before each agent turn.

The server only listens on `127.0.0.1` (localhost) and is not accessible from outside your machine.

**Command:**
```bash
py serve_rag.py
```

---

### `data/pubmed_cache.json`
A flat JSON file keyed by PMID containing every downloaded abstract and its metadata. Acts as a checkpoint — if `build_rag.py` is interrupted, the next run reads this file and skips already-fetched papers. Also allows re-embedding without re-downloading from PubMed.

---

### `data/chroma_db/`
ChromaDB's on-disk storage. Contains binary index files and a SQLite database storing the vector embeddings, original abstract text, and metadata. Managed entirely by ChromaDB — do not edit manually. This folder grows to approximately 500MB–1GB for the full 107k paper corpus.

---

## Setup and Usage

### First-time setup
```bash
# 1. Install Python dependencies
cd "path/to/Colo/rag"
py -m pip install -r requirements.txt

# 2. Test the scraper and embedder (500 papers)
py build_rag.py --limit 500

# 3. Run the full build overnight
py build_rag.py
```

### Every session
```bash
# Terminal 1 — start the RAG server (leave running)
cd "path/to/Colo/rag"
py serve_rag.py

# Then open crc_dialogue.html in Chrome or Edge
# Click RAG Off to connect → should flip green with paper count
# Enter a topic, set turn count, hit Start
```

### If the build is interrupted
```bash
py build_rag.py --resume
```

---

## Design Notes

**Why RAG instead of pasting abstracts manually?**
Pasting the full literature corpus into every API call would cost ~$1 per agent turn for 70k papers. RAG retrieves only the 5 most relevant abstracts per turn (~1,500 tokens), reducing cost by ~99% while keeping responses grounded in real citations.

**Why a local vector store instead of a cloud service?**
ChromaDB runs entirely on your machine. No data leaves your system, no API costs for embeddings at query time, and no dependency on external infrastructure.

**Why two agents instead of one?**
A single agent asked to summarize literature will produce consensus. Two agents with adversarial framings will surface contradictions, gaps, and competing interpretations — which is where the actual research signal lives. The DISAGREE: mechanism and evidence hierarchy rules are designed to prevent the agents from collapsing into agreement prematurely.

**On measuring RAG influence vs. pre-trained knowledge**
The model has likely seen a significant portion of pre-2024 PubMed during training. RAG becomes most valuable for 2024–2025 papers that postdate the training cutoff. A future citation provenance audit — checking whether cited PMIDs exist in the local database — would quantify how much the agents are drawing from retrieved literature vs. internalized knowledge.
