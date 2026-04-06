# CRC Literature Dialogue

## What This Project Does

This tool runs a structured, evidence-grounded dialogue between two AI agents who debate colorectal cancer research literature. The goal is to act as a **literature triage engine** — surfacing the strongest findings, filtering out low-quality evidence, and producing actionable research hypotheses from a large body of publications.

Rather than reading thousands of papers manually, you provide a research question and the two agents deliberate across the literature, challenging each other's claims, flagging weak evidence, and converging on testable hypotheses through a structured consensus mechanism.

The long-term goal is to run this against the full PubMed colorectal cancer corpus (107,000+ papers, 2015–2025) so that every agent response is grounded in retrieved, citable literature — not just the model's pre-trained knowledge.

---

<img width="2547" height="1276" alt="image" src="https://github.com/user-attachments/assets/e5a5f38d-0900-4b11-805b-61c832553fee" />




## The Two Agents

The agents are represented by simple system prompts that can be easily curated to the researchers needs in the UI. The current system uses the following, but the system context is easily and intuitively configurable in addition to the research question prompt.

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

At query time, the search is semantic — not keyword-based. The agent's recent conversation is used to generate 3 query angles simultaneously — the original query, a mechanistic variant, and a clinical/translational variant. All three fire in parallel against the vector store, results are merged and deduplicated by PMID, and the top 12 unique abstracts are injected into the agent's context window before it generates a response. This multi-query approach prevents retrieval from collapsing onto the same papers as the conversation narrows.

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

The RAG toggle connects to `serve_rag.py` running locally. When enabled, each agent turn triggers a multi-query semantic search — 3 parallel queries firing against the vector store, results merged and deduplicated by PMID — injecting the top 12 unique abstracts into the agent's context before the Anthropic API call is made. The API key auto-loads from the local `.env` file via the `/config` endpoint when the RAG server is running.

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
- `GET /config` — serves the Anthropic API key from the local `.env` file to the browser app on page load, so the key never needs to be hardcoded in the HTML or entered manually.
- `POST /search` — accepts a query string, converts it to a vector, finds the top-k nearest abstracts in ChromaDB by cosine similarity, and returns them as JSON. Called 3 times in parallel per agent turn (multi-query retrieval), with results merged and deduplicated before injection.

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
Pasting the full literature corpus into every API call would cost ~$1 per agent turn for 107k papers. RAG retrieves only the 12 most relevant abstracts per turn (~3,500 tokens), reducing cost by ~99% while keeping responses grounded in real citations.

**Why multi-query retrieval instead of a single search?**
As a conversation narrows onto a specific topic, a single query tends to retrieve the same papers on every turn. Three parallel queries — one following the conversation directly, one biased toward mechanistic/resistance literature, one biased toward clinical trial/biomarker literature — ensure retrieval breadth doesn't collapse. Results are deduplicated by PMID so the context window never receives the same abstract twice.

**Why a local vector store instead of a cloud service?**
ChromaDB runs entirely on your machine. No data leaves your system, no API costs for embeddings at query time, and no dependency on external infrastructure.

**Why two agents instead of one?**
A single agent asked to summarize literature will produce consensus. Two agents with adversarial framings will surface contradictions, gaps, and competing interpretations — which is where the actual research signal lives. The DISAGREE: mechanism and evidence hierarchy rules are designed to prevent the agents from collapsing into agreement prematurely.

**Why structured evidence labels?**
The `[RCT]` / `[META]` / `[COHORT]` / `[PRECLINICAL]` / `[EXPERT]` tagging system forces agents to declare the quality of their evidence on every claim. A VERDICT is only valid if backed by `[RCT]` or `[META]` level evidence with no unresolved `DISAGREE:` flags. This is the primary quality filter — it prevents a cell line finding from being argued at the same weight as a phase III trial result.

**On measuring RAG influence vs. pre-trained knowledge**
The model has likely seen a significant portion of pre-2024 PubMed during training. RAG becomes most valuable for 2024–2025 papers that postdate the training cutoff. A planned citation provenance audit — checking whether cited PMIDs exist in the local database — will quantify how much agents draw from retrieved literature vs. internalized knowledge. Papers cited that are absent from the local database are by definition drawn from pre-training, not RAG.

---

## Validation and Accuracy

A core challenge for any AI literature synthesis tool is proving it works beyond subjective expert opinion. Several complementary approaches are planned:

**Citation provenance audit**
After each session, cross-reference every PMID cited by the agents against `pubmed_cache.json`. Citations present in the database are RAG-grounded; citations absent are drawn from pre-training. A high provenance rate indicates the system is genuinely retrieving rather than confabulating. A script to automate this audit is planned.

**Known-answer benchmarking**
Construct a set of questions with established, consensus answers in the CRC literature — e.g., "What did KEYNOTE-177 show about pembrolizumab vs. chemotherapy in MSI-H mCRC first-line?" — and evaluate whether the agents cite the correct trial, reproduce the correct hazard ratio, and reach the correct VERDICT. Systematic deviation from ground truth is measurable without domain expertise.

**Retrieval quality scoring**
For a sample of agent turns, manually assess whether the 12 retrieved abstracts are topically relevant to the query that generated them. A relevance rate below ~70% indicates the embedding model or query strategy needs improvement.

**Contradiction detection rate**
Count how often agents flag genuine contradictions in the literature (unresolved `DISAGREE:` flags that cite conflicting evidence) versus how often they reach premature consensus. A system that never disagrees is not triaging — it is summarizing.

**Longitudinal stability**
Run the same opening topic across multiple sessions and measure VERDICT consistency. High variance suggests the system is sensitive to retrieval noise. Low variance on well-established topics (MSI-H immunotherapy) and appropriate variance on genuinely contested topics (ctDNA decision thresholds) is the target behavior.

---

## Future Directions

### Near-term (in progress)

**Citation provenance audit script**
A post-session script that cross-references every PMID cited by the agents against `pubmed_cache.json`. Citations present in the database are confirmed RAG-grounded; citations absent are flagged as drawn from pre-training. Produces a provenance rate score that quantifies how much the system is genuinely retrieving vs. drawing on internalized model knowledge. This is the primary accuracy validation mechanism.

**Session persistence**
Save and reload full conversation transcripts including message history, agent state, and retrieved citations. Currently a session is lost when the browser closes. Persistence enables longitudinal research sessions across multiple days and allows revisiting prior VERDICTs.

**VERDICT extraction and structured export**
A dedicated export format that pulls only the VERDICT, AGREE/DISAGREE, and NEXT LANE outputs from a full session transcript and formats them as a numbered, citable hypothesis list. Designed to feed directly into a research proposal or manuscript methods section.

**Known-answer benchmark suite**
A curated set of 20–30 questions with established ground-truth answers in the CRC and appendiceal cancer literature — correct hazard ratios, trial primary endpoints, prevalence figures — used to score system factual accuracy and citation accuracy systematically. Each question scored on factual correctness, PMID accuracy, and evidence level tagging.

**Agent-specific query biasing**
Currently both agents use the same multi-query retrieval strategy. The planned implementation biases retrieval by persona — Dr. Clinical's queries are suffixed toward clinical trial and outcomes literature, Dr. Bench's toward mechanistic and TME literature — so each agent draws from the slice of the corpus most relevant to their reasoning frame.

---

### Medium-term

**Configurable corpus**
Allow users to define their own PubMed search query — MeSH terms, date range, journal filters — directly in the UI rather than editing Python config files. Any research domain becomes accessible without code changes. The CRC corpus is the default; users in breast cancer, NSCLC, or rare disease research can point the system at their own literature.

**Epidemiology agent (Agent C)**
A third agent with access to real-world population datasets — SEER, CDC WONDER, CMS claims — that can generate and run EDA pipelines against actual incidence, survival, and treatment data. When Agents A and B debate whether MSI-H patients have better survival outcomes, Agent C runs the SEER query and returns real population-level survival curves. Grounds theoretical deliberation in empirical data.

**Journal tier weighting**
Add a metadata field during the build phase that flags papers from high-impact journals (NEJM, Lancet Oncology, JCO, Nature Medicine). At retrieval time, agents are instructed to prefer higher-tier citations when evidence levels are otherwise equivalent. Addresses the limitation that the current vector search treats a case report and a phase III trial as equal if their abstract text is similarly relevant.

---

### Long-term

**Web-hosted deployment**
Package the system for non-technical users — no local Python server required, no terminal, no build step. Researchers log in, select a domain corpus, enter a research question, and receive a structured VERDICT transcript. The local architecture is a deliberate starting point for development; the end state is a hosted research tool accessible to any lab.

**Multi-domain corpus library**
Pre-built, maintained vector databases for high-value oncology niches — appendiceal cancer, peritoneal malignancies, HIPEC, rare GI tumors — where literature is thin enough that comprehensive synthesis is most valuable and where the gap between what AI can surface and what a researcher has read is largest.

**Systematic review acceleration**
Structured output pipelines that format VERDICT transcripts directly into PRISMA-compatible systematic review frameworks, with auto-generated citation lists, evidence tables, and GRADE-style evidence quality summaries. Designed to reduce the time from research question to manuscript-ready literature review from months to days.
