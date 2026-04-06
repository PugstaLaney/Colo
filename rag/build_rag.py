"""
build_rag.py
────────────
Scrapes PubMed for colorectal cancer abstracts and builds a local
ChromaDB vector store that the dialogue app can query.

Usage
─────
  # Test run — first 500 papers only
  py build_rag.py --limit 500

  # Full build (takes several hours for 60k+ papers)
  py build_rag.py

  # Resume an interrupted build (skips already-embedded PMIDs)
  py build_rag.py --resume

Requirements: py -m pip install -r requirements.txt
"""

# ── Imports ───────────────────────────────────────────────────────────────────
# argparse     — reads command-line flags like --limit and --resume
# json         — reads/writes the local abstract cache file
# time         — adds delays between API calls to respect NCBI rate limits
# pathlib      — constructs file paths in a cross-platform way
# urllib       — makes HTTP requests to the iCite API (stdlib, no install needed)
import argparse
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

# Bio.Entrez  — Python wrapper for NCBI's E-utilities API (PubMed access)
# PersistentClient — ChromaDB client that saves the vector store to disk
# SentenceTransformer — loads the embedding model that converts text to vectors
# tqdm — draws progress bars in the terminal
from Bio import Entrez
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
# All tunable settings live here so you never need to hunt through the code.

NCBI_EMAIL   = "pallaney@gmail.com"       # Required by NCBI to identify who is making requests
NCBI_API_KEY = "c8e0902b764c43d48fd2e3d1d7e929ad1808"  # Raises rate limit from 3 → 10 requests/sec

# MeSH term scoped to colorectal cancer, date-filtered to last 10 years
SEARCH_TERM  = '"colorectal neoplasms"[MeSH]'
DATE_RANGE   = "2015/01/01:2025/12/31[dp]"

BATCH_SIZE   = 500   # How many abstracts to download from PubMed per API call
EMBED_BATCH  = 64    # How many abstracts to embed at once — lower this if you run out of RAM

# File paths — all data lives in Colo/data/ one level above this script
DATA_DIR   = Path(__file__).parent.parent / "data"
CACHE_FILE = DATA_DIR / "pubmed_cache.json"  # Checkpoint file — raw abstracts saved as JSON
CHROMA_DIR = DATA_DIR / "chroma_db"          # ChromaDB folder — stores vectors + metadata
COLLECTION = "crc_abstracts"                 # Name of the collection inside ChromaDB

# Embedding model — converts abstract text into 384-dimensional vectors
# all-MiniLM-L6-v2 is fast and general purpose
# Swap for "allenai-specter" for better biomedical domain accuracy at the cost of speed
EMBED_MODEL = "all-MiniLM-L6-v2"


# ── PubMed helpers ────────────────────────────────────────────────────────────

def configure_entrez():
    # Registers your credentials with the Entrez library before any API calls are made
    Entrez.email = NCBI_EMAIL
    if NCBI_API_KEY:
        Entrez.api_key = NCBI_API_KEY


def search_pubmed(limit: int | None = None):
    """
    Registers the search query on NCBI's server-side history and returns
    the total paper count plus two keys (WebEnv, QueryKey) that act like
    a bookmark — all subsequent fetch calls reference these keys instead
    of re-running the search. retmax=0 means we only want the count,
    not a capped list of IDs.
    """
    print(f"Searching PubMed: {SEARCH_TERM} [{DATE_RANGE}]")

    handle = Entrez.esearch(
        db="pubmed",
        term=f"{SEARCH_TERM} AND {DATE_RANGE}",
        retmax=0,        # Don't return IDs — just register the search and get the count
        usehistory="y",  # Store results server-side so we can page through them in batches
    )
    record = Entrez.read(handle)
    handle.close()

    total = int(record["Count"])
    fetch_count = min(total, limit) if limit else total
    print(f"  Found {total:,} total papers — will fetch {fetch_count:,}")
    return fetch_count, record["WebEnv"], record["QueryKey"]


def fetch_batch(web_env: str, query_key: str, start: int, count: int) -> list[dict]:
    """
    Downloads a single batch of abstracts from PubMed in XML format.
    Uses the server-side history keys from search_pubmed() to page through
    results with no cap. Retries up to 3 times with increasing wait if the
    network call fails — NCBI occasionally drops connections on large pulls.
    """
    for attempt in range(3):
        try:
            handle = Entrez.efetch(
                db="pubmed",
                rettype="xml",
                retmode="xml",
                retstart=start,   # Offset into the full result set
                retmax=count,     # How many records to return in this batch
                webenv=web_env,   # Server-side history bookmark
                query_key=query_key,
            )
            records = Entrez.read(handle)
            handle.close()
            return parse_records(records)  # Convert XML to clean Python dicts
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"  Fetch error ({e}), retrying in {wait}s...")
            time.sleep(wait)
    return []  # Return empty list if all 3 attempts fail — progress is saved, safe to resume


def parse_records(records) -> list[dict]:
    """
    Converts raw Entrez XML records into clean Python dicts with only the
    fields we need. Skips papers with no abstract — they have no useful
    text to embed. Silently skips malformed records rather than crashing.
    Each dict contains: pmid, title, abstract, authors, year, journal.
    """
    results = []
    for article in records.get("PubmedArticle", []):
        try:
            medline = article["MedlineCitation"]
            art     = medline["Article"]

            # Some abstracts are split into labeled sections (background, methods, results)
            # — join them into a single string
            abstract_obj   = art.get("Abstract", {})
            abstract_texts = abstract_obj.get("AbstractText", [])
            if isinstance(abstract_texts, list):
                abstract = " ".join(str(t) for t in abstract_texts)
            else:
                abstract = str(abstract_texts)

            if not abstract.strip():
                continue  # No abstract text = nothing useful to embed, skip

            # Collect up to 5 authors — enough for citation display without bloating metadata
            author_list = art.get("AuthorList", [])
            authors = []
            for a in author_list[:5]:
                last = a.get("LastName", "")
                init = a.get("Initials", "")
                if last:
                    authors.append(f"{last} {init}".strip())

            # Publication year — falls back to MedlineDate string if structured year is missing
            pub_date = art.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
            year = pub_date.get("Year", pub_date.get("MedlineDate", "")[:4])

            pmid = str(medline["PMID"])

            results.append({
                "pmid":     pmid,
                "title":    str(art.get("ArticleTitle", "")),
                "abstract": abstract,
                "authors":  ", ".join(authors),
                "year":     str(year),
                "journal":  str(art.get("Journal", {}).get("Title", "")),
            })
        except Exception:
            continue  # Malformed record — skip silently, don't crash the whole batch
    return results


# ── iCite citation metrics ────────────────────────────────────────────────────

def fetch_icite_batch(pmids: list[str]) -> dict[str, dict]:
    """
    Fetches citation metrics from the NIH iCite API for a batch of PMIDs.
    iCite is completely free, requires no API key, and returns:
      - citation_count  : raw number of times the paper has been cited
      - rcr             : Relative Citation Ratio — a field-normalized impact
                          score where 1.0 = average for papers in that field,
                          2.0 = twice the average, etc. Better than raw citation
                          count because it accounts for field-specific norms.
    Returns a dict keyed by PMID string. Papers not found in iCite get rcr=0.
    Handles network failures silently — missing iCite data just means rcr=0,
    which excludes the paper from ranking bonuses but doesn't break anything.
    """
    if not pmids:
        return {}

    # iCite accepts up to 100 PMIDs per request as a comma-separated list
    url = "https://icite.od.nih.gov/api/pubs?pmids=" + ",".join(pmids)
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        result = {}
        for pub in data.get("data", []):
            pmid = str(pub.get("pmid", ""))
            if pmid:
                result[pmid] = {
                    "citation_count": int(pub.get("citation_count") or 0),
                    "rcr":            float(pub.get("relative_citation_ratio") or 0.0),
                }
        return result
    except Exception:
        return {}  # Network error or rate limit — return empty, safe to continue


def enrich_with_icite(cache: dict) -> dict:
    """
    Loops through all cached records and adds citation_count and rcr fields
    by querying iCite in batches of 100. Only fetches records that don't
    already have iCite data (so re-running is safe and fast).
    Prints progress since this can take a few minutes for large corpora.
    """
    # Find PMIDs that still need iCite enrichment
    needs_enrichment = [
        pmid for pmid, r in cache.items()
        if "rcr" not in r
    ]

    if not needs_enrichment:
        print("  iCite data already present for all records.")
        return cache

    print(f"\nFetching iCite citation metrics for {len(needs_enrichment):,} papers...")
    ICITE_BATCH = 100  # iCite API limit per request

    for i in tqdm(range(0, len(needs_enrichment), ICITE_BATCH)):
        batch_pmids = needs_enrichment[i : i + ICITE_BATCH]
        icite_data  = fetch_icite_batch(batch_pmids)

        for pmid in batch_pmids:
            metrics = icite_data.get(pmid, {"citation_count": 0, "rcr": 0.0})
            cache[pmid]["citation_count"] = metrics["citation_count"]
            cache[pmid]["rcr"]            = metrics["rcr"]

        time.sleep(0.05)  # ~20 req/sec — well within iCite's limits

    print(f"  iCite enrichment complete.")
    return cache


# ── Cache helpers ─────────────────────────────────────────────────────────────

def load_cache() -> dict[str, dict]:
    """
    Loads the local JSON checkpoint file into memory as a dict keyed by PMID.
    If the file doesn't exist yet (first run), returns an empty dict.
    This is what makes resuming possible — completed batches are never re-fetched.
    """
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    """
    Writes the current cache to disk after every batch. If the script is
    interrupted, the next run picks up from here rather than starting over.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


# ── Embedding + ChromaDB ──────────────────────────────────────────────────────

def build_vectorstore(records: list[dict], resume: bool):
    """
    Takes all cached abstract dicts, converts them to vectors using the
    embedding model, and stores both the vectors and original text in ChromaDB.
    If --resume is set, queries ChromaDB for already-stored PMIDs and skips them
    so only new records get embedded.
    """
    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)  # Downloads model on first run (~90MB), cached after

    # PersistentClient saves to disk at CHROMA_DIR — data survives between runs
    client     = PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},  # Use cosine similarity for vector comparisons
    )

    # --resume: fetch the list of PMIDs already in ChromaDB and skip them
    if resume:
        existing = set(collection.get(include=[])["ids"])
        records  = [r for r in records if r["pmid"] not in existing]
        print(f"  Resuming: {len(existing):,} already embedded, {len(records):,} remaining")

    if not records:
        print("  Nothing to embed — store is up to date.")
        return

    print(f"\nEmbedding {len(records):,} abstracts in batches of {EMBED_BATCH}...")

    for i in tqdm(range(0, len(records), EMBED_BATCH)):
        batch = records[i : i + EMBED_BATCH]

        # Concatenate title + abstract — gives the embedding model more context
        # than abstract alone, improving retrieval relevance
        texts = [f"{r['title']}. {r['abstract']}" for r in batch]

        # model.encode() runs the neural network — outputs a list of 384-number vectors
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        # upsert = insert if new, update if PMID already exists
        # Stores three things per record: the vector (for search), the abstract
        # text (returned to the agent), and metadata (title, authors, year, journal)
        collection.upsert(
            ids        = [r["pmid"] for r in batch],
            embeddings = embeddings,
            documents  = [r["abstract"] for r in batch],  # Original text returned at query time
            metadatas  = [
                {
                    "pmid":           r["pmid"],
                    "title":          r["title"][:500],    # ChromaDB has a metadata string length limit
                    "authors":        r["authors"][:300],
                    "year":           r["year"],
                    "journal":        r["journal"][:200],
                    # iCite citation metrics — used by serve_rag.py to re-rank results
                    # rcr (Relative Citation Ratio): field-normalized impact score
                    #   1.0 = field average, 2.0 = twice average, 0 = not yet in iCite
                    "rcr":            float(r.get("rcr", 0.0)),
                    "citation_count": int(r.get("citation_count", 0)),
                }
                for r in batch
            ],
        )

    print(f"\nDone. ChromaDB collection '{COLLECTION}' now has {collection.count():,} records.")
    print(f"Stored at: {CHROMA_DIR}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Parse command-line flags: --limit N caps the fetch for testing, --resume skips
    # already-embedded records so interrupted runs can continue
    parser = argparse.ArgumentParser(description="Build the CRC RAG vector store from PubMed.")
    parser.add_argument("--limit",  type=int, default=None, help="Cap number of papers (for testing)")
    parser.add_argument("--resume", action="store_true",    help="Skip PMIDs already in ChromaDB")
    args = parser.parse_args()

    configure_entrez()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Search ────────────────────────────────────────────────────────
    # Registers the query on NCBI's server and gets back the total count
    # plus history keys for paging — no ID list, no 10k cap
    total, web_env, query_key = search_pubmed(limit=args.limit)

    # ── Step 2: Fetch abstracts ───────────────────────────────────────────────
    # Load whatever was previously cached, then fetch only what's missing.
    # Saves to disk after every batch so interruptions don't lose progress.
    cache = load_cache()
    delay = 0.11 if NCBI_API_KEY else 0.34  # API key = 10 req/sec; no key = 3 req/sec

    already_cached = len(cache)
    if already_cached >= total:
        print(f"\nAll {total:,} papers already cached.")
    else:
        remaining = total - already_cached
        print(f"\nFetching {remaining:,} new abstracts (already have {already_cached:,} cached)...")

        # Page through the full result set using server-side history offsets
        # fetch_batch uses WebEnv/QueryKey so there is no retmax cap here
        offsets = list(range(already_cached, total, BATCH_SIZE))
        for start in tqdm(offsets):
            batch_size    = min(BATCH_SIZE, total - start)
            batch_records = fetch_batch(web_env, query_key, start, batch_size)
            for r in batch_records:
                cache[r["pmid"]] = r
            save_cache(cache)       # Checkpoint after every batch
            time.sleep(delay)       # Respect NCBI rate limits

        print(f"  Cache now contains {len(cache):,} records → {CACHE_FILE}")

    # ── Step 3: Enrich with iCite citation metrics ────────────────────────────
    # Adds citation_count and rcr (Relative Citation Ratio) to every cached
    # record by querying the free NIH iCite API. Safe to re-run — skips
    # records already enriched. Saves updated cache to disk afterward.
    cache = enrich_with_icite(cache)
    save_cache(cache)

    all_records = list(cache.values())

    # ── Step 4: Embed + store ─────────────────────────────────────────────────
    # Convert all cached abstracts to vectors and load into ChromaDB.
    # rcr and citation_count are stored as metadata alongside each vector.
    # --resume skips PMIDs already present in the vector store.
    build_vectorstore(all_records, resume=args.resume)


if __name__ == "__main__":
    main()
