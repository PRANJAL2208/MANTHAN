"""
literature_search.py
---------------------
The tool agents use to fetch REAL evidence instead of hallucinating citations.

Architecture — 4-source cascade (ordered by reliability & coverage):
  1. OpenAlex      — 250M+ papers, completely free, no key, no rate limits.
                     The closest thing to "all research papers" that exists.
  2. PubMed/NCBI   — 35M papers, gold standard for health/bio/medicine.
  3. arXiv         — 2.3M preprints for CS, Physics, Math, Statistics.
  4. Semantic Scholar — 220M papers, fallback when others yield < 2 results.

Results from all sources are merged, deduplicated by title similarity,
and ranked by citation count (most cited = most validated by the field).

Each source result is cached in SQLite so repeated queries within a session
are instant, and failed/empty results are never cached (to allow retries).
"""

import requests
import time
import sqlite3
import json
import os
import re
import threading
import xml.etree.ElementTree as ET

# ─── Config ─────────────────────────────────────────────────────────────────

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
try:
    from llm_client import get_secret
except ImportError:
    def get_secret(key_name, default=None):
        try:
            import streamlit as st
            if key_name in st.secrets:
                return st.secrets[key_name]
        except Exception:
            pass
        return os.environ.get(key_name, default)

SEMANTIC_SCHOLAR_API_KEY = get_secret("SEMANTIC_SCHOLAR_API_KEY")
NCBI_API_KEY = get_secret("NCBI_API_KEY", "")

CACHE_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".search_cache.db")

# Rate-limit throttle for Semantic Scholar (anonymous tier only)
_ss_lock = threading.Lock()
_ss_last_request = 0.0
SS_MIN_INTERVAL = 2.0  # seconds between Semantic Scholar calls


# ─── Cache ──────────────────────────────────────────────────────────────────

def _init_cache():
    try:
        conn = sqlite3.connect(CACHE_DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS api_cache (
                key TEXT PRIMARY KEY,
                value TEXT,
                timestamp REAL
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[literature_search] Could not initialize cache DB: {e}")

_init_cache()


def _get_cached(key: str):
    try:
        conn = sqlite3.connect(CACHE_DB_PATH)
        row = conn.execute("SELECT value FROM api_cache WHERE key = ?", (key,)).fetchone()
        conn.close()
        return json.loads(row[0]) if row else None
    except Exception:
        return None


def _set_cached(key: str, val):
    try:
        conn = sqlite3.connect(CACHE_DB_PATH)
        conn.execute(
            "INSERT OR REPLACE INTO api_cache (key, value, timestamp) VALUES (?, ?, ?)",
            (key, json.dumps(val), time.time())
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[literature_search] Cache write failed: {e}")


# ─── Public aliases for cache functions (used by tests & external callers) ────

init_cache_db      = _init_cache
get_cached_response = _get_cached
set_cached_response = _set_cached


def set_api_key(key: str):
    """Legacy helper — Semantic Scholar key can also be set via SEMANTIC_SCHOLAR_API_KEY env var."""
    global SEMANTIC_SCHOLAR_API_KEY
    SEMANTIC_SCHOLAR_API_KEY = key


# ─── Abstract reconstruction (OpenAlex uses inverted index format) ───────────

def _preprocess_query(query: str) -> str:
    """
    Strip question words and filler from conversational queries.
    'is coffee good for us?' -> 'coffee'
    Improves relevance for natural-language topic inputs.
    """
    question_words = {
        'is', 'are', 'was', 'were', 'do', 'does', 'did', 'will', 'would',
        'can', 'could', 'should', 'may', 'might', 'what', 'why', 'how',
        'which', 'who', 'where', 'when', 'the', 'a', 'an', 'for', 'us',
        'we', 'our',
    }
    words = query.lower().strip('?').split()
    keywords = [w for w in words if w not in question_words and len(w) >= 2]
    result = ' '.join(keywords) if keywords else query
    if result != query.lower().strip('?'):
        print(f"[literature_search] Query: '{query}' -> '{result}'")
    return result


def _reconstruct_abstract(inverted_index: dict) -> str:
    """
    OpenAlex stores abstracts as an inverted index: {"word": [pos1, pos2], ...}.
    This reconstructs the original sentence order.
    """
    if not inverted_index:
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort(key=lambda x: x[0])
    return " ".join(w for _, w in word_positions)


# ─── Deduplication & Quality Ranking ────────────────────────────────────────

def _normalize_title(title: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s]', '', title.lower())).strip()


def _deduplicate_and_rank(papers: list[dict], limit: int, query_keywords: list[str] = None) -> list[dict]:
    """
    Merge papers from multiple sources, remove near-duplicates by title
    similarity (> 80% word overlap), then rank by a combined score:
      - Title keyword match (most important — keeps results on-topic)
      - Log citation count (secondary quality signal)
    This prevents a massively-cited off-topic paper from outranking
    a highly relevant but lower-cited paper.
    """
    seen_normalized = []
    unique = []

    for paper in papers:
        norm = _normalize_title(paper.get("title", ""))
        words_a = set(norm.split())

        is_dup = False
        for seen in seen_normalized:
            words_b = set(seen.split())
            if words_a and words_b:
                overlap = len(words_a & words_b) / max(len(words_a), len(words_b))
                if overlap > 0.8:
                    is_dup = True
                    break

        if not is_dup and paper.get("abstract", "").strip():
            seen_normalized.append(norm)
            unique.append(paper)

    # Relevance-aware ranking:
    # Title keyword matches outweigh raw citation count so that a relevant
    # coffee paper with 1000 citations ranks above an unrelated 46k-citation
    # paper that just mentions the topic word in a tangential context.
    import math
    kw_set = set(query_keywords) if query_keywords else set()

    def _score(p):
        title_lower = p.get('title', '').lower()
        title_hits = sum(1 for k in kw_set if k in title_lower)
        return title_hits * 5000 + math.log1p(p.get('citationCount', 0))

    unique.sort(key=_score, reverse=True)
    return unique[:limit]


# ─── Source 1: OpenAlex (Primary) ───────────────────────────────────────────

def _search_openalex(query: str, limit: int) -> list[dict]:
    """
    OpenAlex: 250M+ works. Free, no key, no rate limits (polite usage).
    Returns journal articles with abstracts sorted by citation count.
    """
    try:
        resp = requests.get(
            "https://api.openalex.org/works",
            params={
                "search": query,
                "per-page": min(limit * 2, 25),  # fetch extra to compensate for no-abstract papers
                # No sort override — let OpenAlex rank by relevance (default)
                # We re-rank by citation count locally in _deduplicate_and_rank
                "filter": "has_abstract:true,type:article",
                "select": "id,title,abstract_inverted_index,publication_year,cited_by_count,doi",
            },
            headers={"mailto": "debate-agents-research@example.com"},
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])

        papers = []
        for work in results:
            abstract = _reconstruct_abstract(work.get("abstract_inverted_index") or {})
            if not abstract:
                continue
            doi = work.get("doi", "")
            papers.append({
                "paperId": work.get("id", "").split("/")[-1],
                "title": (work.get("title") or "Untitled").strip(),
                "abstract": abstract,
                "year": work.get("publication_year") or "n.d.",
                "citationCount": work.get("cited_by_count", 0),
                "url": doi if doi else work.get("id", ""),
                "source": "OpenAlex",
            })
            if len(papers) >= limit:
                break

        print(f"[literature_search] OpenAlex -> {len(papers)} papers")
        return papers

    except Exception as e:
        print(f"[literature_search] OpenAlex failed: {e}")
        return []


# ─── Source 2: PubMed/NCBI (Health & Biomedical) ────────────────────────────

def _search_pubmed(query: str, limit: int) -> list[dict]:
    """
    PubMed: 35M biomedical papers. Free. Excellent for any health/biology topic.
    """
    try:
        # Step 1: Search for PMIDs
        search_params = {
            "db": "pubmed", "term": query, "retmax": limit,
            "retmode": "json", "sort": "relevance",
        }
        if NCBI_API_KEY:
            search_params["api_key"] = NCBI_API_KEY

        r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params=search_params, timeout=15,
        )
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        # Step 2: Fetch abstracts by PMID
        fetch_params = {
            "db": "pubmed", "id": ",".join(ids),
            "rettype": "abstract", "retmode": "xml",
        }
        if NCBI_API_KEY:
            fetch_params["api_key"] = NCBI_API_KEY

        r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params=fetch_params, timeout=20,
        )
        r.raise_for_status()

        root = ET.fromstring(r.text)
        papers = []
        for article in root.findall(".//PubmedArticle"):
            title_elem = article.find(".//ArticleTitle")
            title = (title_elem.text or "Untitled") if title_elem is not None else "Untitled"
            # Strip any XML tags inside ArticleTitle
            title = re.sub(r'<[^>]+>', '', title).strip()

            abstract_parts = article.findall(".//AbstractText")
            abstract = " ".join(
                (ET.tostring(e, method="text", encoding="unicode") or "")
                for e in abstract_parts
            ).strip()
            if not abstract:
                continue

            year_elem = article.find(".//PubDate/Year")
            year = int(year_elem.text) if year_elem is not None and year_elem.text else "n.d."

            pmid_elem = article.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else ""

            papers.append({
                "paperId": f"pubmed:{pmid}",
                "title": title,
                "abstract": abstract,
                "year": year,
                "citationCount": 0,  # PubMed doesn't expose citation count via E-utils
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                "source": "PubMed",
            })

        print(f"[literature_search] PubMed -> {len(papers)} papers")
        return papers

    except Exception as e:
        print(f"[literature_search] PubMed failed: {e}")
        return []


# ─── Source 3: arXiv (CS / Physics / Math / Stats) ──────────────────────────

def _search_arxiv(query: str, limit: int) -> list[dict]:
    """
    arXiv: 2.3M preprints. Free. Best for CS, AI, Physics, Math, Statistics.
    """
    try:
        resp = requests.get(
            "http://export.arxiv.org/api/query",
            params={
                "search_query": f"all:{query}",
                "max_results": limit,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
            timeout=8,  # arXiv can be slow; 8s is enough or we skip it
        )
        resp.raise_for_status()

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.text)

        papers = []
        for entry in root.findall("atom:entry", ns):
            title_elem = entry.find("atom:title", ns)
            summary_elem = entry.find("atom:summary", ns)
            published_elem = entry.find("atom:published", ns)
            id_elem = entry.find("atom:id", ns)

            title = title_elem.text.strip() if title_elem is not None else "Untitled"
            abstract = summary_elem.text.strip() if summary_elem is not None else ""
            if not abstract:
                continue

            year_str = (published_elem.text or "")[:4]
            year = int(year_str) if year_str.isdigit() else "n.d."
            url = id_elem.text.strip() if id_elem is not None else ""

            papers.append({
                "paperId": f"arxiv:{url.split('/')[-1]}",
                "title": title,
                "abstract": abstract,
                "year": year,
                "citationCount": 0,
                "url": url,
                "source": "arXiv",
            })

        print(f"[literature_search] arXiv -> {len(papers)} papers")
        return papers

    except Exception as e:
        print(f"[literature_search] arXiv failed: {e}")
        return []


# ─── Source 4: Semantic Scholar (Fallback) ───────────────────────────────────

def _search_semantic_scholar(query: str, limit: int, max_retries: int = 5) -> list[dict]:
    """
    Semantic Scholar: 220M papers. Used as final fallback due to aggressive
    anonymous rate limits. With an API key it becomes much more reliable.
    """
    global _ss_last_request

    params = {
        "query": query, "limit": limit,
        "fields": "paperId,title,abstract,year,citationCount,url",
    }
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}

    for attempt in range(max_retries):
        try:
            with _ss_lock:
                elapsed = time.time() - _ss_last_request
                if elapsed < SS_MIN_INTERVAL:
                    time.sleep(SS_MIN_INTERVAL - elapsed)
                _ss_last_request = time.time()

            resp = requests.get(SEMANTIC_SCHOLAR_API, params=params, headers=headers, timeout=15)

            if resp.status_code == 429:
                wait = min(5 * (2 ** attempt), 60)
                print(f"[literature_search] S2 rate-limited, retrying in {wait}s... ({attempt+1}/{max_retries})")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            papers = []
            for p in resp.json().get("data", []):
                if not p.get("abstract"):
                    continue
                papers.append({
                    "paperId": p.get("paperId", ""),
                    "title": p.get("title", "Untitled"),
                    "abstract": p.get("abstract", ""),
                    "year": p.get("year", "n.d."),
                    "citationCount": p.get("citationCount", 0),
                    "url": p.get("url", ""),
                    "source": "Semantic Scholar",
                })
            print(f"[literature_search] Semantic Scholar -> {len(papers)} papers")
            return papers

        except requests.RequestException as e:
            wait = min(5 * (2 ** attempt), 60)
            print(f"[literature_search] S2 request failed: {e}. Retrying in {wait}s...")
            if attempt < max_retries - 1:
                time.sleep(wait)

    return []


# ─── Public API ─────────────────────────────────────────────────────────────

def search_papers(query: str, limit: int = 5, max_retries: int = 5) -> list[dict]:
    """
    Main entry point. Searches all available literature sources using a
    cascade strategy, merges results, deduplicates, and ranks by citation count.

    Source priority:
      1. OpenAlex  (primary — comprehensive, fast, no rate limits)
      2. PubMed    (specialist health/bio)
      3. arXiv     (specialist CS/physics/math, if still short)
      4. Semantic Scholar (fallback if above yield < 2 papers with abstracts)

    Returns a list of paper dicts (title, abstract, year, citationCount, url, source).
    Empty results are never cached so future calls can still retry.
    """
    cache_key = f"cascade_v1:{query}:{limit}"
    cached = _get_cached(cache_key)
    if cached is not None:
        print(f"[literature_search] Cache hit -> {len(cached)} papers for: '{query}'")
        return cached

    # Preprocess: strip question words for better API relevance
    search_query = _preprocess_query(query)

    from concurrent.futures import ThreadPoolExecutor

    t0 = time.time()
    # Execute primary sources in parallel to drastically minimize query latency
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_openalex = executor.submit(_search_openalex, search_query, limit)
        future_pubmed = executor.submit(_search_pubmed, search_query, max(3, limit // 2))
        future_arxiv = executor.submit(_search_arxiv, search_query, max(3, limit // 2))

        openalex_res = future_openalex.result()
        pubmed_res = future_pubmed.result()
        arxiv_res = future_arxiv.result()

    print(f"[{time.strftime('%H:%M:%S')}] [Search] Parallel fetch took {time.time()-t0:.1f}s")

    all_papers: list[dict] = []
    all_papers.extend(openalex_res)
    all_papers.extend(pubmed_res)
    all_papers.extend(arxiv_res)

    # 4. Semantic Scholar — final fallback if other sources yield < 2 papers
    if len(all_papers) < 2:
        t3 = time.time()
        s2_res = _search_semantic_scholar(query, limit=limit)
        print(f"[{time.strftime('%H:%M:%S')}] [Search] Semantic Scholar fallback took {time.time()-t3:.1f}s")
        all_papers.extend(s2_res)

    # Merge, deduplicate, rank (pass keywords for title-relevance scoring)
    keywords = search_query.split()
    final = _deduplicate_and_rank(all_papers, limit, query_keywords=keywords)

    if final:  # Never cache empty results
        _set_cached(cache_key, final)

    print(f"[literature_search] Final merged: {len(final)} papers for: '{query}'")
    return final


def fetch_paper_details(paper_id: str) -> dict | None:
    """
    Fetches deeper metadata for a specific Semantic Scholar paper by ID.
    Used when an agent wants to dig into one paper more deeply during rebuttal.
    """
    cache_key = f"details:{paper_id}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
    params = {"fields": "title,abstract,year,citationCount,url,authors,venue,tldr"}

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            data = resp.json()
            details = {
                "title": data.get("title", "Untitled"),
                "abstract": data.get("abstract") or "No abstract available.",
                "year": data.get("year", "n.d."),
                "citationCount": data.get("citationCount", 0),
                "venue": data.get("venue", "Unknown venue"),
                "authors": [a.get("name") for a in data.get("authors", [])][:5],
                "tldr": (data.get("tldr") or {}).get("text", ""),
                "url": data.get("url", ""),
            }
            _set_cached(cache_key, details)
            return details
        except requests.RequestException:
            if attempt < 2:
                time.sleep(2 ** attempt)

    return None


def format_papers_for_prompt(papers: list[dict]) -> str:
    """
    Formats the paper list into a clean numbered text block for LLM prompts.
    Agents cite using [1], [2] etc. which the guardrail checks against.
    """
    if not papers:
        return "(No supporting literature could be retrieved.)"

    blocks = []
    for i, p in enumerate(papers, 1):
        source_tag = f" [{p.get('source', 'source')}]" if p.get('source') else ""
        blocks.append(
            f"[{i}] {p['title']} ({p['year']}, {p.get('citationCount', 0)} citations){source_tag}\n"
            f"    Abstract: {p['abstract'][:600]}"  # Trimmed from 1500 to 600 to reduce token count
        )
    return "\n".join(blocks)