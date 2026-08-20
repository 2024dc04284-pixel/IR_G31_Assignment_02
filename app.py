"""
IR Assignment — Heterogeneous Collection & Text Mining
Run with:  streamlit run app.py

One file, four stages, read top to bottom:
  1. ACQUISITION  three independent, heterogeneous sources feeding one store:
                   1a. web crawl    breadth-first, multiple seeds, configurable
                                     depth, URL-level + document-level dedup
                   1b. public dataset  upload a CSV/JSON file, or fetch one
                                     live from a public URL
                   1c. public API   query the Wikipedia MediaWiki API (JSON,
                                     not HTML — a genuinely different fetch path)
  2. STORAGE      SQLite, document content and metadata in separate tables,
                   metadata tagged with which of the three sources it came from
  3. PREPROCESS   cleaning, tokenising, stopwords, stemming
  4. FEATURES     TF / TF-IDF, keyword extraction, corpus statistics,
                   visualisations, document classification
"""
from __future__ import annotations

import hashlib
import io
import math
import re
import sqlite3
import time
import unicodedata
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urldefrag, urljoin, urlparse, urlunparse
import urllib.robotparser as robotparser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
import networkx as nx
from urllib.parse import urlparse
from sklearn.metrics import precision_score, recall_score, f1_score
import math

DB_PATH = Path(__file__).resolve().parent / "crawl_store.db"
SAMPLE_DATASET_PATH = Path(__file__).resolve().parent / "sample_dataset.csv"
USER_AGENT = "IRAssignmentCrawler/1.0 (student project; contact set in course portal)"

# =============================================================================
# 1a. ACQUISITION — web crawl
# =============================================================================

TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term",
                    "utm_content", "gclid", "fbclid", "ref", "mc_cid", "mc_eid"}
NON_HTML_SUFFIXES = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
                      ".ico", ".css", ".js", ".zip", ".gz", ".mp4", ".mp3",
                      ".doc", ".docx", ".xls", ".xlsx", ".xml", ".json")
DROP_TAGS = ["script", "style", "nav", "footer", "header", "aside", "form",
             "noscript", "iframe", "svg", "button"]


def canonicalize_url(url: str, base: str | None = None) -> str:
    """Fold equivalent URLs (case, default port, www, trailing slash, tracking
    params, fragment) onto one string, so the 'seen' set can dedupe on it."""
    if not url:
        return ""
    url = url.strip()
    if url.startswith(("mailto:", "javascript:", "tel:", "#", "data:")):
        return ""
    if base:
        url = urljoin(base, url)
    url, _ = urldefrag(url)
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        return ""
    netloc = p.netloc.lower()
    if ":" in netloc:
        host, _, port = netloc.rpartition(":")
        if port == {"http": "80", "https": "443"}.get(p.scheme):
            netloc = host
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = p.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    kept = sorted((k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
                  if k.lower() not in TRACKING_PARAMS)
    query = "&".join(f"{k}={v}" for k, v in kept)
    return urlunparse((p.scheme, netloc, path, "", query, ""))


def registered_domain(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def looks_like_html(url: str) -> bool:
    return not urlparse(url).path.lower().endswith(NON_HTML_SUFFIXES)


def extract_page(html: str, url: str) -> dict:
    """Pull article-ish title + text out of a raw HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(DROP_TAGS):
        tag.decompose()
    title = (soup.title.get_text(strip=True) if soup.title else None) \
        or (soup.h1.get_text(strip=True) if soup.h1 else None) or url
    node = soup.find("article") or soup.find("main") or soup.body or soup
    text = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", node.get_text("\n"))).strip()
    links = [canonicalize_url(a["href"], base=url) for a in soup.find_all("a", href=True)]
    links = [l for l in links if l and looks_like_html(l)]
    return {"title": title[:300], "text": f"{title}. {text}" if text else title, "links": links}


def content_hash(text: str) -> str:
    """SHA-256 over whitespace/case-normalised text — the exact-duplicate key."""
    normalised = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


DOC_ID_PREFIXES = {"web": "web", "dataset": "ds", "api": "api"}


def doc_id_for(url: str, source_type: str = "web") -> str:
    prefix = DOC_ID_PREFIXES.get(source_type, source_type)
    return f"{prefix}-" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


class RobotsCache:
    """One parsed robots.txt per host, fetched lazily and reused."""

    def __init__(self):
        self._cache: dict[str, robotparser.RobotFileParser | None] = {}

    def allows(self, url: str) -> bool:
        p = urlparse(url)
        host = f"{p.scheme}://{p.netloc}"
        if host not in self._cache:
            rp = robotparser.RobotFileParser()
            try:
                resp = requests.get(host + "/robots.txt", timeout=8,
                                     headers={"User-Agent": USER_AGENT})
                rp.parse(resp.text.splitlines()) if resp.status_code == 200 else None
                self._cache[host] = rp if resp.status_code == 200 else None
            except Exception:
                self._cache[host] = None
        rp = self._cache[host]
        return True if rp is None else rp.can_fetch(USER_AGENT, url)


def crawl(seeds: list[str], max_depth: int, max_pages: int, same_domain_only: bool,
          delay: float, known_urls: set[str], known_hashes: set[str], log_fn=None):
    """Breadth-first crawl. Returns (documents, stats, seen_urls, log_rows).

    Two independent duplicate checks: URL-level (``known_urls``, checked before
    fetching) and document-level (``known_hashes``, checked after extraction —
    catches the same article republished at a different URL).

    The frontier is one queue per seed domain, visited round-robin rather than
    as a single FIFO. A plain FIFO fetches every seed first but then drains
    almost the entire page budget on whichever seed's outlinks happened to be
    queued first — a page with hundreds of links can starve every other seed
    domain down to just its seed page. Round-robin keeps the crawl balanced
    across domains, which matters for the classifier in step 4: it needs
    several documents from more than one domain.
    """
    from collections import deque
    robots = RobotsCache()
    seen_urls = set(known_urls)
    seen_hashes = set(known_hashes)
    seed_domains = {registered_domain(s) for s in seeds}
    domain_queues: dict[str, deque] = {}
    stats = Counter()
    documents = []
    log_rows = []
    last_hit: dict[str, float] = {}

    for seed in seeds:
        c = canonicalize_url(seed)
        if not c or c in seen_urls:
            stats["duplicate_url"] += 1
            continue
        seen_urls.add(c)
        domain_queues.setdefault(registered_domain(c), deque()).append((c, 0, seed))

    attempt = 0
    while stats["fetched"] < max_pages and any(domain_queues.values()):
        dom_with_work = [d for d, q in domain_queues.items() if q]
        if not dom_with_work:
            break
        dom = dom_with_work[attempt % len(dom_with_work)]
        attempt += 1
        url, depth, seed = domain_queues[dom].popleft()

        if not robots.allows(url):
            stats["robots_blocked"] += 1
            log_rows.append({"url": url, "depth": depth, "status": "robots_blocked"})
            continue

        host = registered_domain(url)
        gap = time.monotonic() - last_hit.get(host, 0.0)
        if gap < delay:
            time.sleep(delay - gap)
        last_hit[host] = time.monotonic()

        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": USER_AGENT})
        except Exception as exc:
            stats["errors"] += 1
            log_rows.append({"url": url, "depth": depth, "status": "error", "detail": str(exc)})
            continue

        if resp.status_code >= 400 or "html" not in resp.headers.get("Content-Type", "html").lower():
            stats["errors"] += 1
            log_rows.append({"url": url, "depth": depth, "status": f"http_{resp.status_code}"})
            continue

        stats["fetched"] += 1
        page = extract_page(resp.text, url)
        if log_fn:
            log_fn(stats["fetched"], max_pages, url)

        if depth < max_depth:
            for link in page["links"]:
                if link in seen_urls:
                    stats["duplicate_url"] += 1
                    continue
                link_dom = registered_domain(link)
                if same_domain_only and not any(link_dom == s or link_dom.endswith("." + s) for s in seed_domains):
                    continue
                seen_urls.add(link)
                domain_queues.setdefault(link_dom, deque()).append((link, depth + 1, seed))

        n_words = len(page["text"].split())
        if n_words < 40:
            stats["too_short"] += 1
            log_rows.append({"url": url, "depth": depth, "status": "too_short"})
            continue

        h = content_hash(page["text"])
        if h in seen_hashes:
            stats["duplicate_doc"] += 1
            log_rows.append({"url": url, "depth": depth, "status": "duplicate_doc"})
            continue
        seen_hashes.add(h)

        stats["stored"] += 1
        log_rows.append({"url": url, "depth": depth, "status": "stored"})
        documents.append({
            "doc_id": doc_id_for(url, "web"), "url": url, "title": page["title"],
            "raw_text": page["text"], "content_hash": h,
            "domain": registered_domain(url), "crawl_depth": depth, "seed_url": seed,
            "http_status": resp.status_code,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_type": "web", 
            "source_detail": "",
            "outgoing_links": page["links"],
        })

    return documents, stats, seen_urls, log_rows


# =============================================================================
# 1b. ACQUISITION — public dataset (file upload or fetched from a public URL)
# =============================================================================

def load_tabular(file_like, filename: str) -> pd.DataFrame:
    """Parse an uploaded/fetched file into a DataFrame. CSV by default;
    JSON if the name says so — the two shapes most public dataset exports use."""
    if filename.lower().endswith(".json"):
        return pd.read_json(file_like)
    return pd.read_csv(file_like)


def dataset_rows_to_items(df: pd.DataFrame, text_col: str, title_col: str,
                           source_name: str, max_rows: int) -> list[dict]:
    """Normalise dataset rows into the same shape build_documents() expects
    from the crawler: url/title/text/domain/source_detail."""
    items = []
    for i, row in df.head(max_rows).iterrows():
        text = str(row[text_col]).strip() if text_col in df.columns and pd.notna(row[text_col]) else ""
        if not text:
            continue
        title = (str(row[title_col]).strip()
                  if title_col and title_col in df.columns and pd.notna(row.get(title_col))
                  else text[:80])
        items.append({
            "url": f"dataset://{source_name}/{i}",
            "title": title, "text": text,
            "domain": f"dataset:{source_name}",
            "source_detail": source_name,
        })
    return items


# =============================================================================
# 1c. ACQUISITION — public API (Wikipedia MediaWiki API, JSON — not HTML)
# =============================================================================

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"


def fetch_wikipedia(query: str, max_results: int) -> list[dict]:
    """Search Wikipedia and pull each hit's plaintext extract in a single HTTP
    request, using ``generator=search`` to feed the search results straight
    into ``prop=extracts`` instead of one search call plus one extract call
    per result. ``exlimit=max`` is required for the extract to come back for
    more than the first page. One request regardless of how many results are
    asked for — a structured-API fetch, not HTML scraping: no BeautifulSoup,
    no link-following, no robots.txt — a different acquisition path entirely."""
    resp = requests.get(WIKIPEDIA_API, params={
        "action": "query", "generator": "search", "gsrsearch": query, "gsrlimit": max_results,
        "prop": "extracts", "explaintext": 1, "exlimit": "max", "format": "json",
    }, headers={"User-Agent": USER_AGENT}, timeout=15)
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})

    items = []
    for page in pages.values():
        pageid, title = page.get("pageid"), page.get("title", "")
        text = (page.get("extract") or "").strip()
        if not pageid or not text:
            continue
        items.append({
            "url": f"https://en.wikipedia.org/?curid={pageid}",
            "title": title, "text": f"{title}. {text}",
            "domain": "api:wikipedia",
            "source_detail": f'wikipedia search "{query}"',
        })
    return items


# =============================================================================
# 1d. ACQUISITION — shared quality gate for the non-crawler sources
# =============================================================================

def build_documents(items: list[dict], source_type: str,
                     known_urls: set[str], known_hashes: set[str]) -> tuple[list[dict], Counter, set[str]]:
    """Apply the same policy the crawler applies inline: URL-level dedup,
    a minimum length, and content-hash dedup — so a document is treated the
    same whether it was crawled, uploaded, or pulled from an API. Content-hash
    dedup here also catches near-duplicates *across* sources, e.g. the same
    Wikipedia article both crawled and fetched via the API."""
    stats = Counter()
    seen_urls = set(known_urls)
    seen_hashes = set(known_hashes)
    documents = []
    for item in items:
        url = item["url"]
        if not url or url in seen_urls:
            stats["duplicate_url"] += 1
            continue
        seen_urls.add(url)

        text = item["text"]
        if len(text.split()) < 40:
            stats["too_short"] += 1
            continue

        h = content_hash(text)
        if h in seen_hashes:
            stats["duplicate_doc"] += 1
            continue
        seen_hashes.add(h)

        stats["stored"] += 1
        documents.append({
            "doc_id": doc_id_for(url, source_type), "url": url,
            "title": item.get("title") or url, "raw_text": text, "content_hash": h,
            "domain": item.get("domain", source_type), "crawl_depth": 0, "seed_url": url,
            "http_status": 200, "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_type": source_type, "source_detail": item.get("source_detail", ""),
        })
    return documents, stats, seen_urls


# =============================================================================
# 2. STORAGE — content and metadata in separate tables, joined on doc_id
# =============================================================================

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    raw_text     TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    n_chars      INTEGER,
    n_words      INTEGER
);
CREATE TABLE IF NOT EXISTS metadata (
    doc_id       TEXT PRIMARY KEY REFERENCES documents(doc_id),
    url          TEXT,
    domain       TEXT,
    title        TEXT,
    crawl_depth  INTEGER,
    seed_url     TEXT,
    http_status  INTEGER,
    fetched_at   TEXT,
    source_type  TEXT DEFAULT 'web',
    source_detail TEXT
);
CREATE TABLE IF NOT EXISTS urls (
    url_canonical TEXT PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS document_links (
    source_doc_id TEXT NOT NULL,
    target_url TEXT NOT NULL,
    PRIMARY KEY (source_doc_id, target_url)
);
"""


def _migrate(con: sqlite3.Connection) -> None:
    """SQLite has no ADD COLUMN IF NOT EXISTS — patch older stores in place
    so a DB created before heterogeneous sources existed still opens cleanly."""
    cols = {r["name"] for r in con.execute("PRAGMA table_info(metadata)")}
    if "source_type" not in cols:
        con.execute("ALTER TABLE metadata ADD COLUMN source_type TEXT DEFAULT 'web'")
    if "source_detail" not in cols:
        con.execute("ALTER TABLE metadata ADD COLUMN source_detail TEXT")


@contextmanager
def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        con.executescript(SCHEMA)
        _migrate(con)
        yield con
        con.commit()
    finally:
        con.close()


def store_documents(docs: list[dict]) -> None:
    with db() as con:
        for d in docs:
            con.execute(
                "INSERT OR IGNORE INTO documents (doc_id, raw_text, content_hash, n_chars, n_words) "
                "VALUES (?,?,?,?,?)",
                (d["doc_id"], d["raw_text"], d["content_hash"],
                 len(d["raw_text"]), len(d["raw_text"].split())))
            con.execute(
                "INSERT OR IGNORE INTO metadata "
                "(doc_id, url, domain, title, crawl_depth, seed_url, http_status, fetched_at, "
                "source_type, source_detail) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (d["doc_id"], d["url"], d["domain"], d["title"], d["crawl_depth"],
                 d["seed_url"], d["http_status"], d["fetched_at"],
                 d.get("source_type", "web"), d.get("source_detail", "")))

def store_document_links(docs):
    """
    Store outgoing hyperlinks extracted from crawled documents.
    """

    with db() as con:

        for doc in docs:

            source_doc_id = doc["doc_id"]

            for target_url in doc.get("outgoing_links", []):

                con.execute(
                    """
                    INSERT OR IGNORE INTO document_links
                    (source_doc_id, target_url)
                    VALUES (?, ?)
                    """,
                    (
                        source_doc_id,
                        target_url
                    )
                )

def store_urls(urls: set[str]) -> None:
    with db() as con:
        con.executemany("INSERT OR IGNORE INTO urls (url_canonical) VALUES (?)",
                         [(u,) for u in urls])


def known_urls() -> set[str]:
    with db() as con:
        return {r["url_canonical"] for r in con.execute("SELECT url_canonical FROM urls")}


def known_hashes() -> set[str]:
    with db() as con:
        return {r["content_hash"] for r in con.execute("SELECT content_hash FROM documents")}


def load_corpus() -> pd.DataFrame:
    with db() as con:
        return pd.read_sql_query(
            "SELECT d.doc_id, d.raw_text, d.n_words, m.url, m.domain, m.title, "
            "m.crawl_depth, m.seed_url, m.fetched_at, m.source_type, m.source_detail "
            "FROM documents d JOIN metadata m ON m.doc_id = d.doc_id ORDER BY d.doc_id", con)


def store_counts() -> tuple[int, int, int]:
    with db() as con:
        docs = con.execute("SELECT COUNT(*) n FROM documents").fetchone()["n"]
        urls = con.execute("SELECT COUNT(*) n FROM urls").fetchone()["n"]
        doms = con.execute("SELECT COUNT(DISTINCT domain) n FROM metadata").fetchone()["n"]
        return docs, urls, doms


def source_breakdown() -> pd.DataFrame:
    with db() as con:
        return pd.read_sql_query(
            "SELECT source_type, COUNT(*) AS documents FROM metadata "
            "GROUP BY source_type ORDER BY documents DESC", con)


def reset_store() -> None:
    with db() as con:
        for t in ("documents", "metadata", "urls", "document_links"):
            con.execute(f"DELETE FROM {t}")

# =============================================================================
# 3. PREPROCESSING
# =============================================================================

STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can't cannot could couldn't did
didn't do does doesn't doing don't down during each few for from further had
hadn't has hasn't have haven't having he he'd he'll he's her here here's hers
herself him himself his how how's i i'd i'll i'm i've if in into is isn't it
it's its itself let's me more most mustn't my myself no nor not of off on once
only or other ought our ours ourselves out over own same shan't she she'd
she'll she's should shouldn't so some such than that that's the their theirs
them themselves then there there's these they they'd they'll they're they've
this those through to too under until up very was wasn't we we'd we'll we're
we've were weren't what what's when when's where where's which while who who's
whom why why's with won't would wouldn't you you'd you'll you're you've your
yours yourself yourselves said also one two would could
""".split())

TOKEN_RE = re.compile(r"[a-z]{2,}")
STEM_SUFFIXES = ("ational", "tional", "ization", "iveness", "fulness", "ousness",
                  "ization", "ation", "ments", "ingly", "edly", "ment", "ness",
                  "ions", "ing", "ies", "ied", "ers", "est", "ly", "ed", "es", "s")


def clean_text(text: str) -> str:
    if not text:
        return ""
    out = re.sub(r"<[^>]+>", " ", text)
    out = re.sub(r"https?://\S+|www\.\S+", " ", out)
    out = "".join(c for c in unicodedata.normalize("NFKD", out) if not unicodedata.combining(c))
    out = out.lower()
    return re.sub(r"\s+", " ", out).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def stem(token: str) -> str:
    """Suffix-stripping stemmer — simple by design, no external model needed."""
    for suf in STEM_SUFFIXES:
        if len(token) - len(suf) >= 3 and token.endswith(suf):
            return token[: -len(suf)]
    return token


def preprocess(text: str, remove_stopwords: bool = True, use_stemming: bool = True) -> list[str]:
    tokens = tokenize(clean_text(text))
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if use_stemming:
        tokens = [stem(t) for t in tokens]
    return tokens


@st.cache_data(show_spinner=False)
def preprocess_corpus(texts: list[str], remove_stopwords: bool, use_stemming: bool):
    started = time.perf_counter()
    tokens = [preprocess(t, remove_stopwords, use_stemming) for t in texts]
    return tokens, time.perf_counter() - started


PREPROCESS_PRESETS = {
    "Tokens only (no stopwords, no stemming)": dict(remove_stopwords=False, use_stemming=False),
    "Stopwords removed": dict(remove_stopwords=True, use_stemming=False),
    "Stopwords removed + stemming": dict(remove_stopwords=True, use_stemming=True),
}


def corpus_stats(token_lists: list[list[str]]) -> dict:
    lengths = np.array([len(t) for t in token_lists]) if token_lists else np.array([0])
    counts = Counter(t for toks in token_lists for t in toks)
    total = int(lengths.sum())
    vocab = len(counts)
    top_term, top_freq = counts.most_common(1)[0] if counts else ("-", 0)
    return {
        "documents": len(token_lists), "total tokens": total, "vocabulary size": vocab,
        "type-token ratio": round(vocab / total, 4) if total else 0.0,
        "mean doc length": round(float(lengths.mean()), 1),
        "median doc length": round(float(np.median(lengths)), 1),
        "most frequent term": f"{top_term} ({top_freq})",
    }


# =============================================================================
# 4. FEATURES, KEYWORDS, VISUALISATION, CLASSIFICATION
# =============================================================================

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


@st.cache_data(show_spinner=False)
def build_vectorizer(token_lists: list[list[str]], kind: str, ngram_range=(1, 1)):
    joined = [" ".join(t) for t in token_lists]
    cls = CountVectorizer if kind == "count" else TfidfVectorizer
    vec = cls(analyzer="word", token_pattern=r"\S+", ngram_range=ngram_range, min_df=1)
    matrix = vec.fit_transform(joined)
    return matrix, vec.get_feature_names_out().tolist()


@st.cache_data(show_spinner=False)
def vectoriser_comparison(token_lists: list[list[str]]) -> pd.DataFrame:
    configs = [("Count, unigram", "count", (1, 1)),
               ("TF-IDF, unigram", "tfidf", (1, 1)),
               ("TF-IDF, uni+bigram", "tfidf", (1, 2))]
    rows = []
    for name, kind, ngr in configs:
        started = time.perf_counter()
        matrix, vocab = build_vectorizer(token_lists, kind, ngr)
        rows.append({
            "strategy": name, "vocabulary": len(vocab), "non-zeros": int(matrix.nnz),
            "sparsity %": round(100 * (1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])), 2),
            "avg terms/doc": round(matrix.nnz / max(matrix.shape[0], 1), 1),
            "build seconds": round(time.perf_counter() - started, 3),
        })
    return pd.DataFrame(rows)


def top_terms_for_doc(matrix, vocab: list[str], row: int, k: int = 12) -> pd.DataFrame:
    dense = np.asarray(matrix[row].todense()).ravel()
    idx = np.argsort(dense)[::-1][:k]
    idx = [i for i in idx if dense[i] > 0]
    return pd.DataFrame({"rank": range(1, len(idx) + 1),
                          "term": [vocab[i] for i in idx],
                          "weight": [round(float(dense[i]), 4) for i in idx]})


def global_top_terms(matrix, vocab: list[str], k: int = 20) -> pd.DataFrame:
    sums = np.asarray(matrix.sum(axis=0)).ravel()
    idx = np.argsort(sums)[::-1][:k]
    return pd.DataFrame({"term": [vocab[i] for i in idx],
                          "total weight": [round(float(sums[i]), 3) for i in idx]})


@st.cache_data(show_spinner=False)
def zipf_fit(token_lists: list[list[str]], top_n: int = 2000):
    counts = Counter(t for toks in token_lists for t in toks)
    freqs = [f for _, f in counts.most_common(top_n)]
    ranks = np.arange(1, len(freqs) + 1)
    if len(freqs) < 10:
        return ranks, np.array(freqs), None
    x, y = np.log10(ranks.astype(float)), np.log10(np.array(freqs, dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    return ranks, np.array(freqs), (slope, intercept)


@st.cache_data(show_spinner=False)
def classify_by_domain(token_lists: list[list[str]], domains: list[str]):
    labels = pd.Series(domains)
    keep = labels.map(labels.value_counts()) >= 4
    idx = [i for i, k in enumerate(keep) if k]
    if len(set(labels[keep])) < 2:
        return None
    tokens = [token_lists[i] for i in idx]
    # pandas 3 backs string columns with Arrow by default, and scikit-learn's
    # indexing helpers cannot slice an ArrowExtensionArray — force a plain
    # numpy object array before it reaches train_test_split.
    y = np.asarray(labels[keep].astype(str).tolist(), dtype=object)
    matrix, vocab = build_vectorizer(tokens, "tfidf", (1, 2))
    X_tr, X_te, y_tr, y_te = train_test_split(matrix, y, test_size=0.25,
                                               random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    labels_sorted = sorted(set(y_te))
    cm = confusion_matrix(y_te, pred, labels=labels_sorted)
    return {
        "n_train": X_tr.shape[0], "n_test": X_te.shape[0], "n_classes": len(set(y)),
        "accuracy": round(accuracy_score(y_te, pred), 3),
        "macro_f1": round(f1_score(y_te, pred, average="macro", zero_division=0), 3),
        "labels": labels_sorted, "confusion_matrix": cm,
    }


def bar_chart(labels, values, title, xlabel="", ylabel="", horizontal=False):
    fig, ax = plt.subplots(figsize=(6, max(3, 0.35 * len(labels))))
    if horizontal:
        ax.barh(labels[::-1], values[::-1], color="#3b6ea5")
        ax.set_xlabel(ylabel)
    else:
        ax.bar(labels, values, color="#3b6ea5")
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=45, ha="right")
    ax.set_title(title)
    fig.tight_layout()
    return fig

# =============================================================================
# 5. WEB SEARCHING, RANKING & PAGERANK
# =============================================================================

from sklearn.metrics.pairwise import cosine_similarity


@st.cache_data(show_spinner=False)
def build_search_index(token_lists):
    """
    Build a TF-IDF index over the preprocessed corpus.

    Returns:
        matrix       : document-term TF-IDF matrix
        vocabulary   : vocabulary terms
        vectorizer   : fitted TF-IDF vectorizer
    """
    joined = [" ".join(tokens) for tokens in token_lists]

    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"\S+",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )

    matrix = vectorizer.fit_transform(joined)
    vocabulary = vectorizer.get_feature_names_out().tolist()

    return matrix, vocabulary, vectorizer


@st.cache_data(show_spinner=False)
def search_documents(
    query,
    corpus_records,
    token_lists,
    top_k=10,
    use_query_expansion=True
):
    """
    Search the indexed corpus using TF-IDF cosine similarity.

    Query processing:
        raw query
            ↓
        existing preprocessing pipeline
            ↓
        optional simple query expansion
            ↓
        TF-IDF query vector
            ↓
        cosine similarity
            ↓
        ranked documents
    """

    if not query or not query.strip():
        return pd.DataFrame()

    query_tokens = preprocess(
        query,
        remove_stopwords=True,
        use_stemming=True
    )

    if not query_tokens:
        return pd.DataFrame()

    # ---------------------------------------------------------
    # Simple query expansion
    # ---------------------------------------------------------
    expanded_tokens = list(query_tokens)

    if use_query_expansion:
        # Add frequently occurring terms from documents
        # that share tokens with the query.
        corpus_counter = Counter(
            token
            for tokens in token_lists
            for token in tokens
        )

        for token in query_tokens:
            related = [
                t for t, freq in corpus_counter.most_common(100)
                if (
                    len(t) >= 4
                    and t != token
                    and t[:4] == token[:4]
                )
            ]

            expanded_tokens.extend(related[:2])

    query_text = " ".join(expanded_tokens)

    # ---------------------------------------------------------
    # Build document index
    # ---------------------------------------------------------
    matrix, vocabulary, vectorizer = build_search_index(token_lists)

    # ---------------------------------------------------------
    # Transform query using same fitted vectorizer
    # ---------------------------------------------------------
    query_vector = vectorizer.transform([query_text])

    # ---------------------------------------------------------
    # Cosine similarity
    # ---------------------------------------------------------
    scores = cosine_similarity(
        query_vector,
        matrix
    ).ravel()

    rows = []

    for i, score in enumerate(scores):
        if score <= 0:
            continue

        doc = corpus_records.iloc[i]

        rows.append({
            "doc_index": i,
            "score": float(score),
            "title": doc["title"],
            "domain": doc["domain"],
            "source_type": doc["source_type"],
            "url": doc["url"],
            "doc_id": doc["doc_id"],
            "preview": str(doc["raw_text"])[:350].replace("\n", " ")
        })

    if not rows:
        return pd.DataFrame()

    results = pd.DataFrame(rows)

    results = (
        results
        .sort_values(
            "score",
            ascending=False
        )
        .head(top_k)
        .reset_index(drop=True)
    )

    results.insert(
        0,
        "rank",
        range(1, len(results) + 1)
    )

    results["score"] = results["score"].round(4)

    return results


# def build_document_graph(corpus):
#     """
#     Build a directed document graph.

#     Nodes:
#         acquired documents

#     Edges:
#         document A -> document B when A's URL links to B's URL.

#     Since the current crawler stores metadata but not the complete
#     outgoing-link list, this function reconstructs a lightweight
#     graph from URL/domain relationships and acquired pages.

#     The graph also includes domain-level relationships so that
#     PageRank remains useful when the corpus contains heterogeneous
#     sources.
#     """

#     graph = nx.DiGraph()

#     if corpus is None or corpus.empty:
#         return graph

#     # Add every document as a node.
#     for _, row in corpus.iterrows():
#         graph.add_node(
#             row["doc_id"],
#             title=row["title"],
#             url=row["url"],
#             domain=row["domain"]
#         )

#     # ---------------------------------------------------------
#     # Create links using URL/domain relationships.
#     #
#     # For crawled web documents, documents from the same
#     # domain are connected. This gives PageRank a graph even
#     # when only document metadata is available.
#     # ---------------------------------------------------------
#     web_docs = corpus[
#         corpus["source_type"].astype(str).str.lower() == "web"
#     ]

#     if len(web_docs) > 1:
#         for _, source in web_docs.iterrows():

#             source_id = source["doc_id"]
#             source_domain = source["domain"]

#             candidates = web_docs[
#                 web_docs["doc_id"] != source_id
#             ]

#             # Link to a limited number of documents from the
#             # same domain to prevent a dense O(N^2) graph.
#             same_domain = candidates[
#                 candidates["domain"] == source_domain
#             ].head(10)

#             for _, target in same_domain.iterrows():
#                 graph.add_edge(
#                     source_id,
#                     target["doc_id"]
#                 )

#     # ---------------------------------------------------------
#     # Connect documents from different acquisition sources
#     # through domain/source relationships.
#     # ---------------------------------------------------------
#     for source_type in corpus["source_type"].dropna().unique():

#         group = corpus[
#             corpus["source_type"] == source_type
#         ]

#         ids = group["doc_id"].tolist()

#         for i in range(len(ids) - 1):
#             graph.add_edge(
#                 ids[i],
#                 ids[i + 1]
#             )

#     return graph

def build_document_graph(corpus):
    """
    Build a directed document graph from actual crawled hyperlinks.

    Nodes:
        All documents currently present in the corpus.

    Edges:
        source_doc_id -> target_doc_id when the source document's
        actual outgoing hyperlink points to another document in
        the current corpus.
    """
    graph = nx.DiGraph()

    if corpus is None or corpus.empty:
        return graph

    # ---------------------------------------------------------
    # Add every document as a node
    # ---------------------------------------------------------
    for _, row in corpus.iterrows():
        graph.add_node(
            row["doc_id"],
            title=row["title"],
            url=row["url"],
            domain=row["domain"]
        )

    # ---------------------------------------------------------
    # Map canonical URL -> document ID
    # ---------------------------------------------------------
    url_to_doc_id = {}

    for _, row in corpus.iterrows():
        url = canonicalize_url(row["url"])

        if url:
            url_to_doc_id[url] = row["doc_id"]

    # ---------------------------------------------------------
    # Read actual hyperlinks collected by the crawler
    # ---------------------------------------------------------
    with db() as con:
        rows = con.execute(
            """
            SELECT source_doc_id, target_url
            FROM document_links
            """
        ).fetchall()

    # ---------------------------------------------------------
    # Convert URL links into document-to-document edges
    # ---------------------------------------------------------
    for source_doc_id, target_url in rows:

        if source_doc_id not in graph:
            continue

        target_url = canonicalize_url(target_url)

        target_doc_id = url_to_doc_id.get(target_url)

        if target_doc_id is not None:
            graph.add_edge(
                source_doc_id,
                target_doc_id
            )

    return graph

@st.cache_data(show_spinner=False)
def calculate_pagerank(corpus_records):
    """
    Calculate PageRank over the document graph.
    """

    graph = build_document_graph(corpus_records)

    if graph.number_of_nodes() == 0:
        return pd.DataFrame()

    pagerank_scores = nx.pagerank(
        graph,
        alpha=0.85,
        max_iter=100,
        tol=1e-6
    )

    rows = []

    for _, row in corpus_records.iterrows():

        rows.append({
            "doc_id": row["doc_id"],
            "title": row["title"],
            "domain": row["domain"],
            "pagerank": pagerank_scores.get(
                row["doc_id"],
                0.0
            )
        })

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "pagerank",
        ascending=False
    ).reset_index(drop=True)

    result.insert(
        0,
        "rank",
        range(1, len(result) + 1)
    )

    result["pagerank"] = result["pagerank"].round(6)

    return result


def combine_relevance_and_pagerank(
    search_results,
    pagerank_df,
    relevance_weight=0.75
):
    """
    Combine query relevance with PageRank.

    Combined score:

        alpha * normalized_TFIDF
        +
        (1-alpha) * normalized_PageRank
    """

    if search_results.empty:
        return search_results

    if pagerank_df.empty:
        return search_results

    results = search_results.copy()

    results = results.merge(
        pagerank_df[
            ["doc_id", "pagerank"]
        ],
        on="doc_id",
        how="left"
    )

    results["pagerank"] = (
        results["pagerank"]
        .fillna(0.0)
    )

    # ---------------------------------------------------------
    # Normalize relevance scores
    # ---------------------------------------------------------
    max_score = results["score"].max()

    if max_score > 0:
        results["norm_relevance"] = (
            results["score"] / max_score
        )
    else:
        results["norm_relevance"] = 0.0

    # ---------------------------------------------------------
    # Normalize PageRank
    # ---------------------------------------------------------
    max_pr = results["pagerank"].max()

    if max_pr > 0:
        results["norm_pagerank"] = (
            results["pagerank"] / max_pr
        )
    else:
        results["norm_pagerank"] = 0.0

    # ---------------------------------------------------------
    # Combined ranking
    # ---------------------------------------------------------
    results["combined_score"] = (
        relevance_weight * results["norm_relevance"]
        +
        (1 - relevance_weight) * results["norm_pagerank"]
    )

    results = results.sort_values(
        "combined_score",
        ascending=False
    ).reset_index(drop=True)

    results.insert(
        0,
        "combined_rank",
        range(1, len(results) + 1)
    )

    results["combined_score"] = (
        results["combined_score"].round(4)
    )

    results["pagerank"] = (
        results["pagerank"].round(6)
    )

    return results


def ranking_comparison_table(
    search_results,
    pagerank_df
):
    """
    Show how relevance ranking and PageRank ranking differ.
    """

    if search_results.empty or pagerank_df.empty:
        return pd.DataFrame()

    relevance = search_results[
        [
            "doc_id",
            "title",
            "score"
        ]
    ].copy()

    relevance["tfidf_rank"] = (
        relevance["score"]
        .rank(
            method="min",
            ascending=False
        )
        .astype(int)
    )

    pr = pagerank_df[
        [
            "doc_id",
            "pagerank"
        ]
    ].copy()

    pr["pagerank_rank"] = (
        pr["pagerank"]
        .rank(
            method="min",
            ascending=False
        )
        .astype(int)
    )

    result = relevance.merge(
        pr,
        on="doc_id",
        how="left"
    )

    result["rank_change"] = (
        result["tfidf_rank"]
        -
        result["pagerank_rank"]
    )

    result = result.sort_values(
        "tfidf_rank"
    )

    return result[
        [
            "tfidf_rank",
            "pagerank_rank",
            "rank_change",
            "title",
            "score",
            "pagerank"
        ]
    ].reset_index(drop=True)


# =============================================================================
# 6. SECTION E — CONTENT-BASED RECOMMENDER SYSTEM
# =============================================================================

@st.cache_data(show_spinner=False)
def build_recommendation_index(token_lists):
    """
    Build a TF-IDF representation of all documents for content-based
    recommendation.

    Returns:
        matrix      : document-term TF-IDF matrix
        vectorizer  : fitted TF-IDF vectorizer
    """

    joined = [" ".join(tokens) for tokens in token_lists]

    if not joined or not any(joined):
        return None, None

    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"\S+",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )

    matrix = vectorizer.fit_transform(joined)

    return matrix, vectorizer


@st.cache_data(show_spinner=False)
def recommend_similar_documents(
    selected_index,
    corpus_records,
    token_lists,
    top_k=5
):
    """
    Content-based recommendation.

    The selected document is represented using TF-IDF.
    Cosine similarity is calculated against every other document.
    The selected document itself is excluded.

    Returns:
        DataFrame containing recommended documents and similarity scores.
    """

    if (
        corpus_records is None
        or len(corpus_records) == 0
        or token_lists is None
        or len(token_lists) == 0
    ):
        return pd.DataFrame()

    if selected_index < 0 or selected_index >= len(corpus_records):
        return pd.DataFrame()

    matrix, vectorizer = build_recommendation_index(token_lists)

    if matrix is None:
        return pd.DataFrame()

    # Selected document vector
    selected_vector = matrix[selected_index]

    # Similarity against every document
    similarities = cosine_similarity(
        selected_vector,
        matrix
    ).ravel()

    # Do not recommend the document itself
    similarities[selected_index] = -1.0

    # Number of recommendations
    k = min(int(top_k), len(corpus_records) - 1)

    if k <= 0:
        return pd.DataFrame()

    # Highest similarity first
    recommended_indices = np.argsort(
        similarities
    )[::-1][:k]

    rows = []

    for rank, idx in enumerate(recommended_indices, start=1):

        score = float(similarities[idx])

        # Ignore zero-similarity documents
        if score <= 0:
            continue

        doc = corpus_records.iloc[idx]

        rows.append({
            "rank": rank,
            "doc_index": int(idx),
            "similarity": round(score, 4),
            "title": doc["title"],
            "domain": doc["domain"],
            "source_type": doc["source_type"],
            "url": doc["url"],
            "doc_id": doc["doc_id"],
            "preview": str(
                doc["raw_text"]
            )[:350].replace("\n", " ")
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# =============================================================================
# 6. EVALUATION METRICS
# =============================================================================

def precision_recall_f1(relevant, retrieved):
    """
    Calculate Precision, Recall and F1-score.

    relevant  : set of ground-truth relevant document IDs
    retrieved : list/set of retrieved document IDs
    """

    relevant = set(relevant)
    retrieved = list(retrieved)

    if not retrieved:
        precision = 0.0
    else:
        precision = sum(
            1 for doc_id in retrieved
            if doc_id in relevant
        ) / len(retrieved)

    if not relevant:
        recall = 0.0
    else:
        recall = sum(
            1 for doc_id in retrieved
            if doc_id in relevant
        ) / len(relevant)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def precision_at_k(relevant, retrieved, k):
    """
    Precision@K
    """

    relevant = set(relevant)
    retrieved_k = list(retrieved)[:k]

    if not retrieved_k:
        return 0.0

    hits = sum(
        1 for doc_id in retrieved_k
        if doc_id in relevant
    )

    return hits / len(retrieved_k)


def recall_at_k(relevant, retrieved, k):
    """
    Recall@K
    """

    relevant = set(relevant)
    retrieved_k = list(retrieved)[:k]

    if not relevant:
        return 0.0

    hits = sum(
        1 for doc_id in retrieved_k
        if doc_id in relevant
    )

    return hits / len(relevant)


def average_precision(relevant, retrieved):
    """
    Average Precision (AP).

    AP = average of precision values at ranks
    where relevant documents are retrieved.
    """

    relevant = set(relevant)

    if not relevant:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for rank, doc_id in enumerate(retrieved, start=1):

        if doc_id in relevant:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / len(relevant)


def mean_average_precision(relevance_sets, retrieved_lists):
    """
    Mean Average Precision (MAP).

    relevance_sets:
        list of ground-truth relevant document sets

    retrieved_lists:
        list of ranked retrieval results
    """

    if not relevance_sets:
        return 0.0

    ap_scores = []

    for relevant, retrieved in zip(
        relevance_sets,
        retrieved_lists
    ):
        ap_scores.append(
            average_precision(relevant, retrieved)
        )

    if not ap_scores:
        return 0.0

    return sum(ap_scores) / len(ap_scores)


def reciprocal_rank(relevant, retrieved):
    """
    Reciprocal Rank for one query.
    """

    relevant = set(relevant)

    for rank, doc_id in enumerate(retrieved, start=1):

        if doc_id in relevant:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(relevance_sets, retrieved_lists):
    """
    Mean Reciprocal Rank (MRR).
    """

    if not relevance_sets:
        return 0.0

    rr_scores = []

    for relevant, retrieved in zip(
        relevance_sets,
        retrieved_lists
    ):
        rr_scores.append(
            reciprocal_rank(relevant, retrieved)
        )

    if not rr_scores:
        return 0.0

    return sum(rr_scores) / len(rr_scores)


def ndcg_at_k(relevant, retrieved, k):
    """
    NDCG@K for binary relevance.

    Relevant document = 1
    Non-relevant document = 0
    """

    relevant = set(relevant)
    retrieved_k = list(retrieved)[:k]

    if not retrieved_k:
        return 0.0

    dcg = 0.0

    for rank, doc_id in enumerate(
        retrieved_k,
        start=1
    ):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_count = min(len(relevant), k)

    if ideal_count == 0:
        return 0.0

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


def evaluate_search_results(
    relevant_docs,
    retrieved_docs,
    k=10
):
    """
    Calculate all required evaluation metrics
    for one query.
    """

    precision, recall, f1 = precision_recall_f1(
        relevant_docs,
        retrieved_docs
    )

    p_at_k = precision_at_k(
        relevant_docs,
        retrieved_docs,
        k
    )

    r_at_k = recall_at_k(
        relevant_docs,
        retrieved_docs,
        k
    )

    ap = average_precision(
        relevant_docs,
        retrieved_docs
    )

    rr = reciprocal_rank(
        relevant_docs,
        retrieved_docs
    )

    ndcg = ndcg_at_k(
        relevant_docs,
        retrieved_docs,
        k
    )

    return {
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Precision@K": p_at_k,
        "Recall@K": r_at_k,
        "AP": ap,
        "RR": rr,
        "NDCG@K": ndcg
    }

# =============================================================================
# STREAMLIT UI
# =============================================================================

st.set_page_config(page_title="Group 31 Information Retrieval assignment 2", layout="wide")
st.title("Group 31 Information Retrieval assignment 2")
st.caption("Acquire a corpus from three heterogeneous sources (web crawl, public dataset,"
           "public API), then preprocess, vectorise, profile and classify it.")

st.session_state.setdefault("corpus", None)
st.session_state.setdefault("tokens", None)

if st.session_state.corpus is None and store_counts()[0] > 0:
    st.session_state.corpus = load_corpus()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "1 · Acquire & store",
        "2 · Preprocess",
        "3 · Features & keywords",
        "4 · Analytics & classification",
        "5 · Web Search & Ranking",
        "6 · Recommender System",
        "7 · Evaluation"
    ]
)

# --------------------------------------------------------------------------- #
# Tab 1 — acquisition: web crawl, public dataset, public API
# --------------------------------------------------------------------------- #
DEFAULT_SEEDS = (
    "https://en.wikipedia.org/wiki/Web_crawler\n"
    "https://en.wikipedia.org/wiki/Information_retrieval\n"
    "https://docs.python.org/3/library/urllib.robotparser.html\n"
    "https://docs.python.org/3/library/sqlite3.html\n"
    "https://www.geeksforgeeks.org/python/page-rank-algorithm-implementation"
)


def show_ingest_metrics(stats: Counter) -> None:
    k1, k2, k3 = st.columns(3)
    k1.metric("New documents stored", stats["stored"])
    k2.metric("Duplicates skipped (URL + content)", stats["duplicate_url"] + stats["duplicate_doc"])
    k3.metric("Skipped as too short (<40 words)", stats["too_short"])


with tab1:
    st.subheader("Acquire documents")
    st.caption("Three independent acquisition paths feed the same SQLite store: crawling "
               "live web pages, ingesting a public dataset file, and querying a public API. "
               "Every document — whatever its source — goes through the same URL-level and "
               "content-hash duplicate checks before it's stored, and the rest of the "
               "pipeline (preprocessing, features, classification) doesn't care where it "
               "came from.")

    method = st.radio("Acquisition method", ["Web crawl", "Public dataset", "Public API (Wikipedia)"],
                       horizontal=True, key="acq_method")

    if method == "Web crawl":
        st.markdown("**Source: live web pages, fetched by following links (breadth-first).**")
        st.caption("One URL per line. The defaults below span two different domains, so "
                   "step 4's classifier has two classes to learn right away — edit freely.")
        seeds_text = st.text_area("Seed URLs", value=DEFAULT_SEEDS, height=110, key="seeds_input")

        c1, c2, c3, c4 = st.columns(4)
        max_depth = c1.slider("Max crawl depth", 0, 3, 1)
        max_pages = c2.number_input("Max pages", 5, 500, 30, step=5)
        same_domain = c3.checkbox("Stay on seed domains", value=True)
        delay = c4.number_input("Delay between requests (s)", 0.0, 5.0, 0.5, step=0.5)

        if st.button("Run crawl", type="primary"):
            seeds = [s.strip() for s in seeds_text.splitlines() if s.strip()]
            if not seeds:
                st.warning("Add at least one seed URL.")
            else:
                progress = st.progress(0.0)
                status = st.empty()

                def log_fn(done, total, url):
                    progress.progress(min(done / total, 1.0))
                    status.text(f"Fetched {done}/{total}: {url}")

                with st.spinner("Crawling…"):
                    docs, stats, seen, log_rows = crawl(
                        seeds, max_depth, int(max_pages), same_domain, delay,
                        known_urls(), known_hashes(), log_fn)
                    store_documents(docs)
                    store_document_links(docs)
                    store_urls(seen)
                    st.session_state.corpus = load_corpus()
                progress.empty()
                status.empty()

                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("Pages fetched", stats["fetched"])
                k2.metric("New documents stored", stats["stored"])
                k3.metric("Duplicate URLs skipped", stats["duplicate_url"])
                k4.metric("Duplicate documents skipped", stats["duplicate_doc"])
                k5.metric("Errors / blocked", stats["errors"] + stats["robots_blocked"])

                with st.expander("Crawl log"):
                    st.dataframe(pd.DataFrame(log_rows), height=280, use_container_width=True)

    elif method == "Public dataset":
        st.markdown("**Source: a public dataset file — use the bundled sample out of the "
                     "box, upload your own locally, or fetch one live from a URL.**")
        dataset_mode = st.radio("Dataset source",
                                 ["Use bundled sample dataset", "Upload file", "Fetch from URL"],
                                 horizontal=True, key="dataset_mode")

        c1, c2, c3 = st.columns(3)
        text_col = c1.text_input("Text column name", "text")
        title_col = c2.text_input("Title column name (optional)", "title")
        max_rows = c3.number_input("Max rows to ingest", 5, 2000, 100, step=5)

        if dataset_mode == "Use bundled sample dataset":
            st.caption(f"`{SAMPLE_DATASET_PATH.name}` — 10 short reference documents on "
                       "information retrieval topics, shipped with the app so this source "
                       "works with zero setup and no internet access.")
            source_name = st.text_input("Dataset label (used as the class/domain tag)",
                                         value="sample_documents", key="bundled_label")
            if st.button("Ingest bundled sample dataset", type="primary"):
                df = load_tabular(SAMPLE_DATASET_PATH, SAMPLE_DATASET_PATH.name)
                if text_col not in df.columns:
                    st.error(f"Column '{text_col}' not found. Available columns: {', '.join(df.columns)}")
                else:
                    items = dataset_rows_to_items(df, text_col, title_col, source_name, int(max_rows))
                    docs, stats, seen = build_documents(items, "dataset", known_urls(), known_hashes())
                    store_documents(docs)
                    store_urls(seen)
                    st.session_state.corpus = load_corpus()
                    show_ingest_metrics(stats)
        elif dataset_mode == "Upload file":
            uploaded = st.file_uploader("CSV or JSON file", type=["csv", "json"])
            source_name = st.text_input("Dataset label (used as the class/domain tag)",
                                         value=(uploaded.name.rsplit(".", 1)[0] if uploaded else "uploaded_dataset"))
            if st.button("Ingest uploaded dataset", type="primary"):
                if not uploaded:
                    st.warning("Choose a file first.")
                else:
                    try:
                        df = load_tabular(uploaded, uploaded.name)
                    except Exception as exc:
                        st.error(f"Could not parse file: {exc}")
                    else:
                        if text_col not in df.columns:
                            st.error(f"Column '{text_col}' not found. Available columns: {', '.join(df.columns)}")
                        else:
                            items = dataset_rows_to_items(df, text_col, title_col, source_name, int(max_rows))
                            docs, stats, seen = build_documents(items, "dataset", known_urls(), known_hashes())
                            store_documents(docs)
                            store_urls(seen)
                            st.session_state.corpus = load_corpus()
                            show_ingest_metrics(stats)
        else:
            dataset_url = st.text_input("Public dataset URL (.csv or .json)")
            default_label = urlparse(dataset_url).netloc if dataset_url else "url_dataset"
            source_name = st.text_input("Dataset label (used as the class/domain tag)", value=default_label)
            if st.button("Fetch & ingest dataset", type="primary"):
                if not dataset_url.strip():
                    st.warning("Enter a dataset URL first.")
                else:
                    try:
                        with st.spinner("Downloading…"):
                            resp = requests.get(dataset_url, timeout=15, headers={"User-Agent": USER_AGENT})
                            resp.raise_for_status()
                            df = load_tabular(io.StringIO(resp.text), dataset_url)
                    except Exception as exc:
                        st.error(f"Could not fetch/parse dataset: {exc}")
                    else:
                        if text_col not in df.columns:
                            st.error(f"Column '{text_col}' not found. Available columns: {', '.join(df.columns)}")
                        else:
                            items = dataset_rows_to_items(df, text_col, title_col, source_name, int(max_rows))
                            docs, stats, seen = build_documents(items, "dataset", known_urls(), known_hashes())
                            store_documents(docs)
                            store_urls(seen)
                            st.session_state.corpus = load_corpus()
                            show_ingest_metrics(stats)

    else:  # Public API (Wikipedia)
        st.markdown("**Source: the Wikipedia MediaWiki API — a JSON search + extract "
                     "endpoint, not HTML scraping.**")
        st.caption("Search and extracts are combined into a single HTTP request "
                   "regardless of how many articles you ask for, so this stays light "
                   "by default — no per-result request fan-out.")
        c1, c2 = st.columns([3, 1])
        query = c1.text_input("Search query", "Information retrieval")
        max_results = c2.number_input("Max articles", 1, 50, 5)
        if st.button("Fetch from Wikipedia API", type="primary"):
            if not query.strip():
                st.warning("Enter a search query first.")
            else:
                try:
                    with st.spinner("Querying Wikipedia API…"):
                        items = fetch_wikipedia(query, int(max_results))
                        docs, stats, seen = build_documents(items, "api", known_urls(), known_hashes())
                        store_documents(docs)
                        store_urls(seen)
                        st.session_state.corpus = load_corpus()
                except Exception as exc:
                    st.error(f"Wikipedia API request failed: {exc}")
                else:
                    show_ingest_metrics(stats)

    st.divider()
    st.subheader("Stored corpus")
    n_docs, n_urls, n_domains = store_counts()
    m1, m2, m3 = st.columns(3)
    m1.metric("Documents stored", n_docs)
    m2.metric("URLs seen (all runs)", n_urls)
    m3.metric("Distinct domains", n_domains)

    if st.button("Reset store (delete all acquired data)"):
        reset_store()
        st.session_state.corpus = None
        st.session_state.tokens = None
        st.rerun()

    if n_docs:
        st.caption("Documents by acquisition source")
        st.dataframe(source_breakdown(), hide_index=True, use_container_width=True)

        with st.expander("Sample of stored documents"):
            preview = load_corpus().head(20).copy()
            preview["raw_text"] = preview["raw_text"].str.slice(0, 180) + "…"
            st.dataframe(preview[["doc_id", "title", "domain", "source_type", "n_words", "raw_text"]],
                         height=320, use_container_width=True)
        with st.expander("Storage layout — content and metadata are separate tables"):
            st.markdown("`documents` holds the payload (raw text, SHA-256 content hash, "
                         "word/char counts). `metadata` holds the descriptors (URL, domain, "
                         "title, crawl depth, seed, fetch time, and which of the three "
                         "acquisition methods produced it). They join on `doc_id`.")
            a, b = st.columns(2)
            a.caption("documents")
            a.dataframe(pd.DataFrame({"column": ["doc_id", "raw_text", "content_hash", "n_chars", "n_words"]}),
                        hide_index=True)
            b.caption("metadata")
            b.dataframe(pd.DataFrame({"column": ["doc_id", "url", "domain", "title", "crawl_depth",
                                                   "seed_url", "http_status", "fetched_at",
                                                   "source_type", "source_detail"]}),
                        hide_index=True)

# --------------------------------------------------------------------------- #
# Tab 2 — preprocessing
# --------------------------------------------------------------------------- #
with tab2:
    if st.session_state.corpus is None:
        st.info("Load a working corpus on the **Crawl & store** tab first.")
    else:
        corpus = st.session_state.corpus
        st.subheader("Preprocessing options")
        c1, c2 = st.columns(2)
        remove_sw = c1.checkbox("Remove stopwords", value=True)
        use_stem = c2.checkbox("Apply stemming", value=True)

        tokens, seconds = preprocess_corpus(corpus["raw_text"].tolist(), remove_sw, use_stem)
        st.session_state.tokens = tokens
        st.caption(f"Preprocessed {len(tokens)} documents in {seconds:.2f}s "
                   "(recomputes automatically whenever the options above change).")
        st.dataframe(pd.DataFrame(corpus_stats(tokens).items(),
                                   columns=["statistic", "value"]),
                     hide_index=True, use_container_width=True)

        st.subheader("Comparative analysis of preprocessing strategies")
        rows = []
        for name, cfg in PREPROCESS_PRESETS.items():
            preset_tokens, secs = preprocess_corpus(corpus["raw_text"].tolist(), **cfg)
            stats = corpus_stats(preset_tokens)
            rows.append({"strategy": name, "vocabulary": stats["vocabulary size"],
                         "avg tokens/doc": stats["mean doc length"],
                         "type-token ratio": stats["type-token ratio"],
                         "seconds": round(secs, 3)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        st.subheader("Stage-by-stage inspector")
        i = st.selectbox("Document", range(len(corpus)),
                            format_func=lambda i: corpus.iloc[i]["title"][:80])
        raw = corpus.iloc[i]["raw_text"]
        cleaned = clean_text(raw)
        toks = tokenize(cleaned)
        st.code(f"0. raw          : {raw[:300]}…\n"
                f"1. cleaned      : {cleaned[:300]}…\n"
                f"2. tokenised    : {toks[:25]}\n"
                f"3. final tokens : {tokens[i][:25]}")

# --------------------------------------------------------------------------- #
# Tab 3 — features & keywords
# --------------------------------------------------------------------------- #
with tab3:
    if not st.session_state.tokens:
        st.info("Preprocess the corpus on the **Preprocess** tab first.")
    else:
        corpus = st.session_state.corpus
        tokens = st.session_state.tokens

        st.subheader("Feature matrix")
        tfidf_matrix, vocab = build_vectorizer(tokens, "tfidf")
        st.caption(f"TF-IDF matrix: {tfidf_matrix.shape[0]} docs x {tfidf_matrix.shape[1]} terms.")

        st.subheader("Comparative analysis of feature extraction strategies")
        st.dataframe(vectoriser_comparison(tokens), hide_index=True, use_container_width=True)

        st.subheader("Keyword extraction (TF-IDF, per document)")
        i = st.selectbox("Document", range(len(corpus)),
                          format_func=lambda i: corpus.iloc[i]["title"][:80], key="kw_doc")
        st.dataframe(top_terms_for_doc(tfidf_matrix, vocab, i),
                     hide_index=True, use_container_width=True)

        st.subheader("Top keywords across the corpus")
        top = global_top_terms(tfidf_matrix, vocab, 20)
        st.pyplot(bar_chart(top["term"].tolist(), top["total weight"].tolist(),
                             "Top 20 terms by summed TF-IDF weight", ylabel="weight", horizontal=True))

# --------------------------------------------------------------------------- #
# Tab 4 — corpus analytics & classification
# --------------------------------------------------------------------------- #
with tab4:
    if not st.session_state.tokens:
        st.info("Preprocess the corpus on the **Preprocess** tab first.")
    else:
        corpus = st.session_state.corpus
        tokens = st.session_state.tokens

        st.subheader("Document profiling")
        profiles = pd.DataFrame([{
            "doc_id": corpus.iloc[i]["doc_id"], "title": corpus.iloc[i]["title"][:60],
            "domain": corpus.iloc[i]["domain"], "source_type": corpus.iloc[i]["source_type"],
            "raw_words": corpus.iloc[i]["n_words"],
            "tokens_after_preprocessing": len(tokens[i]),
            "unique_terms": len(set(tokens[i])),
            "top_terms": ", ".join(t for t, _ in Counter(tokens[i]).most_common(5)),
        } for i in range(len(corpus))])
        st.dataframe(profiles, hide_index=True, use_container_width=True, height=280)

        st.subheader("Corpus characteristics")
        v1, v2, v3 = st.columns(3)
        with v1:
            by_source = corpus["source_type"].value_counts()
            st.pyplot(bar_chart(by_source.index.tolist(), by_source.values.tolist(),
                                 "Documents per acquisition source", ylabel="documents"))
        with v2:
            by_domain = corpus["domain"].value_counts()
            st.pyplot(bar_chart(by_domain.index.tolist(), by_domain.values.tolist(),
                                 "Documents per domain", ylabel="documents"))
        with v3:
            lengths = [len(t) for t in tokens]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(lengths, bins=20, color="#3b6ea5")
            ax.set_title("Document length distribution (tokens)")
            ax.set_xlabel("tokens per document")
            ax.set_ylabel("documents")
            fig.tight_layout()
            st.pyplot(fig)

        st.subheader("Zipf's law — term frequency vs. rank")
        ranks, freqs, fit = zipf_fit(tokens)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.loglog(ranks, freqs, marker=".", linestyle="none", color="#3b6ea5", markersize=3)
        if fit:
            slope, intercept = fit
            ax.loglog(ranks, 10 ** (slope * np.log10(ranks) + intercept), color="#c0392b",
                       label=f"fit slope={slope:.2f}")
            ax.legend()
        ax.set_xlabel("rank (log)")
        ax.set_ylabel("frequency (log)")
        ax.set_title("Zipf plot")
        fig.tight_layout()
        st.pyplot(fig)

        st.divider()
        st.subheader("Document classification (predict domain from text)")
        domains = corpus["domain"].tolist()
        n_eligible = sum(1 for c in Counter(domains).values() if c >= 4)
        if n_eligible < 2:
            st.info("Classification needs at least two domains with 4+ documents each. "
                     "Acquire from more than one site, dataset, or API query — mixing "
                     "sources on the **Acquire & store** tab is the fastest way to get "
                     "there — then come back here.")
        else:
            result = classify_by_domain(tokens, domains)
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Accuracy", result["accuracy"])
            r2.metric("Macro F1", result["macro_f1"])
            r3.metric("Train / test docs", f"{result['n_train']} / {result['n_test']}")
            r4.metric("Classes", result["n_classes"])

            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(result["confusion_matrix"], cmap="Blues")
            ax.set_xticks(range(len(result["labels"])), result["labels"], rotation=45, ha="right")
            ax.set_yticks(range(len(result["labels"])), result["labels"])
            ax.set_xlabel("predicted")
            ax.set_ylabel("actual")
            ax.set_title("Confusion matrix")
            for r in range(len(result["labels"])):
                for c in range(len(result["labels"])):
                    ax.text(c, r, result["confusion_matrix"][r, c], ha="center", va="center")
            fig.colorbar(im)
            fig.tight_layout()
            st.pyplot(fig)

# --------------------------------------------------------------------------- #
# Tab 5 — Section D: Web Search & Ranking
# --------------------------------------------------------------------------- #

with tab5:

    st.header("Web Searching & Ranking")

    st.caption(
        "Search the indexed document collection using TF-IDF relevance, "
        "query expansion and PageRank-based ranking."
    )

    # -----------------------------------------------------------------------
    # Corpus requirement
    # -----------------------------------------------------------------------

    if st.session_state.corpus is None:

        st.warning(
            "No corpus is currently loaded. "
            "Go to **Acquire & store** and acquire documents first."
        )

    elif st.session_state.tokens is None:

        st.warning(
            "The corpus exists, but it has not been preprocessed yet."
        )

        st.info(
            "Go to **Preprocess**, enable your preprocessing options, "
            "and then return here."
        )

    else:

        corpus = st.session_state.corpus
        tokens = st.session_state.tokens

        # ================================================================
        # SEARCH INTERFACE
        # ================================================================

        st.subheader("1. Search indexed collection")

        c1, c2, c3 = st.columns([5, 1, 2])

        with c1:

            query = st.text_input(
                "Search query",
                value="information retrieval",
                placeholder="Enter your search query..."
            )

        with c2:

            top_k = st.number_input(
                "Top-K",
                min_value=1,
                max_value=50,
                value=10,
                step=1
            )

        with c3:

            query_expansion = st.checkbox(
                "Query expansion",
                value=True
            )

        search_button = st.button(
            "Search",
            type="primary",
            use_container_width=True
        )

        # ================================================================
        # SEARCH EXECUTION
        # ================================================================

        if search_button:

            if not query.strip():

                st.warning(
                    "Please enter a search query."
                )

            else:

                with st.spinner(
                    "Searching indexed collection..."
                ):

                    results = search_documents(
                        query=query,
                        corpus_records=corpus,
                        token_lists=tokens,
                        top_k=int(top_k),
                        use_query_expansion=query_expansion
                    )

                st.session_state["search_results"] = results
                st.session_state["search_query"] = query

        # ================================================================
        # DISPLAY SEARCH RESULTS
        # ================================================================

        if "search_results" in st.session_state:

            results = st.session_state["search_results"]

            if results.empty:

                st.info(
                    "No relevant documents were found for this query."
                )

            else:

                st.success(
                    f"Found {len(results)} relevant documents."
                )

                st.subheader(
                    "Ranked retrieval results"
                )

                display_results = results[
                    [
                        "rank",
                        "title",
                        "score",
                        "domain",
                        "source_type",
                        "url"
                    ]
                ].copy()

                display_results["score"] = (
                    display_results["score"]
                    .map(lambda x: f"{x:.4f}")
                )

                st.dataframe(
                    display_results,
                    hide_index=True,
                    use_container_width=True
                )

                # --------------------------------------------------------
                # Detailed result view
                # --------------------------------------------------------

                st.subheader(
                    "Result details"
                )

                selected_rank = st.selectbox(
                    "Select a result",
                    options=results["rank"].tolist(),
                    format_func=lambda x:
                        f"Rank {x} — "
                        f"{results.loc[results['rank'] == x, 'title'].iloc[0][:90]}"
                )

                selected = results[
                    results["rank"] == selected_rank
                ].iloc[0]

                d1, d2 = st.columns([2, 1])

                with d1:

                    st.markdown(
                        f"### {selected['title']}"
                    )

                    st.write(
                        selected["preview"]
                    )

                with d2:

                    st.metric(
                        "TF-IDF relevance",
                        f"{selected['score']:.4f}"
                    )

                    st.write(
                        f"**Domain:** {selected['domain']}"
                    )

                    st.write(
                        f"**Source:** {selected['source_type']}"
                    )

                    st.write(
                        f"**Document ID:** {selected['doc_id']}"
                    )

                    st.write(
                        f"**URL:** {selected['url']}"
                    )

                # --------------------------------------------------------
                # Relevance ranking visualization
                # --------------------------------------------------------

                st.subheader(
                    "Relevance ranking visualization"
                )

                fig, ax = plt.subplots(
                    figsize=(9, 5)
                )

                plot_results = results.head(
                    min(10, len(results))
                ).iloc[::-1]

                ax.barh(
                    plot_results["title"].str.slice(0, 45),
                    plot_results["score"]
                )

                ax.set_xlabel(
                    "TF-IDF cosine similarity"
                )

                ax.set_ylabel(
                    "Document"
                )

                ax.set_title(
                    "Top documents ranked by query relevance"
                )

                fig.tight_layout()

                st.pyplot(fig)

        # ================================================================
        # PAGE RANK
        # ================================================================

        st.divider()

        st.subheader(
            "2. PageRank analysis"
        )

        st.write(
            "PageRank measures the importance of documents in the "
            "document-link graph. It is independent of the current "
            "search query."
        )

        calculate_pr = st.button(
            "Calculate PageRank",
            type="secondary"
        )

        if calculate_pr:

            with st.spinner(
                "Building document graph and calculating PageRank..."
            ):

                pr_df = calculate_pagerank(
                    corpus
                )

            st.session_state["pagerank"] = pr_df

        if "pagerank" in st.session_state:

            pr_df = st.session_state["pagerank"]

            if pr_df.empty:

                st.info(
                    "PageRank could not be calculated because "
                    "the corpus is empty."
                )

            else:

                st.metric(
                    "Documents in PageRank graph",
                    len(pr_df)
                )

                st.dataframe(
                    pr_df.head(20),
                    hide_index=True,
                    use_container_width=True
                )

                # --------------------------------------------------------
                # PageRank visualization
                # --------------------------------------------------------

                plot_pr = pr_df.head(
                    min(15, len(pr_df))
                ).iloc[::-1]

                fig, ax = plt.subplots(
                    figsize=(9, 5)
                )

                ax.barh(
                    plot_pr["title"].str.slice(0, 45),
                    plot_pr["pagerank"]
                )

                ax.set_xlabel(
                    "PageRank score"
                )

                ax.set_ylabel(
                    "Document"
                )

                ax.set_title(
                    "Top documents by PageRank"
                )

                fig.tight_layout()

                st.pyplot(fig)

        # ================================================================
        # COMBINED RANKING
        # ================================================================

        if (
            "search_results" in st.session_state
            and not st.session_state["search_results"].empty
            and "pagerank" in st.session_state
        ):

            st.divider()

            st.subheader(
                "3. Combined relevance + PageRank ranking"
            )

            st.write(
                "The final ranking combines query relevance with "
                "document importance."
            )

            weight = st.slider(
                "Weight given to query relevance",
                min_value=0.0,
                max_value=1.0,
                value=0.75,
                step=0.05
            )

            combined = combine_relevance_and_pagerank(
                st.session_state["search_results"],
                st.session_state["pagerank"],
                relevance_weight=weight
            )

            st.dataframe(
                combined[
                    [
                        "combined_rank",
                        "title",
                        "score",
                        "pagerank",
                        "combined_score",
                        "domain"
                    ]
                ],
                hide_index=True,
                use_container_width=True
            )

            # ------------------------------------------------------------
            # Compare rankings
            # ------------------------------------------------------------

            st.subheader(
                "Ranking comparison"
            )

            comparison = ranking_comparison_table(
                st.session_state["search_results"],
                st.session_state["pagerank"]
            )

            if not comparison.empty:

                st.dataframe(
                    comparison,
                    hide_index=True,
                    use_container_width=True
                )

            # ------------------------------------------------------------
            # Combined ranking visualization
            # ------------------------------------------------------------

            plot_combined = combined.head(
                min(10, len(combined))
            ).iloc[::-1]

            fig, ax = plt.subplots(
                figsize=(9, 5)
            )

            ax.barh(
                plot_combined["title"].str.slice(0, 45),
                plot_combined["combined_score"]
            )

            ax.set_xlabel(
                "Combined ranking score"
            )

            ax.set_ylabel(
                "Document"
            )

            ax.set_title(
                "Final ranking: TF-IDF relevance + PageRank"
            )

            fig.tight_layout()

            st.pyplot(fig)

            # ------------------------------------------------------------
            # Explanation
            # ------------------------------------------------------------

            st.info(
                f"""
                **Ranking interpretation**

                Query relevance weight: **{weight:.2f}**

                PageRank weight: **{1 - weight:.2f}**

                A high TF-IDF score means the document is highly relevant
                to the user's query.

                A high PageRank score means the document is considered
                important within the document graph.

                The combined ranking demonstrates why relevance and
                authority can produce different rankings.
                """
            )

# =============================================================================
# TAB 6 — SECTION E: RECOMMENDER SYSTEM
# =============================================================================

with tab6:

    st.header("Recommender System")

    st.caption(
        "Content-based document recommendation using TF-IDF "
        "representations and cosine similarity."
    )

    # -------------------------------------------------------------------------
    # CORPUS CHECK
    # -------------------------------------------------------------------------

    if st.session_state.corpus is None:

        st.warning(
            "No corpus is currently loaded. "
            "Go to **Acquire & store** and acquire documents first."
        )

    elif st.session_state.tokens is None:

        st.warning(
            "The corpus exists, but it has not been preprocessed yet."
        )

        st.info(
            "Go to **Preprocess**, run preprocessing, "
            "and then return to the Recommender System."
        )

    else:

        corpus = st.session_state.corpus
        tokens = st.session_state.tokens

        # ---------------------------------------------------------------------
        # RECOMMENDER DESCRIPTION
        # ---------------------------------------------------------------------

        st.subheader("1. Content-Based Recommendation")

        st.write(
            """
            The recommender identifies documents that are similar in content
            to a document selected by the user.

            Each document is represented using TF-IDF features and cosine
            similarity is used to measure document-to-document similarity.
            """
        )

        # ---------------------------------------------------------------------
        # DOCUMENT SELECTION
        # ---------------------------------------------------------------------

        st.subheader("2. Select a document")

        document_options = []

        for i in range(len(corpus)):

            title = str(corpus.iloc[i]["title"]).strip()

            if not title:
                title = f"Document {i + 1}"

            document_options.append(
                f"{i} — {title[:100]}"
            )

        selected_option = st.selectbox(
            "Choose a document to find similar documents",
            document_options,
            index=0
        )

        selected_index = int(
            selected_option.split(" — ", 1)[0]
        )

        selected_doc = corpus.iloc[selected_index]

        # ---------------------------------------------------------------------
        # SELECTED DOCUMENT INFORMATION
        # ---------------------------------------------------------------------

        st.subheader("3. Selected document")

        info1, info2, info3 = st.columns(3)

        info1.metric(
            "Document ID",
            str(selected_doc["doc_id"])
        )

        info2.metric(
            "Domain",
            str(selected_doc["domain"])
        )

        info3.metric(
            "Source",
            str(selected_doc["source_type"])
        )

        st.write(
            f"**Title:** {selected_doc['title']}"
        )

        st.write(
            f"**URL:** {selected_doc['url']}"
        )

        with st.expander("View document preview"):

            st.write(
                str(selected_doc["raw_text"])[:1000]
            )

        # ---------------------------------------------------------------------
        # TOP-K CONTROL
        # ---------------------------------------------------------------------

        st.subheader("4. Recommendation settings")

        top_k = st.slider(
            "Number of recommendations (Top-K)",
            min_value=1,
            max_value=min(20, max(1, len(corpus) - 1)),
            value=min(5, max(1, len(corpus) - 1)),
            step=1
        )

        recommend_button = st.button(
            "Generate Recommendations",
            type="primary",
            use_container_width=True
        )

        # ---------------------------------------------------------------------
        # GENERATE RECOMMENDATIONS
        # ---------------------------------------------------------------------

        if recommend_button:

            with st.spinner(
                "Finding similar documents..."
            ):

                recommendations = recommend_similar_documents(
                    selected_index=selected_index,
                    corpus_records=corpus,
                    token_lists=tokens,
                    top_k=int(top_k)
                )

            st.session_state[
                "recommendations"
            ] = recommendations

            st.session_state[
                "recommendation_source_index"
            ] = selected_index

        # ---------------------------------------------------------------------
        # DISPLAY RECOMMENDATIONS
        # ---------------------------------------------------------------------

        if "recommendations" in st.session_state:

            recommendations = st.session_state[
                "recommendations"
            ]

            st.subheader(
                "5. Top-K Recommendations"
            )

            if recommendations.empty:

                st.info(
                    "No sufficiently similar documents were found."
                )

            else:

                # -------------------------------------------------------------
                # Metrics
                # -------------------------------------------------------------

                m1, m2, m3 = st.columns(3)

                m1.metric(
                    "Recommendations",
                    len(recommendations)
                )

                m2.metric(
                    "Highest similarity",
                    f"{recommendations['similarity'].max():.4f}"
                )

                m3.metric(
                    "Average similarity",
                    f"{recommendations['similarity'].mean():.4f}"
                )

                # -------------------------------------------------------------
                # Recommendation table
                # -------------------------------------------------------------

                display_df = recommendations[
                    [
                        "rank",
                        "title",
                        "similarity",
                        "domain",
                        "source_type",
                        "url"
                    ]
                ].copy()

                display_df.columns = [
                    "Rank",
                    "Recommended Document",
                    "Similarity Score",
                    "Domain",
                    "Source",
                    "URL"
                ]

                st.dataframe(
                    display_df,
                    hide_index=True,
                    use_container_width=True
                )

                # -------------------------------------------------------------
                # Detailed recommendation cards
                # -------------------------------------------------------------

                st.subheader(
                    "Recommendation details"
                )

                for _, rec in recommendations.iterrows():

                    with st.expander(
                        f"#{int(rec['rank'])} — "
                        f"{rec['title']} "
                        f"(Similarity: {rec['similarity']:.4f})"
                    ):

                        st.write(
                            f"**Similarity score:** "
                            f"{rec['similarity']:.4f}"
                        )

                        st.write(
                            f"**Domain:** {rec['domain']}"
                        )

                        st.write(
                            f"**Source:** {rec['source_type']}"
                        )

                        st.write(
                            f"**URL:** {rec['url']}"
                        )

                        st.write(
                            f"**Preview:** {rec['preview']}"
                        )

                # -------------------------------------------------------------
                # Similarity visualization
                # -------------------------------------------------------------

                st.subheader(
                    "Recommendation similarity visualization"
                )

                plot_df = recommendations.copy()

                plot_df = plot_df.sort_values(
                    "similarity",
                    ascending=True
                )

                fig, ax = plt.subplots(
                    figsize=(10, 5)
                )

                ax.barh(
                    plot_df["title"].str.slice(0, 50),
                    plot_df["similarity"]
                )

                ax.set_xlabel(
                    "Cosine Similarity"
                )

                ax.set_ylabel(
                    "Recommended Document"
                )

                ax.set_title(
                    "Top-K Content-Based Recommendations"
                )

                fig.tight_layout()

                st.pyplot(fig)

                # -------------------------------------------------------------
                # Explanation
                # -------------------------------------------------------------

                st.info(
                    """
                    **How the recommendation works**

                    1. The selected document is represented using TF-IDF.
                    2. Every other document is also represented in the same
                       TF-IDF feature space.
                    3. Cosine similarity is calculated between the selected
                       document and every other document.
                    4. The selected document itself is excluded.
                    5. Documents are sorted by similarity score.
                    6. The highest-scoring Top-K documents are displayed.

                    A higher similarity score indicates greater textual
                    similarity between the selected document and the
                    recommendation.
                    """
                )

# =============================================================================
# TAB 7 — EVALUATION
# =============================================================================
with tab7:

    st.header("7. Retrieval Evaluation")

    st.write(
        "Evaluate the effectiveness of the information retrieval system using "
        "standard Information Retrieval metrics. Relevance judgments are "
        "provided manually as ground truth for the selected query."
    )

    # -------------------------------------------------------------------------
    # CHECK CORPUS / PREPROCESSING
    # -------------------------------------------------------------------------
    if st.session_state.corpus is None or st.session_state.tokens is None:

        st.warning(
            "Please acquire the corpus and run preprocessing in Sections 1 and 2 "
            "before performing retrieval evaluation."
        )

    else:

        corpus = st.session_state.corpus
        tokens = st.session_state.tokens

        if len(corpus) == 0:

            st.warning("The corpus is empty. Please acquire documents first.")

        else:

            # =================================================================
            # 1. EVALUATION QUERY
            # =================================================================

            st.subheader("1. Evaluation query")

            eval_query = st.text_input(
                "Enter a query to evaluate",
                value=st.session_state.get(
                    "evaluation_query_used",
                    st.session_state.get(
                        "search_query",
                        "information retrieval"
                    )
                ),
                key="evaluation_query_input",
                help=(
                    "Use a meaningful query related to the documents in your "
                    "collection."
                )
            )

            c1, c2 = st.columns(2)

            with c1:
                eval_k = st.number_input(
                    "Evaluation K",
                    min_value=1,
                    max_value=min(50, len(corpus)),
                    value=min(10, len(corpus)),
                    step=1,
                    key="evaluation_k_input",
                    help=(
                        "K is used for Precision@K, Recall@K and NDCG@K."
                    )
                )

            with c2:
                query_expansion_eval = st.checkbox(
                    "Use query expansion",
                    value=True,
                    key="evaluation_query_expansion",
                    help=(
                        "Use the same query expansion strategy used by "
                        "the retrieval system in Section 5."
                    )
                )

            # =================================================================
            # 2. BUILD GROUND-TRUTH DOCUMENT LIST
            # =================================================================

            st.subheader("2. Ground-truth relevance judgement")

            st.write(
                "Select every document that you consider relevant to the "
                "evaluation query. These selections form the ground-truth "
                "relevance set used by Precision, Recall, F1, MAP, MRR "
                "and NDCG."
            )

            # -----------------------------------------------------------------
            # Create a complete document list.
            #
            # IMPORTANT:
            # Unlike the previous version, relevant documents are NOT limited
            # to the documents returned by the search results.
            # -----------------------------------------------------------------

            doc_labels = []
            label_to_index = {}

            for idx, row in corpus.iterrows():

                title = str(row.get("title", "Untitled document"))

                doc_id = str(
                    row.get(
                        "doc_id",
                        f"doc-{idx}"
                    )
                )

                label = (
                    f"{idx} | "
                    f"{title[:100]} | "
                    f"{doc_id}"
                )

                doc_labels.append(label)
                label_to_index[label] = int(idx)

            # Keep previously selected documents when possible.
            previous_relevant = st.session_state.get(
                "evaluation_relevant_indices",
                []
            )

            previous_labels = [
                label
                for label, idx in label_to_index.items()
                if idx in previous_relevant
            ]

            relevant_labels = st.multiselect(
                "Ground-truth relevant documents",
                options=doc_labels,
                default=previous_labels,
                key="evaluation_relevant_docs",
                help=(
                    "Select all documents that should be considered relevant "
                    "to the evaluation query."
                )
            )

            relevant_indices = [
                label_to_index[label]
                for label in relevant_labels
            ]

            relevant_indices = sorted(
                set(relevant_indices)
            )

            # -----------------------------------------------------------------
            # Ground-truth summary
            # -----------------------------------------------------------------

            q1, q2 = st.columns(2)

            with q1:
                st.metric(
                    "Corpus documents",
                    len(corpus)
                )

            with q2:
                st.metric(
                    "Ground-truth relevant documents",
                    len(relevant_indices)
                )

            if relevant_indices:

                st.success(
                    f"{len(relevant_indices)} relevant document(s) selected "
                    "as ground truth."
                )

            else:

                st.info(
                    "Select at least one relevant document before running "
                    "the evaluation."
                )

            # =================================================================
            # 3. RUN EVALUATION
            # =================================================================

            run_evaluation = st.button(
                "Run Evaluation",
                type="primary",
                use_container_width=True,
                key="run_section_f_evaluation"
            )

            if run_evaluation:

                if not eval_query.strip():

                    st.warning(
                        "Please enter an evaluation query."
                    )

                elif not relevant_indices:

                    st.warning(
                        "Please select at least one relevant document "
                        "as ground truth."
                    )

                else:

                    with st.spinner(
                        "Running retrieval and calculating IR metrics..."
                    ):

                        # -----------------------------------------------------
                        # Retrieve enough documents for evaluation.
                        #
                        # We use up to 50 rather than the small UI search K so
                        # Recall is not unnecessarily restricted.
                        # -----------------------------------------------------

                        retrieval_k = min(
                            50,
                            len(corpus)
                        )

                        baseline = search_documents(
                            query=eval_query,
                            corpus_records=corpus,
                            token_lists=tokens,
                            top_k=retrieval_k,
                            use_query_expansion=query_expansion_eval
                        )

                        # -----------------------------------------------------
                        # Make sure doc_index exists.
                        # -----------------------------------------------------

                        if (
                            baseline is not None
                            and not baseline.empty
                            and "doc_index" in baseline.columns
                        ):

                            baseline_ranked = (
                                baseline["doc_index"]
                                .astype(int)
                                .tolist()
                            )

                        else:

                            baseline_ranked = []

                        # -----------------------------------------------------
                        # Local metric calculation.
                        #
                        # This is intentionally self-contained so Section F
                        # does not depend on another UI block.
                        # -----------------------------------------------------

                        def calculate_section_f_metrics(
                            ranked_indices,
                            relevant_indices,
                            k
                        ):

                            relevant_set = set(
                                int(x)
                                for x in relevant_indices
                            )

                            ranked = [
                                int(x)
                                for x in ranked_indices
                            ]

                            # Remove accidental duplicate document indices
                            # while preserving ranking order.
                            ranked_unique = list(
                                dict.fromkeys(ranked)
                            )

                            hits = [
                                1 if idx in relevant_set else 0
                                for idx in ranked_unique
                            ]

                            retrieved_relevant = sum(hits)

                            # -------------------------------------------------
                            # Precision
                            # -------------------------------------------------

                            precision = (
                                retrieved_relevant / len(ranked_unique)
                                if ranked_unique
                                else 0.0
                            )

                            # -------------------------------------------------
                            # Recall
                            # -------------------------------------------------

                            recall = (
                                retrieved_relevant / len(relevant_set)
                                if relevant_set
                                else 0.0
                            )

                            # -------------------------------------------------
                            # F1
                            # -------------------------------------------------

                            if precision + recall > 0:

                                f1 = (
                                    2
                                    * precision
                                    * recall
                                    / (precision + recall)
                                )

                            else:

                                f1 = 0.0

                            # -------------------------------------------------
                            # Precision@K
                            # -------------------------------------------------

                            top_hits = hits[:k]

                            p_at_k = (
                                sum(top_hits) / k
                                if k > 0
                                else 0.0
                            )

                            # -------------------------------------------------
                            # Recall@K
                            # -------------------------------------------------

                            r_at_k = (
                                sum(top_hits) / len(relevant_set)
                                if relevant_set
                                else 0.0
                            )

                            # -------------------------------------------------
                            # Average Precision
                            # -------------------------------------------------

                            running_hits = 0
                            precision_sum = 0.0

                            for rank, hit in enumerate(
                                hits,
                                start=1
                            ):

                                if hit:

                                    running_hits += 1

                                    precision_sum += (
                                        running_hits / rank
                                    )

                            ap = (
                                precision_sum / len(relevant_set)
                                if relevant_set
                                else 0.0
                            )

                            # -------------------------------------------------
                            # Reciprocal Rank
                            # -------------------------------------------------

                            rr = 0.0

                            for rank, hit in enumerate(
                                hits,
                                start=1
                            ):

                                if hit:

                                    rr = 1.0 / rank
                                    break

                            # -------------------------------------------------
                            # Binary NDCG@K
                            # -------------------------------------------------

                            dcg = 0.0

                            for rank, hit in enumerate(
                                top_hits,
                                start=1
                            ):

                                if hit:

                                    dcg += (
                                        1.0
                                        / math.log2(rank + 1)
                                    )

                            ideal_hits = min(
                                len(relevant_set),
                                k
                            )

                            idcg = sum(
                                1.0 / math.log2(rank + 1)
                                for rank in range(
                                    1,
                                    ideal_hits + 1
                                )
                            )

                            ndcg = (
                                dcg / idcg
                                if idcg > 0
                                else 0.0
                            )

                            return {
                                "Precision": round(
                                    precision,
                                    4
                                ),
                                "Recall": round(
                                    recall,
                                    4
                                ),
                                "F1-score": round(
                                    f1,
                                    4
                                ),
                                "Precision@K": round(
                                    p_at_k,
                                    4
                                ),
                                "Recall@K": round(
                                    r_at_k,
                                    4
                                ),
                                "MAP": round(
                                    ap,
                                    4
                                ),
                                "MRR": round(
                                    rr,
                                    4
                                ),
                                "NDCG@K": round(
                                    ndcg,
                                    4
                                )
                            }

                        # -----------------------------------------------------
                        # Calculate baseline metrics
                        # -----------------------------------------------------

                        baseline_metrics = (
                            calculate_section_f_metrics(
                                baseline_ranked,
                                relevant_indices,
                                int(eval_k)
                            )
                        )

                        comparison_rows = [
                            {
                                "Method": "TF-IDF retrieval",
                                **baseline_metrics
                            }
                        ]

                        # =====================================================
                        # OPTIONAL TF-IDF + PAGERANK COMPARISON
                        # =====================================================

                        combined = pd.DataFrame()

                        pagerank_available = (
                            "pagerank"
                            in st.session_state
                            and
                            isinstance(
                                st.session_state["pagerank"],
                                pd.DataFrame
                            )
                            and
                            not st.session_state["pagerank"].empty
                        )

                        if pagerank_available:

                            pr_df = (
                                st.session_state["pagerank"]
                                .copy()
                            )

                            # -------------------------------------------------
                            # Identify the PageRank document-index column.
                            # -------------------------------------------------

                            pr_index_col = None

                            for candidate in [
                                "doc_index",
                                "index"
                            ]:

                                if candidate in pr_df.columns:

                                    pr_index_col = candidate
                                    break

                            # -------------------------------------------------
                            # Identify PageRank score column.
                            # -------------------------------------------------

                            pr_score_col = None

                            for candidate in [
                                "pagerank",
                                "page_rank",
                                "PageRank"
                            ]:

                                if candidate in pr_df.columns:

                                    pr_score_col = candidate
                                    break

                            if (
                                pr_index_col is not None
                                and pr_score_col is not None
                                and not baseline.empty
                            ):

                                # -------------------------------------------------
                                # Create a clean PageRank lookup.
                                # -------------------------------------------------

                                pr_lookup = (
                                    pr_df[
                                        [
                                            pr_index_col,
                                            pr_score_col
                                        ]
                                    ]
                                    .copy()
                                )

                                pr_lookup[pr_index_col] = (
                                    pd.to_numeric(
                                        pr_lookup[pr_index_col],
                                        errors="coerce"
                                    )
                                )

                                pr_lookup[pr_score_col] = (
                                    pd.to_numeric(
                                        pr_lookup[pr_score_col],
                                        errors="coerce"
                                    )
                                    .fillna(0.0)
                                )

                                pr_lookup = (
                                    pr_lookup
                                    .dropna(
                                        subset=[pr_index_col]
                                    )
                                )

                                # -------------------------------------------------
                                # Merge TF-IDF results with PageRank.
                                # -------------------------------------------------

                                combined = baseline.copy()

                                combined = combined.merge(
                                    pr_lookup,
                                    left_on="doc_index",
                                    right_on=pr_index_col,
                                    how="left"
                                )

                                combined["pagerank_score"] = (
                                    combined[pr_score_col]
                                    .fillna(0.0)
                                )

                                # -------------------------------------------------
                                # Normalise TF-IDF scores.
                                # -------------------------------------------------

                                tfidf_min = (
                                    combined["score"].min()
                                )

                                tfidf_max = (
                                    combined["score"].max()
                                )

                                if tfidf_max > tfidf_min:

                                    combined["tfidf_norm"] = (
                                        (
                                            combined["score"]
                                            - tfidf_min
                                        )
                                        /
                                        (
                                            tfidf_max
                                            - tfidf_min
                                        )
                                    )

                                else:

                                    combined["tfidf_norm"] = 0.0

                                # -------------------------------------------------
                                # Normalise PageRank.
                                # -------------------------------------------------

                                pr_min = (
                                    combined["pagerank_score"]
                                    .min()
                                )

                                pr_max = (
                                    combined["pagerank_score"]
                                    .max()
                                )

                                if pr_max > pr_min:

                                    combined["pagerank_norm"] = (
                                        (
                                            combined[
                                                "pagerank_score"
                                            ]
                                            - pr_min
                                        )
                                        /
                                        (
                                            pr_max
                                            - pr_min
                                        )
                                    )

                                else:

                                    combined["pagerank_norm"] = 0.0

                                # -------------------------------------------------
                                # Combine relevance and PageRank.
                                #
                                # 75% TF-IDF + 25% PageRank
                                # -------------------------------------------------

                                combined["combined_score"] = (
                                    0.75
                                    * combined["tfidf_norm"]
                                    +
                                    0.25
                                    * combined["pagerank_norm"]
                                )

                                combined = (
                                    combined
                                    .sort_values(
                                        "combined_score",
                                        ascending=False
                                    )
                                    .reset_index(
                                        drop=True
                                    )
                                )

                                combined["rank"] = (
                                    np.arange(
                                        1,
                                        len(combined) + 1
                                    )
                                )

                                combined_ranked = (
                                    combined["doc_index"]
                                    .astype(int)
                                    .tolist()
                                )

                                combined_metrics = (
                                    calculate_section_f_metrics(
                                        combined_ranked,
                                        relevant_indices,
                                        int(eval_k)
                                    )
                                )

                                comparison_rows.append(
                                    {
                                        "Method":
                                            "TF-IDF + PageRank",
                                        **combined_metrics
                                    }
                                )

                        # =====================================================
                        # STORE RESULTS
                        # =====================================================

                        st.session_state[
                            "evaluation_results"
                        ] = baseline

                        st.session_state[
                            "evaluation_combined"
                        ] = combined

                        st.session_state[
                            "evaluation_metrics"
                        ] = pd.DataFrame(
                            comparison_rows
                        )

                        st.session_state[
                            "evaluation_relevant_indices"
                        ] = relevant_indices

                        st.session_state[
                            "evaluation_query_used"
                        ] = eval_query

                        st.session_state[
                            "evaluation_k_used"
                        ] = int(eval_k)

                        st.session_state[
                            "evaluation_query_expansion_used"
                        ] = query_expansion_eval

                    st.success(
                        "Retrieval evaluation completed successfully."
                    )

            # =================================================================
            # 4. DISPLAY RESULTS
            # =================================================================

            if "evaluation_metrics" in st.session_state:

                st.divider()

                st.subheader(
                    "3. Retrieval effectiveness"
                )

                metrics_df = (
                    st.session_state[
                        "evaluation_metrics"
                    ]
                )

                st.dataframe(
                    metrics_df,
                    hide_index=True,
                    use_container_width=True
                )

                # -------------------------------------------------------------
                # Metric cards
                # -------------------------------------------------------------

                baseline_row = metrics_df.iloc[0]

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Precision",
                    f"{baseline_row['Precision']:.4f}"
                )

                c2.metric(
                    "Recall",
                    f"{baseline_row['Recall']:.4f}"
                )

                c3.metric(
                    "F1-score",
                    f"{baseline_row['F1-score']:.4f}"
                )

                c4.metric(
                    "MAP",
                    f"{baseline_row['MAP']:.4f}"
                )

                c5, c6, c7, c8 = st.columns(4)

                c5.metric(
                    "Precision@K",
                    f"{baseline_row['Precision@K']:.4f}"
                )

                c6.metric(
                    "Recall@K",
                    f"{baseline_row['Recall@K']:.4f}"
                )

                c7.metric(
                    "MRR",
                    f"{baseline_row['MRR']:.4f}"
                )

                c8.metric(
                    "NDCG@K",
                    f"{baseline_row['NDCG@K']:.4f}"
                )

                # =================================================================
                # 5. COMPARATIVE ANALYSIS
                # =================================================================

                st.subheader(
                    "4. Comparative analysis"
                )

                if len(metrics_df) > 1:

                    st.write(
                        "The table compares the baseline TF-IDF retrieval "
                        "with the TF-IDF + PageRank ranking. Higher values "
                        "indicate better retrieval effectiveness."
                    )

                else:

                    st.info(
                        "PageRank results are not available for this "
                        "evaluation. Run **Calculate PageRank** in "
                        "Section 5 and then rerun Section 7."
                    )

                # -------------------------------------------------------------
                # Comparison chart
                # -------------------------------------------------------------

                metric_names = [
                    "Precision",
                    "Recall",
                    "F1-score",
                    "Precision@K",
                    "Recall@K",
                    "MAP",
                    "MRR",
                    "NDCG@K"
                ]

                plot_df = (
                    metrics_df
                    .set_index("Method")
                    [metric_names]
                )

                fig, ax = plt.subplots(
                    figsize=(12, 5)
                )

                plot_df.T.plot(
                    kind="bar",
                    ax=ax
                )

                ax.set_ylim(
                    0,
                    1.05
                )

                ax.set_ylabel(
                    "Score"
                )

                ax.set_xlabel(
                    "IR metric"
                )

                ax.set_title(
                    "Retrieval effectiveness comparison"
                )

                ax.tick_params(
                    axis="x",
                    rotation=35
                )

                fig.tight_layout()

                st.pyplot(fig)

                plt.close(fig)

                # =================================================================
                # 6. RANKED RESULTS USED FOR EVALUATION
                # =================================================================

                st.subheader(
                    "5. Ranked results used for evaluation"
                )

                eval_results = (
                    st.session_state[
                        "evaluation_results"
                    ]
                )

                relevant_set = set(
                    st.session_state[
                        "evaluation_relevant_indices"
                    ]
                )

                if (
                    eval_results is not None
                    and not eval_results.empty
                ):

                    shown = (
                        eval_results
                        .copy()
                    )

                    shown["relevant"] = (
                        shown["doc_index"]
                        .astype(int)
                        .isin(relevant_set)
                    )

                    shown["judgment"] = np.where(
                        shown["relevant"],
                        "Relevant",
                        "Not relevant"
                    )

                    columns_to_show = [
                        "rank",
                        "doc_index",
                        "title",
                        "score",
                        "judgment",
                        "domain",
                        "url"
                    ]

                    available_columns = [
                        col
                        for col in columns_to_show
                        if col in shown.columns
                    ]

                    st.dataframe(
                        shown[
                            available_columns
                        ],
                        hide_index=True,
                        use_container_width=True
                    )

                # =================================================================
                # 7. RELEVANCE BY RANK
                # =================================================================

                if (
                    eval_results is not None
                    and not eval_results.empty
                ):

                    st.subheader(
                        "6. Relevance distribution across ranks"
                    )

                    rank_analysis = (
                        eval_results[
                            [
                                "rank",
                                "doc_index"
                            ]
                        ]
                        .copy()
                    )

                    rank_analysis["Relevant"] = (
                        rank_analysis["doc_index"]
                        .astype(int)
                        .isin(relevant_set)
                    )

                    fig, ax = plt.subplots(
                        figsize=(10, 4)
                    )

                    ax.plot(
                        rank_analysis["rank"],
                        rank_analysis["Relevant"].astype(int),
                        marker="o"
                    )

                    ax.set_xlabel(
                        "Rank position"
                    )

                    ax.set_ylabel(
                        "Relevance"
                    )

                    ax.set_yticks(
                        [0, 1]
                    )

                    ax.set_yticklabels(
                        [
                            "Not relevant",
                            "Relevant"
                        ]
                    )

                    ax.set_title(
                        "Relevant Documents Across Ranking Positions"
                    )

                    ax.grid(
                        True,
                        alpha=0.3
                    )

                    fig.tight_layout()

                    st.pyplot(fig)

                    plt.close(fig)

                # =================================================================
                # 8. AUTOMATIC INTERPRETATION
                # =================================================================

                st.subheader(
                    "7. Evaluation analysis"
                )

                precision = float(
                    baseline_row["Precision"]
                )

                recall = float(
                    baseline_row["Recall"]
                )

                f1 = float(
                    baseline_row["F1-score"]
                )

                p_at_k = float(
                    baseline_row["Precision@K"]
                )

                r_at_k = float(
                    baseline_row["Recall@K"]
                )

                map_score = float(
                    baseline_row["MAP"]
                )

                mrr_score = float(
                    baseline_row["MRR"]
                )

                ndcg_score = float(
                    baseline_row["NDCG@K"]
                )

                analysis_points = []

                # -------------------------------------------------------------
                # Precision
                # -------------------------------------------------------------

                if precision >= 0.8:

                    analysis_points.append(
                        "Precision is high, indicating that most retrieved "
                        "documents are relevant."
                    )

                elif precision >= 0.5:

                    analysis_points.append(
                        "Precision is moderate, indicating that the "
                        "retrieval system returns a reasonable proportion "
                        "of relevant documents."
                    )

                else:

                    analysis_points.append(
                        "Precision is relatively low, indicating that "
                        "several retrieved documents are not relevant "
                        "to the query."
                    )

                # -------------------------------------------------------------
                # Recall
                # -------------------------------------------------------------

                if recall >= 0.8:

                    analysis_points.append(
                        "Recall is high, indicating that the system retrieves "
                        "most of the known relevant documents."
                    )

                elif recall >= 0.5:

                    analysis_points.append(
                        "Recall is moderate, so some relevant documents "
                        "may still be missing from the retrieved results."
                    )

                else:

                    analysis_points.append(
                        "Recall is low, suggesting that the system misses "
                        "a substantial portion of the known relevant documents."
                    )

                # -------------------------------------------------------------
                # F1
                # -------------------------------------------------------------

                analysis_points.append(
                    f"The F1-score of {f1:.4f} summarizes the balance "
                    "between precision and recall."
                )

                # -------------------------------------------------------------
                # P@K
                # -------------------------------------------------------------

                analysis_points.append(
                    f"Precision@K is {p_at_k:.4f}, showing the quality "
                    f"of the top-{int(eval_k)} retrieved results."
                )

                # -------------------------------------------------------------
                # R@K
                # -------------------------------------------------------------

                analysis_points.append(
                    f"Recall@K is {r_at_k:.4f}, showing how much of the "
                    f"known relevant set appears within the top-{int(eval_k)} "
                    "results."
                )

                # -------------------------------------------------------------
                # MAP
                # -------------------------------------------------------------

                analysis_points.append(
                    f"MAP is {map_score:.4f}. Higher MAP indicates that "
                    "relevant documents tend to appear earlier in the ranking."
                )

                # -------------------------------------------------------------
                # MRR
                # -------------------------------------------------------------

                analysis_points.append(
                    f"MRR is {mrr_score:.4f}. A higher value means that "
                    "the first relevant document appears closer to the top."
                )

                # -------------------------------------------------------------
                # NDCG
                # -------------------------------------------------------------

                analysis_points.append(
                    f"NDCG@K is {ndcg_score:.4f}, reflecting the quality "
                    "of the ranking while giving greater importance to "
                    "higher-ranked relevant documents."
                )

                for point in analysis_points:

                    st.write(
                        "• " + point
                    )

                # =================================================================
                # 9. FINAL INFERENCE
                # =================================================================

                st.subheader(
                    "8. Evaluation inference"
                )

                if (
                    precision >= 0.7
                    and recall >= 0.7
                    and ndcg_score >= 0.7
                ):

                    st.success(
                        "Overall, the retrieval system demonstrates strong "
                        "effectiveness. The retrieved results contain a high "
                        "proportion of relevant documents, cover a substantial "
                        "portion of the known relevant set, and maintain good "
                        "ranking quality."
                    )

                elif (
                    precision >= 0.7
                    and recall < 0.7
                ):

                    st.info(
                        "The system shows good precision but comparatively "
                        "lower recall. This indicates that the ranking is "
                        "selective: the retrieved documents are generally "
                        "relevant, but some relevant documents are not "
                        "being retrieved."
                    )

                elif (
                    precision < 0.7
                    and recall >= 0.7
                ):

                    st.info(
                        "The system achieves relatively good recall but "
                        "lower precision. This means the system retrieves "
                        "many relevant documents but also returns irrelevant "
                        "documents."
                    )

                else:

                    st.info(
                        "The evaluation indicates room for improvement in "
                        "retrieval effectiveness and ranking quality. "
                        "Preprocessing, query expansion, TF-IDF weighting "
                        "and PageRank integration can potentially improve "
                        "the results."
                    )

                # =================================================================
                # 10. METRIC DEFINITIONS
                # =================================================================

                with st.expander(
                    "Metric definitions"
                ):

                    st.markdown(
                        """
                        **Precision**  
                        Fraction of retrieved documents that are relevant.

                        **Recall**  
                        Fraction of the known relevant documents that were retrieved.

                        **F1-score**  
                        Harmonic mean of Precision and Recall.

                        **Precision@K**  
                        Precision considering only the top-K ranked documents.

                        **Recall@K**  
                        Recall considering only the top-K ranked documents.

                        **MAP (Mean Average Precision)**  
                        For this single-query evaluation, MAP is equivalent to
                        Average Precision. It rewards relevant documents appearing
                        earlier in the ranking.

                        **MRR (Mean Reciprocal Rank)**  
                        For this single-query evaluation, MRR is equivalent to
                        Reciprocal Rank and measures how early the first relevant
                        document appears.

                        **NDCG@K**  
                        Measures ranking quality while giving greater importance
                        to relevant documents appearing near the top of the ranking.
                        """
                    )