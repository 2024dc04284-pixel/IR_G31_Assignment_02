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
            "source_type": "web", "source_detail": "",
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
        for t in ("documents", "metadata", "urls"):
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

tab1, tab2, tab3, tab4 = st.tabs(
    ["1 · Acquire & store", "2 · Preprocess", "3 · Features & keywords", "4 · Analytics & classification"])

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
