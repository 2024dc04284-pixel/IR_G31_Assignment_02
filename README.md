# Heterogeneous Collection & Text Mining

Group 31 Information Retrieval assignment — acquire a document collection from
**three heterogeneous sources** (web crawling, a public dataset, and a public
API), then run it through a text mining pipeline: preprocessing, feature
engineering, statistical analysis, keyword extraction, and classification.

## Install and run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. Needs outbound internet for the crawler,
the Wikipedia API, and the "fetch dataset from URL" option — the bundled
sample dataset and file upload both work fully offline.

Every acquisition method ships with working defaults, so each tab can be run
by just pressing its button — nothing has to be typed in first:
- Web crawl defaults to five seed URLs across two domains.
- Public dataset defaults to a bundled `sample_dataset.csv` (10 short
  documents), so it needs no upload and no network access.
- The Wikipedia API defaults to the query "Information retrieval" with 5
  results, fetched in a single HTTP request (see below).

## Walkthrough

1. **Acquire & store** — pick one of three acquisition methods, run it, then
   repeat with another to build a heterogeneous corpus. **Load store into
   working corpus** happens automatically after each run.
   - *Web crawl* — paste a few seed URLs (one per line), set crawl depth and
     page limit, run the crawl. Breadth-first frontier over all seeds.
   - *Public dataset* — defaults to the bundled sample dataset (just click
     **Ingest bundled sample dataset**); or switch to uploading a local
     CSV/JSON file, or fetching one live from a URL. Point at the text (and
     optionally title) column; each row becomes a document.
   - *Public API (Wikipedia)* — search Wikipedia's MediaWiki JSON API by
     query term and ingest the plaintext extract of each hit, all in one
     request (`generator=search` combined with `prop=extracts`, so the
     request count doesn't grow with the number of results asked for). This
     is a structured API fetch (JSON), not HTML scraping.
2. **Preprocess** — configure stopword removal / stemming, preprocess the
   corpus, inspect one document stage by stage, run the strategy comparison.
3. **Features & keywords** — build TF / TF-IDF matrices, compare vectoriser
   strategies, extract per-document and corpus-wide keywords.
4. **Analytics & classification** — document profiles, corpus statistics
   (including a documents-per-source breakdown), Zipf's law plot, length
   distribution, and a classifier that predicts a document's domain
   (web domain, `dataset:<name>`, or `api:wikipedia`) from its text.

## What's implemented

| Requirement | Where |
|---|---|
| Heterogeneous acquisition — web crawling | `crawl()` — breadth-first frontier over all seed URLs |
| Heterogeneous acquisition — public dataset | `load_tabular()` / `dataset_rows_to_items()` — file upload or fetch-by-URL, CSV/JSON |
| Heterogeneous acquisition — public API | `fetch_wikipedia()` — Wikipedia MediaWiki JSON API (search + extract in one request), not HTML scraping |
| Default corpus for every source with no user input | `DEFAULT_SEEDS` (crawl), `sample_dataset.csv` (dataset), preset query (API) |
| Configurable crawl depth | depth slider on the Web crawl method |
| Duplicate URL handling | canonical-URL `seen` set, persisted in the `urls` table — applied to all three sources |
| Duplicate document handling | SHA-256 content hash, checked before storing — catches duplicates *within and across* sources |
| Source provenance | `source_type` / `source_detail` columns on `metadata`, so every document is traceable to web / dataset / api |
| Metadata stored separately from content | SQLite `documents` table (text) vs. `metadata` table (url, domain, title, source, timestamps) |
| Preprocessing | `clean_text`, `tokenize`, stopword removal, suffix-stripping stemmer |
| Feature engineering | TF and TF-IDF via scikit-learn `CountVectorizer` / `TfidfVectorizer` |
| Comparative analysis — preprocessing | strategy comparison table (Preprocess tab) |
| Comparative analysis — features | vectoriser comparison table (Features tab) |
| Keyword extraction | top TF-IDF terms per document and corpus-wide |
| Document profiling | per-document length, vocabulary, source, top terms table |
| Document classification | logistic regression predicting domain from TF-IDF features |
| Visualisations | source-type bar chart, domain bar chart, length histogram, Zipf log-log plot, keyword bar chart, confusion matrix |

## Notes

- The crawler respects `robots.txt` and applies a per-host delay between
  requests. The dataset and API paths don't crawl or scrape HTML at all, so
  robots.txt doesn't apply to them.
- Crawling a large news site can return thin text if the article body is
  rendered by JavaScript — this crawler reads static HTML only.
- All acquired data — regardless of source — lives in `crawl_store.db`
  (SQLite, created on first run). Use **Reset store** on the Acquire & store
  tab to start over. Opening a store created before heterogeneous sources
  existed still works: the schema is migrated in place on first open.
- Mixing sources is the easiest way to satisfy the classifier's "2+ classes
  with 4+ documents" requirement — e.g. crawl one site, ingest one dataset,
  and pull a couple of Wikipedia queries.
- The Wikipedia fetch is capped by the "Max articles" input (default 5,
  max 50) and always issues exactly one HTTP request per click, so raising
  it doesn't multiply the number of requests sent.
