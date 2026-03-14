# PDF Chatbot

A Flask-based chatbot that lets you upload PDF or DOCX files and ask questions about their content using OpenAI embeddings and LlamaIndex.

## Project Structure

```
pdf-chat/
├── app.py                    # Main Flask application
├── benchmark_embeddings.py   # Embedding strategy benchmark script
├── requirements.txt          # Python dependencies
├── .env                      # API keys (not committed to git)
├── .gitignore
├── templates/
│   └── index.html            # Chat UI
├── uploads/                  # Uploaded files (auto-created)
└── storage/                  # Persisted vector index (auto-created)
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-your-key-here
FLASK_SECRET_KEY=any-random-string
```

`FLASK_SECRET_KEY` is optional and defaults to `dev-secret-key` if not set.

### 3. Run the app

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## Application Flow

### Upload Flow

```
User selects PDF/DOCX
        |
        v
POST /upload (multipart form)
        |
        v
File saved to ./uploads/                      ~2ms
        |
        v
PDFReader / DocxReader parses into chunks      ~90ms
        |
        v
VectorStoreIndex batch-embeds all chunks       ~700ms  (batch_size=50)
        |
        v
Index persisted to ./storage/                  ~40ms
        |
        v
Index + engine cached in memory                ~35ms
        |
        v
Response: { status, timings_ms: { ... } }
```

### Chat Flow (Optimized)

```
User types a question
        |
        v
POST /ask  { message: "..." }
        |
        v
Load engine from memory cache                 ~0.1ms
        |
        v
Build chat history from session                ~0.0ms
        |
        v
RAG: embed query + vector search + LLM        ~1300ms  (OpenAI API)
        |
        v
Response: { answer, history, timings_ms }
```

### Session & Chat History

- Chat history is stored in a server-side Flask session (filesystem-backed).
- Each message pair `{ user, assistant }` is appended after every successful exchange.
- History is passed to the chat engine so follow-up questions have conversational context.
- Uploading a new file resets the chat history and invalidates the in-memory cache.

## Embedding Benchmark

The `benchmark_embeddings.py` script tests all architectural combinations through the actual Flask API endpoints, measures every component in **milliseconds**, and compares optimized vs unoptimized flows.

### Run the benchmark

```bash
python benchmark_embeddings.py
```

Requires a PDF in `./uploads/` (upload one through the app first).

### Architectures tested

**Upload (3 configs):**

| # | Architecture | Batch Size |
|---|---|---|
| A1 | Sequential | 1 |
| A2 | Batch | 20 |
| A3 | Batch large | 50 |

**Query - Unoptimized (reload from disk every request):**

| # | Architecture | Description |
|---|---|---|
| B1 | No history | Reload index + recreate engine per query |
| B2 | With history | Same, with accumulating chat history |

**Query - Optimized (in-memory cache + engine reuse):**

| # | Architecture | Description |
|---|---|---|
| C1 | No history | Cached engine, fresh session per query |
| C2 | With history | Cached engine, accumulating history |
| C3 | Repeated query | Same question 3x (fully warm) |

**Query - No embeddings:**

| # | Architecture | Description |
|---|---|---|
| D1 | Direct LLM | No RAG, no vector search, just LLM |

### Benchmark Results (milliseconds)

Measured with `atlassian-git-cheatsheet.pdf` (102 KB, 2 chunks) on `text-embedding-ada-002`:

**Upload latency breakdown:**

| Architecture | Total | Embed | Parse | Persist | Cache warmup |
|---|---|---|---|---|---|
| A1. Sequential (batch=1) | 1771ms | 1457ms | 236ms | 38ms | 37ms |
| A2. Batch (batch=20) | 970ms | 795ms | 82ms | 51ms | 41ms |
| A3. Batch (batch=50) | **871ms** | **703ms** | 92ms | 40ms | 34ms |

**Query latency - component breakdown:**

| Architecture | Engine load | History build | RAG+LLM | Total |
|---|---|---|---|---|
| B1. Unoptimized, no history | 24.8ms | - | 1524ms | 1550ms |
| B2. Unoptimized, with history | 17.6ms | - | 1220ms | 1239ms |
| C1. **Optimized, no history** | **0.1ms** | **0.0ms** | 1317ms | **1317ms** |
| C2. Optimized, with history | 0.2ms | 0.1ms | 2465ms | 2465ms |
| D1. Direct LLM (no RAG) | - | - | 1257ms | 1257ms |

**Optimization impact:**

| Metric | Before | After | Improvement |
|---|---|---|---|
| Engine load per query | 25ms | **0.1ms** | **250x faster** |
| History build | ~1ms | **0.0ms** | eliminated |
| Total overhead (non-LLM) | ~26ms | **0.1ms** | **260x faster** |

> The RAG+LLM call (OpenAI API) is the irreducible floor at ~1200-2800ms. All controllable overhead has been reduced to sub-millisecond.

### Where the time goes (optimized query)

```
Engine load (from cache)   :    0.1ms   (0%)
History build              :    0.0ms   (0%)
RAG + LLM response         : 1317.0ms  (100%)
-----------------------------------------
Total                      : 1317.1ms  (100%)
```

### Key Takeaways

- **Engine overhead reduced from 25ms to 0.1ms** (250x) by caching the index and chat engine in memory.
- **Batch embedding** cuts upload time from 1771ms to 871ms (2x faster) by reducing API round-trips.
- **All controllable latency is now sub-millisecond.** The remaining ~1300ms is entirely the OpenAI API call (embedding the query + LLM response generation), which is the irreducible floor.
- **Direct LLM vs RAG:** Direct LLM is ~1257ms but has no document context. RAG adds ~60ms of retrieval overhead for document-grounded answers.

## Optimizations Applied to the App

| Optimization | Where | Impact |
|---|---|---|
| **In-memory index cache** | `get_cached_index()` | Index loaded from disk once, reused from memory (25ms -> 0.1ms) |
| **In-memory engine reuse** | `get_cached_engine()` | Chat engine created once, reused across queries |
| **Batch embedding** | `embed_batch_size=50` | 2x faster upload (1771ms -> 871ms) |
| **Embedding disk cache** | `./embedding_cache/` + memory dict | Avoids re-embedding duplicate text chunks |
| **Index persistence** | `storage_context.persist()` | Documents embedded once; queries never re-index |
| **Component timing** | `timings_ms` in API responses | Every request returns ms-level breakdown |
| **Environment variables** | `.env` + `python-dotenv` | API keys not hardcoded in source |
| **Cache warmup on upload** | Pre-loads index + engine after indexing | First query after upload has zero cold-start |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serves the chat UI |
| POST | `/upload` | Upload a PDF or DOCX file for indexing |
| POST | `/ask` | Optimized query (cached engine) |
| POST | `/ask_no_cache` | Unoptimized query (reloads from disk, for benchmarking) |

### POST /upload

**Request:** `multipart/form-data` with a `file` field.

**Response:**
```json
{
  "status": "success",
  "timings_ms": {
    "file_save_ms": 2.7,
    "parse_ms": 91.8,
    "embed_index_ms": 702.9,
    "persist_ms": 40.0,
    "cache_warmup_ms": 33.8,
    "total_ms": 871.3
  }
}
```

### POST /ask

**Request:**
```json
{ "message": "What is the document about?" }
```

**Response:**
```json
{
  "answer": "The document covers...",
  "history": [
    { "user": "What is the document about?", "assistant": "The document covers..." }
  ],
  "timings_ms": {
    "engine_load_ms": 0.1,
    "history_build_ms": 0.0,
    "rag_llm_ms": 1316.7,
    "total_ms": 1316.8
  }
}
```

## Dependencies

- **Flask** — Web framework
- **LlamaIndex** — Document indexing, embeddings, and RAG chat engine
- **Flask-Session** — Server-side session storage for chat history
- **python-dotenv** — Load environment variables from `.env`
