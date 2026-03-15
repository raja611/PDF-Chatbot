# PDF Chatbot

A Flask-based chatbot that lets you upload PDF or DOCX files and ask questions about their content using OpenAI embeddings and LlamaIndex.

## Project Structure

```
pdf-chat/
├── app.py                    # Main Flask application
├── guardrails.py             # Input/output guardrails (validation, PII, rate limiting)
├── metrics.py                # Prometheus metrics definitions + Flask integration
├── benchmark_embeddings.py   # Embedding strategy benchmark script
├── Dockerfile                # Docker build file
├── .dockerignore             # Files excluded from Docker build
├── requirements.txt          # Python dependencies
├── .env                      # API keys (not committed to git)
├── .gitignore
├── templates/
│   └── index.html            # Chat UI
├── uploads/                  # Uploaded files (auto-created)
├── storage/                  # Persisted vector index (auto-created)
└── embedding_cache/          # Cached embeddings (auto-created)
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

## Deployment with Docker

### Build the image

```bash
docker build -t pdf-chatbot .
```

### Run the container

```bash
docker run -d \
  --name pdf-chatbot \
  -p 5000:5000 \
  -e OPENAI_API_KEY=sk-your-key-here \
  -e FLASK_SECRET_KEY=your-secret-key \
  pdf-chatbot
```

The app is available at [http://localhost:5000](http://localhost:5000).

### Using docker compose

Create a `docker-compose.yml`:

```yaml
services:
  pdf-chatbot:
    build: .
    ports:
      - "5000:5000"
    env_file:
      - .env
    volumes:
      - uploads:/app/uploads
      - storage:/app/storage
      - embedding_cache:/app/embedding_cache

volumes:
  uploads:
  storage:
  embedding_cache:
```

Then run:

```bash
docker compose up -d
```

### Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | - | Your OpenAI API key |
| `FLASK_SECRET_KEY` | No | `dev-secret-key` | Secret key for Flask sessions |
| `PORT` | No | `5000` | Port the app listens on |

### Volumes

Mount these paths if you want data to persist across container restarts:

| Path | Purpose |
|---|---|
| `/app/uploads` | Uploaded PDF/DOCX files |
| `/app/storage` | Persisted vector index |
| `/app/embedding_cache` | Cached embeddings |

### Production notes

- Set `FLASK_SECRET_KEY` to a strong random value in production.
- The default Flask dev server is used. For production, use a WSGI server like gunicorn:

```dockerfile
# Replace the CMD in Dockerfile with:
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "app:app"]
```

Add gunicorn to `requirements.txt`:

```
gunicorn
```

- Use `--workers 1` because the app uses in-memory caches (global variables). Multiple workers would each have their own cache, wasting memory. Use `--threads` for concurrency instead.
- Behind a reverse proxy (nginx, Caddy, etc.), set `X-Forwarded-For` headers and trust the proxy.

## Application Flow

### Upload Flow

```
User selects PDF/DOCX
        |
        v
POST /upload (multipart form)
        |
        v
File saved to ./uploads/                      ~3ms
        |
        v
PDFReader / DocxReader parses into chunks      ~146ms
        |
        v
VectorStoreIndex batch-embeds all chunks       ~848ms  (batch_size=50)
        |
        v
Index persisted to ./storage/                  ~37ms
        |
        v
Index + engine cached in memory                ~29ms
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
RAG: embed query + vector search + LLM        ~1778ms  (OpenAI API)
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
| A1. Sequential (batch=1) | 2217ms | 1992ms | 148ms | 46ms | 29ms |
| A2. Batch (batch=20) | 1527ms | 1382ms | 70ms | 44ms | 29ms |
| A3. Batch (batch=50) | **1063ms** | **848ms** | 146ms | 37ms | 29ms |

**Query latency - component breakdown:**

| Architecture | Engine load | History build | RAG+LLM | Total |
|---|---|---|---|---|
| B1. Unoptimized, no history | 20.2ms | - | 1508ms | 1529ms |
| B2. Unoptimized, with history | 24.0ms | - | 1380ms | 1404ms |
| C1. **Optimized, no history** | **0.1ms** | **0.0ms** | 1778ms | **1778ms** |
| C2. Optimized, with history | 0.2ms | 0.1ms | 2147ms | 2147ms |
| C3. Optimized, repeated query | 0.1ms | 0.0ms | 2528ms | 2528ms |
| D1. Direct LLM (no RAG) | - | - | 1234ms | 1234ms |

**Optimization impact:**

| Metric | Before | After | Improvement |
|---|---|---|---|
| Engine load per query | 20.2ms | **0.1ms** | **200x faster** |
| History build | ~1ms | **0.0ms** | eliminated |
| Total overhead (non-LLM) | ~21ms | **0.1ms** | **210x faster** |

> The RAG+LLM call (OpenAI API) is the irreducible floor at ~1200-2800ms. All controllable overhead has been reduced to sub-millisecond.

### Where the time goes (optimized query)

```
Engine load (from cache)   :    0.1ms   (0%)
History build              :    0.0ms   (0%)
RAG + LLM response         : 1778.1ms  (100%)
-----------------------------------------
Total                      : 1778.1ms  (100%)
```

### Key Takeaways

- **Engine overhead reduced from ~20ms to 0.1ms** (200x) by caching the index and chat engine in memory.
- **Batch embedding** cuts upload time from 2217ms to 1063ms (2.1x faster) by reducing API round-trips.
- **All controllable latency is now sub-millisecond.** The remaining ~1200-1800ms is entirely the OpenAI API call (embedding the query + LLM response generation), which is the irreducible floor.
- **Direct LLM vs RAG:** Direct LLM averages ~1234ms but has no document context. RAG is comparable at ~1778ms but answers are grounded in the actual PDF content.

## Optimizations Applied to the App

| Optimization | Where | Impact |
|---|---|---|
| **In-memory index cache** | `get_cached_index()` | Index loaded from disk once, reused from memory (20ms -> 0.1ms) |
| **In-memory engine reuse** | `get_cached_engine()` | Chat engine created once, reused across queries |
| **Batch embedding** | `embed_batch_size=50` | 2.1x faster upload (2217ms -> 1063ms) |
| **Embedding disk cache** | `./embedding_cache/` + memory dict | Avoids re-embedding duplicate text chunks |
| **Index persistence** | `storage_context.persist()` | Documents embedded once; queries never re-index |
| **Component timing** | `timings_ms` in API responses | Every request returns ms-level breakdown |
| **Environment variables** | `.env` + `python-dotenv` | API keys not hardcoded in source |
| **Cache warmup on upload** | Pre-loads index + engine after indexing | First query after upload has zero cold-start |

## Guardrails

The `guardrails.py` module protects the app from bad inputs and unsafe outputs.

### Input guardrails

| Check | Rule | Response |
|---|---|---|
| Empty input | Message cannot be blank | 400 + `EMPTY_INPUT` |
| Too short | Min 2 characters | 400 + `INPUT_TOO_SHORT` |
| Too long | Max 2000 characters | 400 + `INPUT_TOO_LONG` |
| Prompt injection | Detects "ignore previous instructions", "act as", `<script>`, etc. | 400 + `INJECTION_DETECTED` |

### File guardrails

| Check | Rule | Response |
|---|---|---|
| No file | File field missing or empty | 400 + `NO_FILE` |
| Wrong type | Only `.pdf` and `.docx` allowed | 400 + `INVALID_FILE_TYPE` |
| Too large | Max 20 MB | 400 + `FILE_TOO_LARGE` |
| Empty file | 0 bytes | 400 + `EMPTY_FILE` |

### Output guardrails

| Check | Action |
|---|---|
| PII detection | Scans for emails, phone numbers, SSNs, credit card numbers |
| PII redaction | Replaces with `[REDACTED_EMAIL]`, `[REDACTED_PHONE]`, etc. |

The response includes a `guardrails` field showing what was detected:

```json
{
  "guardrails": {
    "pii_detected": true,
    "pii_types": ["email", "phone"],
    "pii_redacted": true
  }
}
```

### Rate limiting

- 20 queries per minute per session
- Returns 429 + `RATE_LIMITED` when exceeded

## Observability (Prometheus)

The `metrics.py` module exposes Prometheus metrics at `GET /metrics`.

### Available metrics

**Counters:**

| Metric | Labels | Description |
|---|---|---|
| `pdf_chatbot_requests_total` | method, endpoint, status | Total HTTP requests |
| `pdf_chatbot_uploads_total` | file_type, status | File uploads (success/blocked/unsupported) |
| `pdf_chatbot_queries_total` | status | Queries (success/blocked/rate_limited/error) |
| `pdf_chatbot_guardrail_blocks_total` | guardrail_code | Requests blocked by guardrails |
| `pdf_chatbot_pii_detections_total` | pii_type | PII found and redacted in output |
| `pdf_chatbot_errors_total` | endpoint, error_type | Unhandled exceptions |

**Histograms (latency):**

| Metric | Description |
|---|---|
| `pdf_chatbot_upload_duration_seconds` | Total upload + indexing time |
| `pdf_chatbot_upload_embed_duration_seconds` | Embedding-only time during upload |
| `pdf_chatbot_query_duration_seconds` | Total query latency |
| `pdf_chatbot_query_engine_load_seconds` | Engine load time per query |
| `pdf_chatbot_query_rag_llm_seconds` | RAG + LLM response time |

**Gauges:**

| Metric | Description |
|---|---|
| `pdf_chatbot_active_requests` | Currently in-flight requests |
| `pdf_chatbot_index_loaded` | 1 if index is in memory, 0 otherwise |
| `pdf_chatbot_embedding_cache_entries` | Number of cached embeddings |

### Scraping with Prometheus

Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: "pdf-chatbot"
    scrape_interval: 15s
    static_configs:
      - targets: ["localhost:5000"]
```

### Docker Compose with Prometheus + Grafana

```yaml
services:
  pdf-chatbot:
    build: .
    ports:
      - "5000:5000"
    env_file:
      - .env
    volumes:
      - uploads:/app/uploads
      - storage:/app/storage

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

volumes:
  uploads:
  storage:
```

Create `prometheus.yml` in the project root:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "pdf-chatbot"
    static_configs:
      - targets: ["pdf-chatbot:5000"]
```

Then: `docker compose up -d` and open Grafana at `http://localhost:3000` (admin/admin), add Prometheus as a data source (`http://prometheus:9090`).

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serves the chat UI |
| POST | `/upload` | Upload a PDF or DOCX file for indexing |
| POST | `/ask` | Optimized query (cached engine) |
| POST | `/ask_no_cache` | Unoptimized query (reloads from disk, for benchmarking) |
| GET | `/metrics` | Prometheus metrics endpoint |
| GET | `/health` | Health check (index status, cache size) |

### POST /upload

**Request:** `multipart/form-data` with a `file` field.

**Response:**
```json
{
  "status": "success",
  "timings_ms": {
    "file_save_ms": 3.4,
    "parse_ms": 146.2,
    "embed_index_ms": 847.8,
    "persist_ms": 37.3,
    "cache_warmup_ms": 28.7,
    "total_ms": 1063.4
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
    "rag_llm_ms": 1276.7,
    "total_ms": 1276.7
  },
  "guardrails": {
    "pii_detected": false
  }
}
```

## Dependencies

- **Flask** -- Web framework
- **LlamaIndex** -- Document indexing, embeddings, and RAG chat engine
- **Flask-Session** -- Server-side session storage for chat history
- **python-dotenv** -- Load environment variables from `.env`
- **prometheus-client** -- Prometheus metrics exposition
