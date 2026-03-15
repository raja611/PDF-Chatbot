"""
Prometheus metrics for the PDF chatbot.

Exposes /metrics endpoint for scraping.
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from flask import request, Response
import time
import functools


# -- Counters ------------------------------------------------------------------

REQUEST_COUNT = Counter(
    "pdf_chatbot_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

UPLOAD_COUNT = Counter(
    "pdf_chatbot_uploads_total",
    "Total file uploads",
    ["file_type", "status"],
)

QUERY_COUNT = Counter(
    "pdf_chatbot_queries_total",
    "Total queries asked",
    ["status"],
)

GUARDRAIL_BLOCKS = Counter(
    "pdf_chatbot_guardrail_blocks_total",
    "Requests blocked by guardrails",
    ["guardrail_code"],
)

PII_DETECTIONS = Counter(
    "pdf_chatbot_pii_detections_total",
    "PII detected and redacted in output",
    ["pii_type"],
)

ERRORS = Counter(
    "pdf_chatbot_errors_total",
    "Total errors",
    ["endpoint", "error_type"],
)

# -- Histograms ----------------------------------------------------------------

UPLOAD_LATENCY = Histogram(
    "pdf_chatbot_upload_duration_seconds",
    "Upload + indexing latency",
    buckets=[0.5, 1, 2, 5, 10, 30, 60],
)

UPLOAD_EMBED_LATENCY = Histogram(
    "pdf_chatbot_upload_embed_duration_seconds",
    "Embedding-only latency during upload",
    buckets=[0.1, 0.5, 1, 2, 5, 10],
)

QUERY_LATENCY = Histogram(
    "pdf_chatbot_query_duration_seconds",
    "Query (ask) total latency",
    buckets=[0.5, 1, 2, 5, 10, 30],
)

QUERY_ENGINE_LATENCY = Histogram(
    "pdf_chatbot_query_engine_load_seconds",
    "Engine load latency per query",
    buckets=[0.0001, 0.001, 0.01, 0.05, 0.1, 0.5],
)

QUERY_RAG_LLM_LATENCY = Histogram(
    "pdf_chatbot_query_rag_llm_seconds",
    "RAG + LLM response latency",
    buckets=[0.5, 1, 2, 5, 10, 30],
)

# -- Gauges --------------------------------------------------------------------

ACTIVE_REQUESTS = Gauge(
    "pdf_chatbot_active_requests",
    "Currently in-flight requests",
    ["endpoint"],
)

INDEX_LOADED = Gauge(
    "pdf_chatbot_index_loaded",
    "Whether an index is currently loaded in memory (1=yes, 0=no)",
)

EMBEDDING_CACHE_SIZE = Gauge(
    "pdf_chatbot_embedding_cache_entries",
    "Number of embeddings cached in memory",
)

# -- Info ----------------------------------------------------------------------

APP_INFO = Info(
    "pdf_chatbot",
    "Application metadata",
)
APP_INFO.info({
    "version": "2.0.0",
    "embed_model": "text-embedding-ada-002",
})


# -- Flask integration ---------------------------------------------------------

def register_metrics(app):
    """Register /metrics endpoint and request hooks."""

    @app.route("/metrics")
    def prometheus_metrics():
        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

    @app.before_request
    def _before():
        request._prom_start = time.perf_counter()
        endpoint = request.path
        ACTIVE_REQUESTS.labels(endpoint=endpoint).inc()

    @app.after_request
    def _after(response):
        endpoint = request.path
        ACTIVE_REQUESTS.labels(endpoint=endpoint).dec()

        if endpoint == "/metrics":
            return response

        elapsed = time.perf_counter() - getattr(request, "_prom_start", time.perf_counter())
        status = str(response.status_code)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status=status,
        ).inc()

        return response

    return app
