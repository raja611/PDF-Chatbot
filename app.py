from flask import Flask, render_template, request, jsonify, session
from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.readers.file import PDFReader, DocxReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import RichPromptTemplate
from flask_session import Session
from llama_index.core.llms import ChatMessage
from dotenv import load_dotenv

import os
import time
import logging
import hashlib
import json

from guardrails import validate_query, validate_file, check_rate_limit, check_output
from metrics import (
    register_metrics,
    UPLOAD_COUNT, UPLOAD_LATENCY, UPLOAD_EMBED_LATENCY,
    QUERY_COUNT, QUERY_LATENCY, QUERY_ENGINE_LATENCY, QUERY_RAG_LLM_LATENCY,
    GUARDRAIL_BLOCKS, PII_DETECTIONS, ERRORS,
    INDEX_LOADED, EMBEDDING_CACHE_SIZE,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)
register_metrics(app)

Settings.llm = OpenAI(api_key=OPENAI_API_KEY)
Settings.embed_model = OpenAIEmbedding(
    api_key=OPENAI_API_KEY,
    embed_batch_size=50,
)

PERSIST_DIR = "./storage"
EMBED_CACHE_DIR = "./embedding_cache"

# -- In-memory caches ----------------------------------------------------------
_index_cache = None
_chat_engine_cache = None
_embedding_mem_cache = {}


def get_cached_index(force_reload=False):
    """Load index once into memory, reuse on subsequent calls."""
    global _index_cache
    if _index_cache is None or force_reload:
        if not os.path.exists(PERSIST_DIR):
            INDEX_LOADED.set(0)
            return None
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        _index_cache = load_index_from_storage(storage_context)
        INDEX_LOADED.set(1)
    return _index_cache


def get_cached_engine(force_reload=False):
    """Create chat engine once, reuse on subsequent calls."""
    global _chat_engine_cache
    if _chat_engine_cache is None or force_reload:
        index = get_cached_index(force_reload)
        if index is None:
            return None
        _chat_engine_cache = index.as_chat_engine(
            chat_mode="condense_question", verbose=True
        )
    return _chat_engine_cache


def invalidate_caches():
    global _index_cache, _chat_engine_cache
    _index_cache = None
    _chat_engine_cache = None
    INDEX_LOADED.set(0)


def get_embed_cache_key(text):
    return hashlib.sha256(text.encode()).hexdigest()


def get_cached_embedding(text):
    """Check memory first, then disk."""
    key = get_embed_cache_key(text)
    if key in _embedding_mem_cache:
        return _embedding_mem_cache[key]

    cache_file = os.path.join(EMBED_CACHE_DIR, f"{key}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            emb = json.load(f)
        _embedding_mem_cache[key] = emb
        return emb
    return None


def save_embedding_to_cache(text, embedding):
    key = get_embed_cache_key(text)
    _embedding_mem_cache[key] = embedding
    os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(EMBED_CACHE_DIR, f"{key}.json")
    with open(cache_file, "w") as f:
        json.dump(embedding, f)
    EMBEDDING_CACHE_SIZE.set(len(_embedding_mem_cache))


template = RichPromptTemplate("""
You are a PDF assistant that helps users answer questions strictly using the content from the uploaded PDF document. 
Do not guess or fabricate any information. If the answer is not found in the PDF, respond with: 
"The information you're asking for is not available in the document." 
Keep your responses accurate, relevant, and concise.
Context:
{context_str}

Question:
{query_str}
""")


@app.route("/")
def index_page():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    t_start = time.perf_counter()

    file = request.files.get("file")
    file_check = validate_file(file)
    if not file_check:
        GUARDRAIL_BLOCKS.labels(guardrail_code=file_check.code).inc()
        UPLOAD_COUNT.labels(file_type="unknown", status="blocked").inc()
        return jsonify({"status": "error", "message": file_check.message, "code": file_check.code}), 400

    filename = file.filename.lower()
    ext = filename.rsplit(".", 1)[-1]

    saved_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(saved_path)
    t_save = time.perf_counter()

    if filename.endswith(".pdf"):
        documents = PDFReader().load_data(file=saved_path)
    elif filename.endswith(".docx"):
        documents = DocxReader().load_data(file=saved_path)
    else:
        UPLOAD_COUNT.labels(file_type=ext, status="unsupported").inc()
        return jsonify({"status": "error", "message": "Unsupported file type"}), 400
    t_parse = time.perf_counter()

    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    t_embed = time.perf_counter()

    index.storage_context.persist(persist_dir=PERSIST_DIR)
    t_persist = time.perf_counter()

    invalidate_caches()
    get_cached_index(force_reload=True)
    get_cached_engine(force_reload=True)
    t_warmup = time.perf_counter()

    session['chat_history'] = []

    total_s = t_warmup - t_start
    embed_s = t_embed - t_parse

    timings = {
        "file_save_ms": round((t_save - t_start) * 1000, 1),
        "parse_ms": round((t_parse - t_save) * 1000, 1),
        "embed_index_ms": round(embed_s * 1000, 1),
        "persist_ms": round((t_persist - t_embed) * 1000, 1),
        "cache_warmup_ms": round((t_warmup - t_persist) * 1000, 1),
        "total_ms": round(total_s * 1000, 1),
    }

    UPLOAD_LATENCY.observe(total_s)
    UPLOAD_EMBED_LATENCY.observe(embed_s)
    UPLOAD_COUNT.labels(file_type=ext, status="success").inc()
    logger.info(f"Upload timings: {timings}")

    return jsonify({"status": "success", "timings_ms": timings})


@app.route("/ask", methods=["POST"])
def ask():
    t_start = time.perf_counter()

    user_input = request.json.get("message") if request.is_json else None

    # -- Guardrails: input validation --
    input_check = validate_query(user_input)
    if not input_check:
        GUARDRAIL_BLOCKS.labels(guardrail_code=input_check.code).inc()
        QUERY_COUNT.labels(status="blocked").inc()
        return jsonify({"answer": input_check.message, "code": input_check.code}), 400

    # -- Guardrails: rate limiting --
    sid = session.sid if hasattr(session, "sid") else request.remote_addr
    rate_check = check_rate_limit(str(sid))
    if not rate_check:
        GUARDRAIL_BLOCKS.labels(guardrail_code=rate_check.code).inc()
        QUERY_COUNT.labels(status="rate_limited").inc()
        return jsonify({"answer": rate_check.message, "code": rate_check.code}), 429

    if not os.path.exists(PERSIST_DIR):
        return jsonify({"answer": "Please upload a file first."})

    chat_engine = get_cached_engine()
    t_engine = time.perf_counter()
    engine_s = t_engine - t_start
    QUERY_ENGINE_LATENCY.observe(engine_s)

    if chat_engine is None:
        return jsonify({"answer": "Please upload a file first."})

    raw_history = session.get('chat_history', [])

    chat_history = []
    for item in raw_history:
        chat_history.append(ChatMessage(role="user", content=item["user"]))
        chat_history.append(ChatMessage(role="assistant", content=item["assistant"]))
    t_history = time.perf_counter()

    try:
        response = chat_engine.chat(user_input, chat_history)
        t_llm = time.perf_counter()
        rag_llm_s = t_llm - t_history

        # -- Guardrails: output safety --
        answer_text, output_meta = check_output(response.response)
        if output_meta.get("pii_detected"):
            for pii_type in output_meta.get("pii_types", []):
                PII_DETECTIONS.labels(pii_type=pii_type).inc()

        raw_history.append({
            "user": user_input,
            "assistant": answer_text,
        })
        session['chat_history'] = raw_history

        total_s = t_llm - t_start
        timings = {
            "engine_load_ms": round(engine_s * 1000, 1),
            "history_build_ms": round((t_history - t_engine) * 1000, 1),
            "rag_llm_ms": round(rag_llm_s * 1000, 1),
            "total_ms": round(total_s * 1000, 1),
        }

        QUERY_LATENCY.observe(total_s)
        QUERY_RAG_LLM_LATENCY.observe(rag_llm_s)
        QUERY_COUNT.labels(status="success").inc()

        return jsonify({
            "answer": answer_text,
            "history": raw_history,
            "timings_ms": timings,
            "guardrails": output_meta,
        })

    except Exception as e:
        ERRORS.labels(endpoint="/ask", error_type=type(e).__name__).inc()
        QUERY_COUNT.labels(status="error").inc()
        logger.exception("Error in /ask")
        return jsonify({"answer": f"An error occurred: {str(e)}"}), 500


@app.route("/ask_no_cache", methods=["POST"])
def ask_no_cache():
    """Unoptimized endpoint for benchmarking (reloads index from disk every time)."""
    t_start = time.perf_counter()

    user_input = request.json.get("message") if request.is_json else None

    input_check = validate_query(user_input)
    if not input_check:
        GUARDRAIL_BLOCKS.labels(guardrail_code=input_check.code).inc()
        return jsonify({"answer": input_check.message, "code": input_check.code}), 400

    if not os.path.exists(PERSIST_DIR):
        return jsonify({"answer": "Please upload a file first."})

    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    t_index = time.perf_counter()

    chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
    t_engine = time.perf_counter()

    raw_history = session.get('chat_history', [])
    chat_history = []
    for item in raw_history:
        chat_history.append(ChatMessage(role="user", content=item["user"]))
        chat_history.append(ChatMessage(role="assistant", content=item["assistant"]))

    try:
        response = chat_engine.chat(user_input, chat_history)
        t_llm = time.perf_counter()

        answer_text, output_meta = check_output(response.response)

        timings = {
            "index_load_ms": round((t_index - t_start) * 1000, 1),
            "engine_create_ms": round((t_engine - t_index) * 1000, 1),
            "rag_llm_ms": round((t_llm - t_engine) * 1000, 1),
            "total_ms": round((t_llm - t_start) * 1000, 1),
        }

        QUERY_LATENCY.observe(t_llm - t_start)

        return jsonify({
            "answer": answer_text,
            "timings_ms": timings,
        })

    except Exception as e:
        ERRORS.labels(endpoint="/ask_no_cache", error_type=type(e).__name__).inc()
        logger.exception("Error in /ask_no_cache")
        return jsonify({"answer": f"An error occurred: {str(e)}"}), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "index_loaded": _index_cache is not None,
        "embedding_cache_size": len(_embedding_mem_cache),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
