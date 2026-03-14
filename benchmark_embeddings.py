"""
Comprehensive benchmark — tests every architectural combination through the
actual Flask API endpoints and compares latency in MILLISECONDS.

Architectures compared:

  UPLOAD:
    A1. Sequential (batch=1)
    A2. Batch small (batch=20)
    A3. Batch large (batch=50)

  QUERY — UNOPTIMIZED (reload index from disk every request):
    B1. /ask_no_cache — full reload every query
    B2. /ask_no_cache — with chat history accumulating

  QUERY — OPTIMIZED (in-memory cached index + engine reuse):
    C1. /ask — cached engine, no history
    C2. /ask — cached engine, with history
    C3. /ask — cached engine, repeated same question (warm embedding)

  QUERY — NO EMBEDDINGS:
    D1. Direct LLM call — no RAG, no vector search

  Component-level timing breakdown is shown for each query.

Run:
    python benchmark_embeddings.py
"""

import os
import io
import time
import shutil
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.llms import ChatMessage

from app import (
    app, PERSIST_DIR, Settings, OPENAI_API_KEY,
    invalidate_caches, get_cached_engine, get_cached_index,
)

UPLOAD_DIR = "./uploads"
TEST_PDF = os.path.join(UPLOAD_DIR, "atlassian-git-cheatsheet.pdf")

SAMPLE_QUESTIONS = [
    "What is git?",
    "How do you create a new branch?",
    "What does git stash do?",
]

SEP = "=" * 78
THIN = "-" * 78

ms = lambda s: f"{s * 1000:.1f}ms"


def clear_storage():
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
    invalidate_caches()


def get_client():
    app.config["TESTING"] = True
    return app.test_client()


def upload(client, pdf_path):
    with open(pdf_path, "rb") as f:
        data = {"file": (io.BytesIO(f.read()), os.path.basename(pdf_path))}
        return client.post("/upload", data=data, content_type="multipart/form-data")


def ask_optimized(client, question):
    return client.post("/ask", json={"message": question})


def ask_unoptimized(client, question):
    return client.post("/ask_no_cache", json={"message": question})


def set_batch(size):
    Settings.embed_model = OpenAIEmbedding(
        api_key=OPENAI_API_KEY,
        embed_batch_size=size,
    )


def print_timings(timings, indent="    "):
    for k, v in timings.items():
        print(f"{indent}{k:<22}: {v}ms")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION A: Upload / Indexing
# ═══════════════════════════════════════════════════════════════════════════

def bench_uploads(client):
    print(f"\n{SEP}")
    print("  SECTION A: UPLOAD / INDEXING (all times in ms)")
    print(SEP)

    configs = [
        ("A1. Sequential (batch=1)",  1),
        ("A2. Batch (batch=20)",     20),
        ("A3. Batch (batch=50)",     50),
    ]
    results = []

    for label, batch_size in configs:
        print(f"\n{THIN}")
        print(f"  {label}")
        print(THIN)

        set_batch(batch_size)
        clear_storage()

        start = time.perf_counter()
        resp = upload(client, TEST_PDF)
        total = time.perf_counter() - start

        body = resp.get_json()
        timings = body.get("timings_ms", {})

        print_timings(timings)
        results.append({"label": label, "batch": batch_size, "total_ms": timings.get("total_ms", total * 1000), "timings": timings})

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION B: Query — UNOPTIMIZED (reload from disk every time)
# ═══════════════════════════════════════════════════════════════════════════

def bench_unoptimized(client):
    print(f"\n{SEP}")
    print("  SECTION B: QUERY - UNOPTIMIZED /ask_no_cache (all times in ms)")
    print(f"  (Reloads index from disk + recreates engine every request)")
    print(SEP)

    set_batch(50)
    clear_storage()
    upload(client, TEST_PDF)

    # B1: No history
    print(f"\n{THIN}")
    print("  B1. Unoptimized - no chat history")
    print(THIN)

    b1_times = []
    for q in SAMPLE_QUESTIONS:
        with client.session_transaction() as sess:
            sess["chat_history"] = []

        start = time.perf_counter()
        resp = ask_unoptimized(client, q)
        total = time.perf_counter() - start

        body = resp.get_json()
        timings = body.get("timings_ms", {})
        answer = body.get("answer", "")[:60]

        print(f"\n    Q: \"{q}\"")
        print_timings(timings)
        print(f"    A: \"{answer}...\"")
        b1_times.append(timings)

    # B2: With history
    print(f"\n{THIN}")
    print("  B2. Unoptimized - with accumulating chat history")
    print(THIN)

    with client.session_transaction() as sess:
        sess["chat_history"] = []

    b2_times = []
    for q in SAMPLE_QUESTIONS:
        start = time.perf_counter()
        resp = ask_unoptimized(client, q)
        total = time.perf_counter() - start

        body = resp.get_json()
        timings = body.get("timings_ms", {})
        answer = body.get("answer", "")[:60]

        print(f"\n    Q: \"{q}\"")
        print_timings(timings)
        print(f"    A: \"{answer}...\"")
        b2_times.append(timings)

    return {"B1_no_history": b1_times, "B2_with_history": b2_times}


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION C: Query — OPTIMIZED (in-memory cache + engine reuse)
# ═══════════════════════════════════════════════════════════════════════════

def bench_optimized(client):
    print(f"\n{SEP}")
    print("  SECTION C: QUERY - OPTIMIZED /ask (all times in ms)")
    print(f"  (In-memory cached index + reused chat engine)")
    print(SEP)

    set_batch(50)
    clear_storage()
    upload(client, TEST_PDF)

    # C1: No history, cached engine
    print(f"\n{THIN}")
    print("  C1. Optimized - no chat history (cached engine)")
    print(THIN)

    c1_times = []
    for q in SAMPLE_QUESTIONS:
        with client.session_transaction() as sess:
            sess["chat_history"] = []

        start = time.perf_counter()
        resp = ask_optimized(client, q)
        total = time.perf_counter() - start

        body = resp.get_json()
        timings = body.get("timings_ms", {})
        answer = body.get("answer", "")[:60]

        print(f"\n    Q: \"{q}\"")
        print_timings(timings)
        print(f"    A: \"{answer}...\"")
        c1_times.append(timings)

    # C2: With history, cached engine
    print(f"\n{THIN}")
    print("  C2. Optimized - with accumulating chat history")
    print(THIN)

    with client.session_transaction() as sess:
        sess["chat_history"] = []

    c2_times = []
    for q in SAMPLE_QUESTIONS:
        start = time.perf_counter()
        resp = ask_optimized(client, q)
        total = time.perf_counter() - start

        body = resp.get_json()
        timings = body.get("timings_ms", {})
        answer = body.get("answer", "")[:60]
        history_len = len(body.get("history", []))

        print(f"\n    Q: \"{q}\" [history: {history_len}]")
        print_timings(timings)
        print(f"    A: \"{answer}...\"")
        c2_times.append(timings)

    # C3: Repeated same question (warm everything)
    print(f"\n{THIN}")
    print("  C3. Optimized - same question repeated 3x (fully warm)")
    print(THIN)

    c3_times = []
    for i in range(3):
        with client.session_transaction() as sess:
            sess["chat_history"] = []

        start = time.perf_counter()
        resp = ask_optimized(client, SAMPLE_QUESTIONS[0])
        total = time.perf_counter() - start

        body = resp.get_json()
        timings = body.get("timings_ms", {})

        print(f"\n    Run {i+1}: total = {timings.get('total_ms', '?')}ms")
        print_timings(timings)
        c3_times.append(timings)

    return {"C1_no_history": c1_times, "C2_with_history": c2_times, "C3_repeated": c3_times}


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION D: Direct LLM (no embeddings, no RAG)
# ═══════════════════════════════════════════════════════════════════════════

def bench_direct_llm():
    print(f"\n{SEP}")
    print("  SECTION D: DIRECT LLM - No embeddings, no vector search (ms)")
    print(SEP)

    llm = Settings.llm
    d_times = []

    for q in SAMPLE_QUESTIONS:
        start = time.perf_counter()
        resp = llm.chat([ChatMessage(role="user", content=q)])
        elapsed = time.perf_counter() - start

        answer = str(resp)[:60]
        print(f"\n    Q: \"{q}\"")
        print(f"    total_ms              : {elapsed*1000:.1f}ms")
        print(f"    A: \"{answer}...\"")
        d_times.append({"total_ms": round(elapsed * 1000, 1)})

    return d_times


# ═══════════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

def print_final_summary(upload_res, unopt_res, opt_res, direct_res):
    print(f"\n\n{'#' * 78}")
    print("  FINAL COMPARISON - ALL ARCHITECTURES (milliseconds)")
    print(f"{'#' * 78}")

    # ── Upload ───────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  UPLOAD LATENCY")
    print(SEP)
    print(f"\n  {'Architecture':<35} {'Total ms':>10} {'Embed ms':>10} {'Parse ms':>10}")
    print(f"  {'-'*67}")

    for r in upload_res:
        t = r["timings"]
        print(f"  {r['label']:<35} {t.get('total_ms','?'):>10} {t.get('embed_index_ms','?'):>10} {t.get('parse_ms','?'):>10}")

    # ── Query comparison ─────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  QUERY LATENCY - HEAD-TO-HEAD COMPARISON")
    print(SEP)

    def avg_total(times_list):
        totals = [t.get("total_ms", 0) for t in times_list]
        return sum(totals) / len(totals) if totals else 0

    def avg_component(times_list, key):
        vals = [t.get(key, 0) for t in times_list]
        return sum(vals) / len(vals) if vals else 0

    rows = [
        ("B1. Unoptimized, no history",     unopt_res["B1_no_history"]),
        ("B2. Unoptimized, with history",    unopt_res["B2_with_history"]),
        ("C1. Optimized, no history",        opt_res["C1_no_history"]),
        ("C2. Optimized, with history",      opt_res["C2_with_history"]),
        ("C3. Optimized, repeated query",    opt_res["C3_repeated"]),
        ("D1. Direct LLM (no RAG)",          direct_res),
    ]

    print(f"\n  {'Architecture':<38} {'Avg Total':>11} {'Engine':>10} {'RAG+LLM':>10}")
    print(f"  {'-'*71}")

    for label, times in rows:
        avg_t = avg_total(times)
        eng = avg_component(times, "engine_load_ms") or avg_component(times, "index_load_ms")
        rag = avg_component(times, "rag_llm_ms")
        eng_str = f"{eng:.1f}ms" if eng else "-"
        rag_str = f"{rag:.1f}ms" if rag else f"{avg_t:.1f}ms"
        print(f"  {label:<38} {avg_t:>10.1f}ms {eng_str:>10} {rag_str:>10}")

    # ── Speedup matrix ───────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  OPTIMIZATION IMPACT")
    print(SEP)

    b1_avg = avg_total(unopt_res["B1_no_history"])
    c1_avg = avg_total(opt_res["C1_no_history"])
    c3_avg = avg_total(opt_res["C3_repeated"])
    d1_avg = avg_total(direct_res)

    b1_eng = avg_component(unopt_res["B1_no_history"], "index_load_ms") + avg_component(unopt_res["B1_no_history"], "engine_create_ms")
    c1_eng = avg_component(opt_res["C1_no_history"], "engine_load_ms")

    print(f"""
  In-memory index + engine cache:
    Unoptimized engine overhead : {b1_eng:.1f}ms (load from disk + create engine)
    Optimized engine overhead   : {c1_eng:.1f}ms (already in memory)
    Savings per query           : {b1_eng - c1_eng:.1f}ms

  Overall query latency:
    Unoptimized (B1)            : {b1_avg:.1f}ms avg
    Optimized (C1)              : {c1_avg:.1f}ms avg
    Optimized repeated (C3)     : {c3_avg:.1f}ms avg
    Direct LLM, no RAG (D1)    : {d1_avg:.1f}ms avg

  Speedups:
    Optimized vs Unoptimized    : {b1_avg / c1_avg:.2f}x faster
    Direct LLM vs Optimized RAG : {c1_avg / d1_avg:.2f}x slower (but document-aware)
""")

    # ── Where time goes ──────────────────────────────────────────────────
    print(f"{SEP}")
    print("  WHERE THE TIME GOES (optimized query breakdown)")
    print(SEP)

    c1_engine = avg_component(opt_res["C1_no_history"], "engine_load_ms")
    c1_hist = avg_component(opt_res["C1_no_history"], "history_build_ms")
    c1_rag = avg_component(opt_res["C1_no_history"], "rag_llm_ms")
    c1_total = avg_total(opt_res["C1_no_history"])

    pct = lambda part, total: f"{part/total*100:.0f}%" if total > 0 else "-"

    print(f"""
    Engine load (from cache)   : {c1_engine:>8.1f}ms  ({pct(c1_engine, c1_total)})
    History build              : {c1_hist:>8.1f}ms  ({pct(c1_hist, c1_total)})
    RAG + LLM response         : {c1_rag:>8.1f}ms  ({pct(c1_rag, c1_total)})
    -------------------------------------
    Total                      : {c1_total:>8.1f}ms  (100%)

  The RAG+LLM call dominates at ~{pct(c1_rag, c1_total)} of total time.
  Engine/history overhead is now in the single-digit milliseconds.
""")
    print(SEP)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def run_benchmark():
    if not os.path.exists(TEST_PDF):
        print(f"ERROR: Test PDF not found at {TEST_PDF}")
        print("Upload a PDF through the app first, or place one in ./uploads/")
        return

    client = get_client()
    original_embed = Settings.embed_model

    file_size = os.path.getsize(TEST_PDF) / 1024
    print(SEP)
    print("  COMPREHENSIVE ARCHITECTURE BENCHMARK (all times in milliseconds)")
    print(SEP)
    print(f"  File       : {os.path.basename(TEST_PDF)} ({file_size:.0f} KB)")
    print(f"  Model      : {original_embed.model_name}")
    print(f"  Questions  : {len(SAMPLE_QUESTIONS)}")
    print(f"  Tests      : 3 upload + 2 unoptimized + 3 optimized + 1 direct LLM")
    print(SEP)

    upload_res = bench_uploads(client)
    unopt_res = bench_unoptimized(client)
    opt_res = bench_optimized(client)
    direct_res = bench_direct_llm()

    Settings.embed_model = original_embed
    print_final_summary(upload_res, unopt_res, opt_res, direct_res)


if __name__ == "__main__":
    run_benchmark()
