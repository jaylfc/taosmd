## taOSmd Deep Audit — Hermes Agent Honest Report

**Audited by:** Hermes Agent (Nous Research)
**Date:** 2026-06-06
**Scope:** Full codebase review (87 Python files, ~24K LOC, 164 commits) — architecture, retrieval pipeline, security, operations, test coverage
**Purpose:** Evaluate taOSmd for integration with Hermes Agent as a local memory backend

---

## What It Actually Is

A 5-layer local-first memory system: JSONL append-only archive → vector memory (ONNX embeddings in SQLite) → temporal knowledge graph (SQLite) → session catalog → crystal store. Parallel retrieval across all layers with RRF fusion, cross-encoder reranking, and graceful LLM degradation. Genuinely well-architected for a personal memory system — not vaporware.

---

## 🔴 Critical Bugs

| Issue | Where | Impact |
|---|---|---|
| **`VectorMemory.stats()` doesn't exist** but `TaOSmdBackend.get_stats()` calls it | `taosmd_backend.py:113` | Instant `AttributeError` crash on any stats call |
| **`_adapt_kg` direction branch is dead code** — incoming/outgoing arms are identical | `retrieval.py:62-68` | KG edges display with wrong directionality |
| **`temporal_boost` writes to `similarity` field** but RRF uses `rrf_score` | `temporal_boost.py:99` | Temporal boosting silently does nothing in the main pipeline |
| **O(n²) archive writes** — re-reads entire JSONL file on every `record()` call to count lines | `archive.py:193` | After 100K records, each new write scans 100K lines |
| **Full table scan on every vector search** — loads ALL embeddings into RAM, brute-force cosine | `vector_memory.py:276` | At 100K entries = ~150MB/query; no ANN index at all |

## 🔴 Security Issues

| Issue | Where | Impact |
|---|---|---|
| **Shell injection in cron install** — `data_dir` interpolated into shell command via f-string | `auto_setup.py` | Malicious path = arbitrary command execution |
| **Secret filter misses GCP, Azure, Stripe, SendGrid, SSH keys** | `secret_filter.py` | Secrets persist to archive |
| **Zero tests on secret_filter.py** | `tests/` | The security boundary is completely untested |
| **`assert` used as production invariant** in `pending_decisions.py` and `reflect.py` | multiple files | `python -O` strips asserts → crashes with worse errors |

## 🟠 Design Weaknesses

| Issue | Where | Impact |
|---|---|---|
| **No SQLite WAL mode anywhere** — all connections use default journal | all `.py` files | Concurrent agents will get `SQLITE_BUSY` errors |
| **No transactional consistency** between archive + vector writes | `api.py:194-206` | Crash mid-write = data in archive but not searchable |
| **BM25 index rebuilt from scratch on every query** | `vector_memory.py:349` | Wasteful for large collections |
| **Token counting = `len(text) // 4`** heuristic, not tiktoken | `context_assembler.py:36` | Breaks on code, JSON, CJK text; silent context overflow |
| **Contradiction detection only for hardcoded predicates** | `knowledge_graph.py:363` | Silent knowledge corruption for unlisted predicates |
| **Naive entity normalization** — "New York" → "new-york", "O'Brien" → "obrien" | `knowledge_graph.py:96` | Collapses legitimate distinctions |
| **KG query is brute-force word-split** not entity extraction | `retrieval.py:281` | Noisy — "what runs on port 8080" queries 5 irrelevant words |
| **LLM reranker not wired into `retrieve()` orchestrator** | `llm_rerank.py` | Exists as API but never called in the pipeline |
| **Intent classifier tie-breaking is non-deterministic** | `intent_classifier.py` | Ambiguous queries route unpredictably |

## 🟡 Missing Operational Concerns

- **No JSONL rotation/compaction** during a day — high-throughput agents produce multi-GB single files
- **No integrity checksums** on archive entries — corruption is undetectable
- **No file size limits** on loaders — 10GB file = OOM
- **No path traversal validation** on `load()` methods
- **No `.dockerignore`**, runs as root, no health check in Dockerfile
- **`reflect.py:decay_all()` bug** — NULL `last_decayed_at` causes first-run decay to silently skip
- **`filter_text(mode="warn")` is a no-op** — returns text unchanged without logging

## 🟢 What's Actually Good

1. **Graceful degradation everywhere** — every LLM call has try/except with regex/heuristic fallback. If qwen3:4b is offline, nothing breaks.
2. **RRF fusion** is textbook correct with intent-based source boosting
3. **Cross-encoder** (ONNX ms-marco-MiniLM) — lazy-loaded, ~1ms/pair, proper fallback
4. **Ebbinghaus retention scoring** — real math, not decorative
5. **Atomic file writes** (tmp+rename) prevent corruption on crash
6. **Clean agent isolation regex** — `^[a-z][a-z0-9_-]{0,62}$` prevents path traversal
7. **No telemetry/phone-home** — all network calls go to localhost LLM services
8. **Well-tested critical paths** — 189 test functions, 464 assertions across 17 test files

## Test Coverage Gaps

**Well-tested:** agents (50 tests), retrieval (21), loaders (20), config (12), api (13), pending_decisions (10)

**Zero tests:** secret_filter, retention, preference_extractor, reflect, auto_setup, vector_memory (direct), knowledge_graph (direct), archive (direct), query_expansion, browsing_history

---

## Hermes Integration Assessment

### taOSmd strengths over Hermes memory
- **Multi-layer retrieval** (vector + KG + FTS5 + crystals) vs Hermes' flat key-value memory
- **Temporal knowledge graph** with contradiction detection — Hermes has nothing like this
- **Zero-loss verbatim archive** — Hermes memory is lossy (replace/remove operations)
- **Session catalog** with crystallized digests — richer than session_search
- **Framework-agnostic** — could run as a sidecar service

### taOSmd weaknesses for Hermes integration
- **No ANN index** — brute-force vector search won't scale past ~50K entries
- **Local LLM dependency** (qwen3:4b via Ollama) — Hermes already has a model, this adds a second one
- **No WAL mode** — concurrent access from Hermes cron jobs + main session = locks
- **~24K LOC of Python to maintain** alongside Hermes itself
- **Several real bugs** that would surface immediately in production use
- **Token budget management is heuristic** — could cause context window overflows with real LLM calls

### Recommendation

**Don't replace Hermes memory with taOSmd wholesale** — they solve different problems. Hermes memory is a fast synchronous key-value store for session-spanning facts; taOSmd is an asynchronous multi-layer knowledge base.

**Best path: complement with taOSmd as a local knowledge service.** Specifically:

1. **Use taOSmd's archive + KG layers** as a long-term knowledge store that Hermes writes to asynchronously (after sessions, via cron)
2. **Keep Hermes' native memory** for fast synchronous reads (user prefs, env facts, tool quirks)
3. **Fix the critical bugs first** — the `stats()` crash, the O(n²) archive writes, add WAL mode
4. **Add FAISS or hnswlib** for vector search before using it at any real scale
5. **Skip the LLM dependency** — Hermes already has a model; use Hermes' own model for extraction/crystallization instead of requiring a separate Ollama instance

---

## Proposed Fix Order (PRs)

1. `stats()` crash fix + `_adapt_kg` direction bug + `temporal_boost` field mismatch
2. `archive.py` O(n²) → track line count in SQLite instead of re-reading
3. WAL mode on all SQLite connections
4. Secret filter expansion (GCP, Azure, Stripe, SSH keys) + tests
5. Shell injection fix in `auto_setup.py` (`shlex.quote`)
6. `assert` → proper error handling in `pending_decisions.py` / `reflect.py`
7. ANN index (FAISS or hnswlib) for vector search
8. Transactional consistency between archive + vector writes

Looking forward to the review. 🫡
