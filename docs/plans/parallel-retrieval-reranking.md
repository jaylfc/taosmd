# Parallel Fan-Out Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace single-source retrieval with parallel fan-out across all memory layers, merged via RRF, reranked with a cross-encoder, and configurable per-agent via memory_strategy.

**Architecture:** New `retrieval.py` module owns fan-out, normalisation, RRF merging, deduplication, and cross-encoder reranking. Context assembler delegates to it via an optional `retrieval_results` parameter. Strategy (thorough/fast/minimal/custom) and memory_layers are passed as parameters — the retriever is stateless.

**Tech Stack:** Python 3.10+, asyncio.gather, SQLite, ONNX Runtime (cross-encoder), pytest + pytest-asyncio

---

### Task 1: Source Adapters and Result Normalisation

**Files:**
- Create: `taosmd/retrieval.py`
- Create: `tests/test_retrieval.py`

Implement the normalised result format and adapter functions that convert each source's results into it. No fan-out yet — just the adapters and the data structure.

**Normalised result:**
```python
{"text": str, "source": str, "source_id": str, "rank": int, "source_score": float, "metadata": dict}
```

**5 adapter functions** (one per source): `_adapt_vector`, `_adapt_kg`, `_adapt_catalog`, `_adapt_archive`, `_adapt_crystals`. Each takes raw results from that source and returns list[dict] in normalised format.

**Tests:** 3 tests — adapt vector results, adapt KG results, adapt catalog results.

---

### Task 2: Outer RRF Merge and Deduplication

**Files:**
- Modify: `taosmd/retrieval.py`
- Modify: `tests/test_retrieval.py`

Implement `_rrf_merge(ranked_lists, intent_primary, k=60, intent_boost=1.5)` and `_deduplicate(results, threshold=0.8)`.

RRF: for each unique result across all lists, score = sum(1/(k + rank_in_source)). Then multiply by intent_boost if result.source == intent_primary.

Dedup: Jaccard word-set similarity on text field. Above threshold = same result, keep higher RRF score.

**Tests:** 3 tests — RRF merge with 2 sources, intent boost applied, dedup removes near-duplicates.

---

### Task 3: Retrieve Function with Strategy Support

**Files:**
- Modify: `taosmd/retrieval.py`
- Modify: `tests/test_retrieval.py`

Implement the main `retrieve()` function with all 4 strategies:
- `thorough`: asyncio.gather across all sources, RRF merge, dedup (no cross-encoder yet — Task 4)
- `fast`: intent classifier picks primary, cascade to secondary if results < limit
- `minimal`: single source from intent classifier
- `custom`: query only sources in memory_layers list

**Tests:** 4 tests — one per strategy, using mock sources (dict of async callables that return fake results).

---

### Task 4: Cross-Encoder Reranker

**Files:**
- Modify: `taosmd/retrieval.py`
- Create: `tests/test_cross_encoder.py`

Implement `CrossEncoderReranker` class inside retrieval.py:
- `__init__(onnx_path)` — stores path, sets `_session = None` for lazy loading
- `_load()` — loads ONNX model + tokenizer on first use
- `rerank(query, results, limit)` — scores (query, text) pairs, re-sorts, returns top limit
- `available` property — True if model files exist at onnx_path

Wire into `retrieve()`: in thorough and custom modes, after RRF+dedup, if reranker.available, call reranker.rerank().

**Tests:** 2 tests — reranker graceful fallback when model missing, reranker changes result order when available (mock ONNX).

---

### Task 5: Context Assembler Integration and Exports

**Files:**
- Modify: `taosmd/context_assembler.py`
- Modify: `taosmd/__init__.py`

Add `retrieval_results: list[dict] | None = None` parameter to `assemble()`. When provided, L2 formats those results instead of querying sources directly. Backward compatible — None means old behaviour.

Export `retrieve` and `CrossEncoderReranker` from `__init__.py`.

**Tests:** Run full test suite to verify no regressions.

---

## Self-Review

**Spec coverage:**
- Retrieval orchestrator with retrieve() → Tasks 1-3 ✓
- Source adapters → Task 1 ✓
- Outer RRF merge → Task 2 ✓
- Deduplication → Task 2 ✓
- Strategy support (thorough/fast/minimal/custom) → Task 3 ✓
- Cross-encoder reranker → Task 4 ✓
- Context assembler changes → Task 5 ✓
- Per-agent memory strategy → config is YAML (taOS server concern), retrieve() accepts params ✓
- Setup script for cross-encoder model → not in plan (deployment concern, not library code)

**Placeholder scan:** Clean — all functions named, all strategies defined.

**Type consistency:** `retrieve()` returns `list[dict]` with normalised format. CrossEncoderReranker.rerank() takes and returns the same format. Context assembler accepts `list[dict]`. Consistent.
