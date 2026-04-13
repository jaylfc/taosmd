# Parallel Fan-Out Retrieval with Cross-Encoder Reranking

## Goal

Replace taOSmd's single-source retrieval with a parallel fan-out that queries all memory layers simultaneously, merges results via Reciprocal Rank Fusion (RRF), reranks with a cross-encoder, and supports per-agent strategy configuration. Target: push Recall@5 past 97.0% on LongMemEval-S.

## Architecture

New `retrieval.py` module that owns all retrieval logic. The context assembler delegates to it instead of querying sources directly.

```
Query → Retrieval Orchestrator
          │
          ├─ asyncio.gather (parallel fan-out)
          │    ├─ Vector Memory (inner RRF: semantic + keyword)
          │    ├─ Knowledge Graph (entity query + graph expansion)
          │    ├─ Session Catalog (FTS topic search)
          │    ├─ Archive (FTS full-text search)
          │    └─ Crystal Store (FTS narrative search)
          │
          ├─ Normalise → common result format
          ├─ Outer RRF merge (k=60) + intent weight boost
          ├─ Deduplicate (Jaccard > 0.8)
          ├─ Cross-encoder rerank top 3*limit → limit
          │
          └─ Return ranked results
               │
               └─ Context Assembler formats L0-L3 for LLM
```

## Retrieval Orchestrator (`taosmd/retrieval.py`)

### Core Function

```python
async def retrieve(
    query: str,
    strategy: str = "thorough",
    memory_layers: list[str] | None = None,
    sources: dict = {},
    limit: int = 5,
) -> list[dict]
```

`sources` is a dict of initialised stores: `{"vector": VectorMemory, "kg": KnowledgeGraph, "catalog": SessionCatalog, "archive": ArchiveStore, "crystals": CrystalStore}`. Only available sources are queried — missing keys are skipped.

### Strategies

| Strategy | Fan-out | Sources | Cross-encoder | Intent classifier role |
|----------|---------|---------|---------------|----------------------|
| `thorough` | All parallel | All available | Yes | Weighting (boost predicted source 1.5x) |
| `fast` | Cascade | Primary → secondary if needed | No | Selection (pick primary source) |
| `minimal` | Single | Intent-predicted best | No | Selection |
| `custom` | Selected parallel | From `memory_layers` list | Yes | Weighting |

### Source Adapters

Each source returns results in different shapes. The orchestrator normalises them to a common format before merging:

```python
{
    "text": str,
    "source": str,         # "vector" / "kg" / "catalog" / "archive" / "crystals"
    "source_id": str,
    "rank": int,
    "source_score": float,
    "metadata": dict,
}
```

Adapter functions per source:
- **vector** → `search()` results already have `text`, `similarity`, `metadata`
- **kg** → `query_entity()` results mapped to `"{subject} {predicate} {object}"` text
- **catalog** → `search_topic()` results mapped to `"[{date} {start}-{end}] {topic}: {description}"` text
- **archive** → `search_fts()` results mapped to summary + highlight text
- **crystals** → `search()` results mapped to narrative text

### Outer RRF Merge

```python
rrf_score = sum(1 / (k + rank_in_source)) for each source where result appears
```

k = 60 (standard). After RRF, apply intent weight boost: multiply score by 1.5 for results from the intent-predicted primary source.

### Deduplication

After RRF merge, deduplicate by Jaccard word-set similarity on text field. Threshold = 0.8. Keep the result with the higher RRF score.

## Cross-Encoder Reranker

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (ONNX variant, ~90MB)
**Location:** `models/cross-encoder-onnx/`

Embedded in `retrieval.py` as a `CrossEncoderReranker` class:
- `__init__(onnx_path)` — stores path, lazy loads on first use
- `rerank(query, results, limit)` — scores (query, text) pairs via ONNX, re-sorts, returns top `limit`
- Graceful fallback: if model not available, returns RRF-ranked results unchanged

**Performance:** ~1ms per (query, doc) pair. For 15 candidates (3x limit): ~15ms total.

**Pipeline position:** Retrieve 3x limit from each source → outer RRF → deduplicate → cross-encoder reranks merged pool → return top `limit`.

## Per-Agent Memory Strategy

**Configuration in agent YAML:**

```yaml
memory:
  strategy: thorough
  layers:           # only for custom strategy
    - vector
    - kg
    - catalog
```

**Defaults:** `strategy: thorough`, no layers field. Agents without a memory section get `thorough`.

The retrieval orchestrator is stateless — `strategy` and `memory_layers` are passed as parameters per call. The taOS server reads agent YAML and passes them through. The taOSmd management app (spec 3) provides a UI for editing these settings.

## Context Assembler Changes

`context_assembler.py` is modified to accept pre-retrieved results:

```python
async def assemble(
    self,
    query: str,
    retrieval_results: list[dict] | None = None,  # NEW — from retrieval orchestrator
    ...
) -> dict
```

When `retrieval_results` is provided:
- L1 still loads core KG facts (always present)
- L2 formats the retrieval results instead of querying sources directly
- L3 loads deep context from split files / archive lines for top-ranked results

When `retrieval_results` is None (backward compat), L2 queries sources directly as before.

## Intent Classifier Changes

Add `catalog_weight` to all strategies (already done in spec 1). No other changes — the classifier already returns weights that the retriever uses for boosting.

## Setup Script Changes

Add cross-encoder ONNX model download to `scripts/setup.sh`:

```bash
# Cross-encoder for reranking (optional, ~90MB)
huggingface-cli download cross-encoder/ms-marco-MiniLM-L-6-v2-onnx --local-dir models/cross-encoder-onnx
```

## Benchmark Expectations

Based on our benchmark results:
- Raw semantic: 95.0%
- Hybrid (inner RRF): 96.6%
- Hybrid + query expansion: 97.0%
- Parallel fan-out + outer RRF + cross-encoder: target 97.5%+

The cross-encoder is expected to rescue 2-3 of the remaining 15 misses by better distinguishing similar-scoring candidates. Fan-out adds catalog and archive as retrieval sources that may catch misses the vector store alone cannot.

## Out of Scope

- taOSmd management app (spec 3)
- Embedding model upgrade (all-mpnet-base-v2) — separate investigation
- LLM-based query expansion in the retriever — already in query_expansion.py, wired by caller
