# Session Catalog Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a three-stage processing pipeline (splitter → enricher → crystallizer) that converts raw zero-loss archive JSONL files into structured, searchable session records with per-session split files and timeline browsing.

**Architecture:** The pipeline reads archive JSONL files (never modifying them), detects session boundaries via time-gap heuristics, writes per-session split files, then uses an LLM to enrich sessions with topics/descriptions/categories. The enriched sessions feed into the existing crystallize system. A new timeline intent type routes temporal queries to the catalog.

**Tech Stack:** Python 3.10+, SQLite (FTS5), ONNX Runtime, Ollama (qwen3-4b / qwen3.5:9b), pytest + pytest-asyncio

---

## Tasks

### Task 1: Session Catalog Core — Schema and Splitter
Rewrite session_catalog.py with new schema, splitter, split file writer, query methods.
Tests: test_session_catalog.py (4 tests: detect sessions, create split files, catalog entries, idempotency)

### Task 2: Session Enricher — LLM Topic/Category Generation
Add enrich_session(), _llm_enrich(), _update_enrichment() to session_catalog.py.
Tests: enricher fallback, FTS update after enrichment.

### Task 3: Catalog Pipeline Orchestrator
Create catalog_pipeline.py with CatalogPipeline class: index_day(), detect_best_tier(), index_yesterday(), index_range(), rebuild().
Tests: pipeline index_day, tier detection.

### Task 4: Crystallize Integration
Add catalog_session_id column to crystals table.
Tests: column existence check.

### Task 5: Intent Classifier — Timeline Intent
Add INTENT_TIMELINE with detection patterns and search strategy including catalog_weight.
Tests: timeline queries classified correctly, non-timeline unchanged.

### Task 6: Context Assembler — Wire Catalog into L1/L2
Add catalog parameter, search catalog in L1 archival, load split file content in L2.

### Task 7: Update __init__.py and Final Wiring
Export SessionCatalog + CatalogPipeline, add data/sessions/ to .gitignore.

See full implementation details in the conversation history.
