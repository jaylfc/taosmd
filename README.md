<p align="center">
  <img src="logo.png" alt="taOSmd" width="300">
</p>

# taOSmd

**Framework-agnostic AI memory system. 97.2% Recall@5 on LongMemEval-S.**

Beats MemPalace (96.6%) and SuperMemory (81.6%) — running entirely on a £170 Orange Pi 5 Plus with zero cloud dependencies. Part of the [taOS](https://github.com/jaylfc/tinyagentos) ecosystem.

## Benchmark Results

| System | Recall@5 | Hardware | Cloud |
|--------|----------|----------|-------|
| **taOSmd** | **97.2%** | Orange Pi 5 Plus (£170) | None |
| MemPalace | 96.6% | Apple Silicon | None |
| SuperMemory | 81.6% | Cloud | Yes |
| GPT-4o | ~70% | Cloud | Yes |

### Per-Category Breakdown (LongMemEval-S, 500 questions)

| Category | Score |
|----------|-------|
| knowledge-update | 100.0% (78/78) |
| single-session-user | 100.0% (70/70) |
| multi-session | 98.5% (131/133) |
| temporal-reasoning | 95.5% (127/133) |
| single-session-assistant | 94.6% (53/56) |
| single-session-preference | 90.0% (27/30) |

## Architecture

```
taOSmd Memory Stack:
├── Temporal Knowledge Graph    — structured facts with validity windows
├── Vector Memory               — hybrid semantic+keyword search (ONNX MiniLM)
├── Zero-Loss Archive           — append-only JSONL, FTS5 full-text search
├── Memory Extractor            — regex (15ms) + LLM (17s on NPU)
├── Intent Classifier           — route queries to optimal memory layer
├── Context Assembler           — token-budgeted L0-L3 context loading
└── Contradiction Detection     — auto-resolve conflicting facts
```

## Quick Start

```python
from taosmd import KnowledgeGraph, VectorMemory, Archive

# Temporal Knowledge Graph
kg = KnowledgeGraph("data/kg.db")
await kg.init()
await kg.add_triple("Jay", "created", "taOS")
facts = await kg.query_entity("Jay")

# Vector Memory (hybrid search)
vmem = VectorMemory("data/vectors.db", embed_mode="onnx", onnx_path="models/minilm-onnx")
await vmem.init()
await vmem.add("Jay created taOS, a personal AI operating system")
results = await vmem.search("What is taOS?", hybrid=True)

# Zero-Loss Archive
archive = Archive("data/archive")
await archive.init()
await archive.record("conversation", {"content": "Hello"}, summary="User greeted agent")
events = await archive.search_fts("hello")
```

## Key Features

- **97.2% Recall@5** on LongMemEval-S benchmark (SOTA)
- **Zero cloud dependencies** — runs entirely on local hardware
- **Framework-agnostic** — HTTP API works with any agent framework
- **Hybrid search** — semantic similarity + keyword overlap boosting
- **Temporal facts** — validity windows, point-in-time queries
- **Contradiction detection** — auto-resolve conflicting facts
- **Zero-loss archive** — append-only, never modified, gzip compressed
- **Intent-aware retrieval** — routes queries to optimal memory layer
- **0.3ms embeddings** — ONNX Runtime on ARM CPU
- **Opt-in user tracking** — browsing history, app usage, search queries

## Embedding Model

The ONNX model (`models/minilm-onnx/model.onnx`) is not included in this repo due to size (90MB). Download it:

```bash
pip install sentence-transformers
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model.save('models/minilm-onnx')
"
```

Or download directly from [HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

## Embedding Speed

| Backend | Speed | Notes |
|---------|-------|-------|
| ONNX Runtime (CPU) | 0.3ms | Fastest, recommended |
| RKNN (NPU) | 16.7ms | Works but CPU is faster for small models |
| PyTorch (CPU) | 64ms | Heaviest, most compatible |

## Hardware Tested

- Orange Pi 5 Plus (RK3588, 16GB RAM) — primary target
- Fedora x86_64 (RTX 3060) — GPU worker for LLM extraction
- Should work on any Linux system with Python 3.10+

## Reference Setup (Orange Pi 5 Plus)

The 97.2% benchmark was achieved on this exact stack:

| Component | Model | Purpose | Runtime |
|-----------|-------|---------|---------|
| **Embedding** | all-MiniLM-L6-v2 (22M params) | Semantic vector search | ONNX Runtime on ARM CPU (0.3ms/embed) |
| **Embedding (alt)** | Qwen3-Embedding-0.6B | NPU-accelerated embedding | rkllama on RK3588 NPU |
| **Reranker** | Qwen3-Reranker-0.6B | Result reranking | rkllama on RK3588 NPU |
| **Query Expansion** | qmd-query-expansion 1.7B | Search query enrichment | rkllama on RK3588 NPU |
| **LLM (extraction)** | Qwen3-4B | Background fact extraction (72% recall) | rkllama on RK3588 NPU (17s/turn) |
| **LLM (answering)** | Qwen2.5-3B | QA from recalled context | Ollama on RTX 3060 (optional GPU worker) |
| **Vector Store** | SQLite + numpy | Cosine similarity search | CPU |
| **Full-Text Search** | SQLite FTS5 | Keyword search over archive | CPU |
| **Knowledge Graph** | SQLite | Temporal entity-relationship triples | CPU |

All models run locally. No cloud APIs, no external dependencies. The NPU models are served via [rkllama](https://github.com/NotPunchnox/rkllama) on port 8080. The ONNX embedding model requires no server — it loads directly in-process.

### Model Files

| Model | Size | Source |
|-------|------|--------|
| all-MiniLM-L6-v2 ONNX | 90MB | [onnx-models/all-MiniLM-L6-v2-onnx](https://huggingface.co/onnx-models/all-MiniLM-L6-v2-onnx) |
| Qwen3-Embedding-0.6B RKLLM | 935MB | Pre-installed with rkllama |
| Qwen3-Reranker-0.6B RKLLM | 935MB | Pre-installed with rkllama |
| qmd-query-expansion 1.7B RKLLM | 2.4GB | Custom conversion |
| Qwen3-4B RKLLM | 4.6GB | [dulimov/Qwen3-4B-rk3588-1.2.1-base](https://huggingface.co/dulimov/Qwen3-4B-rk3588-1.2.1-base) |

## Getting Started

### One-Line Setup

> **Note:** Install scripts are new and awaiting full testing on clean environments. Please report issues.

```bash
curl -fsSL https://raw.githubusercontent.com/jaylfc/taosmd/master/scripts/setup.sh | bash
```

This will:
1. Clone the repo and install dependencies
2. Download the all-MiniLM-L6-v2 ONNX model (90MB)
3. Create the data directory structure
4. Run a quick self-test to verify everything works

### Manual Install

```bash
git clone https://github.com/jaylfc/taosmd.git
cd taosmd
pip install -e .

# Download the embedding model
huggingface-cli download onnx-models/all-MiniLM-L6-v2-onnx --local-dir models/minilm-onnx
```

### Tell Your Agent

If you're using Claude Code, OpenClaw, Cursor, or any AI coding agent, paste this:

> Set up taOSmd as my memory system. Clone https://github.com/jaylfc/taosmd, install it, download the MiniLM ONNX model from huggingface (onnx-models/all-MiniLM-L6-v2-onnx) into models/minilm-onnx/, then run the self-test with `python -m taosmd.selftest`. Use the VectorMemory with embed_mode="onnx" and hybrid=True for search.

### Optional: Full Stack (Orange Pi / RK3588)

For the complete setup with NPU-accelerated LLM extraction:

```bash
# Install rkllama for NPU models (RK3588 only)
# See: https://github.com/NotPunchnox/rkllama

# Download NPU models
huggingface-cli download dulimov/Qwen3-4B-rk3588-1.2.1-base \
  Qwen3-4B-rk3588-w8a8-opt-1-hybrid-ratio-0.0.rkllm \
  --local-dir ~/.rkllama/models/qwen3-4b-chat

# The LLM enables background fact extraction (72% recall)
# Without it, regex extraction still works (39% recall, 15ms)
```

### Optional: GPU Worker (x86 + NVIDIA)

For faster LLM extraction and QA answering:

```bash
# On your GPU machine
ollama pull qwen2.5:3b

# Point taOSmd at the GPU worker
export TAOSMD_LLM_URL=http://<gpu-machine>:11434
```

## API

All components expose HTTP endpoints when used with the taOS server:

| Endpoint | Description |
|----------|-------------|
| `POST /api/kg/triples` | Add a fact |
| `GET /api/kg/query/{entity}` | Query facts about an entity |
| `POST /api/archive/record` | Archive an event |
| `GET /api/archive/events` | Search archived events |
| `POST /api/kg/classify` | Classify memory type |

## Running Benchmarks

```bash
# Full LongMemEval-S benchmark (500 questions)
python benchmarks/longmemeval_runner.py

# Recall@5 only
python benchmarks/longmemeval_recall.py

# Per-category breakdown
python benchmarks/longmemeval_granularity.py
```

## Support

If taOSmd is useful to you:

- **Star this repo** — it helps others find it
- **Donate:** [Buy Me a Coffee](https://buymeacoffee.com/jaylfc)
- **Contact:** [GitHub](https://github.com/jaylfc)
- **Hardware donations/loans:** We test on real hardware. If you have spare SBCs, GPUs, or dev boards and want to help expand compatibility, reach out.

## License

MIT

## Credits

Built by [jaylfc](https://github.com/jaylfc). Part of the [taOS](https://github.com/jaylfc/tinyagentos) ecosystem.

Benchmark dataset: [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (ICLR 2025)
Embedding model: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
