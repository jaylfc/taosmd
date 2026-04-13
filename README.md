<p align="center">
  <img src="logo.png" alt="taOSmd" width="300">
</p>

# taOSmd

**Framework-agnostic AI memory system. 97.2% Recall@5 on LongMemEval-S.**

Beats MemPalace (96.6%) and SuperMemory (81.6%) — running entirely on a £170 Orange Pi 5 Plus with zero cloud dependencies.

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

## Installation

```bash
pip install taosmd
```

Or from source:

```bash
git clone https://github.com/jaylfc/taosmd.git
cd taosmd
pip install -e .
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

## License

MIT

## Credits

Built by [JAN LABS](https://github.com/jaylfc). Part of the [taOS](https://github.com/jaylfc/tinyagentos) ecosystem.

Benchmark dataset: [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (ICLR 2025)
Embedding model: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
