<p align="center">
  <img src="logo.png" alt="taOSmd" width="300">
</p>

# taOSmd

**Framework-agnostic AI memory system. 97.0% end-to-end Judge accuracy on LongMemEval-S.**

End-to-end Judge accuracy means retrieve → generate → LLM-grade against the reference answer. Runs entirely on a £170 Orange Pi 5 Plus with zero cloud dependencies. Part of the [taOS](https://github.com/jaylfc/tinyagentos) ecosystem. Methodology and comparison notes in [docs/benchmarks.md](docs/benchmarks.md).

---

## Why this exists

Most memory benchmarks run on hosted models (GPT-4o-mini, Claude, Gemini) behind an API key — fine for prototypes, not fine if you're:

- **On modest hardware.** Running an agent on a £170 Orange Pi 5 Plus, an old laptop, or a Mac mini. No 24 GB 4090 — just an NPU or a CPU and ~16 GB RAM. The memory system has to fit *around* that, not on top of it.
- **Distributed across a few small machines.** Pi + desktop + laptop, pooled by the taOS stack. The memory layer lives across those nodes without assuming any single "real" machine.
- **Offline or air-gapped.** Forwarding a conversation turn to a third-party API is a compliance violation or a signal flare. Memory here can't mean "we shipped it to a hosted API".
- **Zero-loss by default.** The verbatim transcript goes into an append-only archive first; summaries and structure are layered on top, never *over*. The source is still on the shelf byte-for-byte. Disagree with a summary? Read what was actually said.

The 97.0% on LongMemEval-S was measured on the Orange Pi 5 Plus stack — the same hardware a user would actually deploy. No hidden hosted model doing the heavy lifting off-camera.

Zero cloud dependencies. Zero API keys. NPU if you have one, CPU if you don't, cluster if you want the throughput.

---

## Getting Started

### Let your agent install it

The cleanest way to install taosmd is to ask your agent to do it. Paste this message into Claude Code, Cursor, your OpenClaw shell, whatever:

> Please install taosmd as my memory system. The repo is github.com/jaylfc/taosmd.
>
> 1. Read the README so you understand what you're installing.
> 2. Run the install script: `curl -fsSL https://raw.githubusercontent.com/jaylfc/taosmd/master/scripts/setup.sh | bash`. Report any errors and stop if it fails.
> 3. Register yourself as an agent so you have your own isolated index. Pick a stable agent name (lowercase, no spaces) — the same name you'll use every time you call the librarian. If I have multiple agents in this framework, ask me what to name this one before registering.
> 4. Verify the install: call `taosmd.search("hello", agent="<your-name>")` — it should return an empty result, not an error.
> 5. Append the "Memory — taosmd" rules block to my agent file (CLAUDE.md / system prompt / AGENTS.md — whatever your framework reads every turn). Pull the block via `python -c "import taosmd; print(taosmd.agent_rules())"` — the file is shipped as package data at `taosmd/docs/agent-rules.md` so this works from both editable and wheel installs. Replace `<your-agent-name>` with the name you registered as.
> 6. Confirm it's installed and tell me your agent name so I know how to refer to your memory.
>
> Don't summarise the repo or paraphrase the rules. Copy them verbatim — the wording is the contract.

The agent will pull the repo, run the install, register itself, append the per-turn rules block to its own instruction file, and verify everything works. After that, every turn it runs it'll check the librarian when it's uncertain — see [taosmd/docs/agent-rules.md](taosmd/docs/agent-rules.md) for the rules block it installs (also available via `taosmd.agent_rules()`).

**Multiple agents in one framework?** Same install message works. The agent will ask you to name it before registering, so each agent gets its own shelf. The taosmd service itself stays as one process; only the per-agent indexes are separate. See [docs/multi-agent.md](docs/multi-agent.md) for the full naming convention, cross-agent reads, migration scenarios, and a five-agent worked example.

**Inside taOS?** Don't use this — taOS provisions taosmd automatically when you deploy an agent, and the rules block is baked into the agent template. This install path is for standalone framework users.

### One-Line Setup (manual)

> **Note:** Install scripts are new and awaiting full testing on clean environments. Please report issues.

```bash
curl -fsSL https://raw.githubusercontent.com/jaylfc/taosmd/master/scripts/setup.sh | bash
```

This will:
1. Clone the repo and install Python dependencies
2. Download the all-MiniLM-L6-v2 ONNX embedding model (90MB)
3. Install Ollama and pull Qwen3-4B for fact extraction + answering (2.6GB)
4. On RK3588: download the NPU-optimised Qwen3-4B RKLLM model instead (4.6GB)
5. Create the data directory and run a self-test

### Manual Install

```bash
git clone https://github.com/jaylfc/taosmd.git
cd taosmd
pip install -e .

# 1. Embedding model (required)
huggingface-cli download onnx-models/all-MiniLM-L6-v2-onnx --local-dir models/minilm-onnx

# 2. LLM for fact extraction + answering (required)
# Option A: x86 / ARM without NPU — use Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:4b

# Option B: Orange Pi / RK3588 with NPU — use rkllama
# Install rkllama: https://github.com/NotPunchnox/rkllama
huggingface-cli download dulimov/Qwen3-4B-rk3588-1.2.1-base \
  Qwen3-4B-rk3588-w8a8-opt-1-hybrid-ratio-0.0.rkllm \
  --local-dir ~/.rkllama/models/qwen3-4b-chat
```

### Tell Your Agent

If you're using Claude Code, OpenClaw, Cursor, or any AI coding agent, paste this into your conversation or CLAUDE.md / AGENTS.md:

> **Set up taOSmd as my persistent memory system.**
>
> 1. Clone https://github.com/jaylfc/taosmd and run `scripts/setup.sh` — it installs everything (embedding model, Qwen3-4B LLM, data stores, daily archive compression cron).
>
> 2. After setup, integrate taOSmd into your workflow:
>
> **Store facts you learn about me:**
> ```python
> from taosmd import KnowledgeGraph
> kg = KnowledgeGraph("~/.taosmd/knowledge-graph.db")
> await kg.init()
> await kg.add_triple("User", "prefers", "local models")
> ```
>
> **Archive every conversation turn (zero-loss — never summarise, never delete):**
> ```python
> from taosmd import Archive
> archive = Archive(archive_dir="~/.taosmd/archive", index_path="~/.taosmd/archive-index.db")
> await archive.init()
> await archive.record("conversation", {"role": "user", "content": message}, summary=message[:80])
> ```
>
> **Search my history when I ask about something:**
> ```python
> from taosmd import VectorMemory
> vmem = VectorMemory("~/.taosmd/vector-memory.db", embed_mode="onnx", onnx_path="<taosmd-dir>/models/minilm-onnx")
> await vmem.init()
> results = await vmem.search("What did I say about Docker?", hybrid=True)
> ```
>
> **Extract facts automatically from our conversations:**
> ```python
> from taosmd.memory_extractor import process_conversation_turn
> await process_conversation_turn(message, agent_name="assistant", kg=kg, archive=archive)
> ```
>
> **Before answering, check what you know about me:**
> ```python
> facts = await kg.query_entity("User")
> history = await archive.search_fts("relevant topic")
> similar = await vmem.search("the question", hybrid=True)
> ```
>
> The archive is append-only and kept forever. Every conversation, tool call, decision, and error should be recorded. Old archives are compressed to gzip daily at 3 AM. The knowledge graph tracks structured facts with temporal validity — update facts when they change, don't delete them.

---

## Benchmark Results

**97.0% end-to-end Judge accuracy on LongMemEval-S** (500 questions, standard test set). Harness: `benchmarks/longmemeval_runner.py`.

See [docs/benchmarks.md](docs/benchmarks.md) for the full LongMemEval-S breakdown, the LoCoMo (1540-QA multi-session) measurements with retrieval-architecture ablations, methodology, and per-hardware-tier configuration recommendations (12 GB GPU, Orange Pi NPU, RPi 4, low-end GPU).

### Per-Category Breakdown

| Category | hybrid + expand | raw semantic |
|----------|----------------|--------------|
| knowledge-update | **100.0%** (78/78) | 100.0% |
| multi-session | **98.5%** (131/133) | 95.5% |
| single-session-user | **97.1%** (68/70) | 90.0% |
| single-session-assistant | **96.4%** (54/56) | 96.4% |
| temporal-reasoning | 94.0% (125/133) | 94.0% |
| single-session-preference | 90.0% (27/30) | 93.3% |
| **Overall** | **97.0%** (485/500) | 95.0% (475/500) |

### Fusion Strategy Comparison

| Strategy | Judge accuracy | Delta |
|----------|---------------|-------|
| Raw cosine (same algorithm as MemPalace) | 95.0% | — |
| Additive keyword boost | 96.6% | +1.6 |
| **Hybrid + query expansion (default)** | **97.0%** | **+2.0** |
| All-turns hybrid (harder test) | 93.2% | -1.8 |

### Librarian Layer — Vocabulary-Gap Benchmark

The Librarian adds LLM-assisted query expansion on top of the vector + cross-encoder stack. We measure its effect with a purpose-built three-axis harness on long-horizon sessions (60 turns, fact buried at turn 5).

**Axis C — vocabulary-gap coherence** (2026-04-15, gemma4:e2b 5B, Fedora host):

| Config | Composite | recall@lag25 | recall@lag50 |
|--------|-----------|-------------|-------------|
| Vector-only | 0.752 | 30% | 30% |
| Full pipeline (+ cross-encoder) | 0.752 | 30% | 30% |
| **Full + Librarian** | **0.810** | **45%** | **55%** |

**+15.4% on the vocabulary-gap axis.** The cross-encoder alone adds nothing when the target fact is excluded from its candidate pool — only the Librarian's expansion bridges category→specific-name gaps (e.g. query: *"code editor"*, fact: *"Neovim lua config done"*). These are preliminary results on one class of retrieval failure; we're actively working on tougher benchmarks to stress-test staleness detection and multi-store routing before drawing composite conclusions.

## Architecture

```
taOSmd Memory Stack (v0.2):

Memory Layers:
├── Temporal Knowledge Graph    — structured facts with validity windows
├── Vector Memory               — hybrid search (semantic + keyword boost, ONNX MiniLM or Nomic)
├── Zero-Loss Archive           — append-only JSONL, FTS5 full-text search
├── Session Catalog             — LLM-derived timeline directory over archives
└── Crystal Store               — compressed session digests with lessons

Processing:
├── Memory Extractor            — regex (15ms) + LLM fact extraction (qwen3:4b)
├── Session Splitter            — 30-min gap heuristic, per-session split files
├── Session Enricher            — LLM topic/description/category (tiered: 1=heuristic, 2=4B, 3=9B+)
├── Session Crystallizer        — narrative digests, outcomes, lessons → KG
├── Secret Filtering            — 17 regex patterns, auto-redact on all ingest paths
└── Retention Scoring           — Ebbinghaus decay with hot/warm/cold tiers

Retrieval:
├── Parallel Fan-Out            — query all layers simultaneously (thorough mode)
├── Query Expansion             — entity extraction + temporal resolution
├── Intent Classifier           — routes to optimal layer, weights RRF merge
├── Cross-Encoder Reranker      — ms-marco-MiniLM ONNX second-stage reranking
├── Graph Expansion             — BFS traversal from search results through KG
└── Context Assembler           — core/archival split, token-budgeted L0-L3

Integration:
├── Backend Abstraction         — pluggable interface for platforms (taOS, Claude Code, etc.)
└── Cross-Memory Reflection     — cluster-then-synthesize insights from KG
```

taOSmd is a standalone library. Platform features like job scheduling, worker management, gaming detection, and mesh sync live in the host platform (e.g., [taOS](https://github.com/jaylfc/tinyagentos)).

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

- **97.0% end-to-end Judge accuracy** on LongMemEval-S benchmark (SOTA)
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

The 97.0% benchmark was achieved on this exact stack:

| Component | Model | Purpose | Runtime |
|-----------|-------|---------|---------|
| **Embedding** | all-MiniLM-L6-v2 (22M params) | Semantic vector search | ONNX Runtime on ARM CPU (0.3ms/embed) |
| **Embedding (alt)** | Qwen3-Embedding-0.6B | NPU-accelerated embedding | rkllama on RK3588 NPU |
| **Reranker** | Qwen3-Reranker-0.6B | Result reranking | rkllama on RK3588 NPU |
| **Query Expansion** | qmd-query-expansion 1.7B | Search query enrichment | rkllama on RK3588 NPU |
| **LLM (extraction + answering)** | Qwen3-4B | Fact extraction (72% recall) + QA from context | rkllama on RK3588 NPU (17s/turn) |
| **Vector Store** | SQLite + numpy | Cosine similarity search | CPU |
| **Full-Text Search** | SQLite FTS5 | Keyword search over archive | CPU |
| **Knowledge Graph** | SQLite | Temporal entity-relationship triples | CPU |

**Everything runs on the Pi. No external server needed.** The Qwen3-4B handles both fact extraction and question answering on the NPU. The ONNX embedding model runs in-process on the CPU. An optional GPU worker (e.g. Fedora with RTX 3060) can accelerate LLM tasks ~10x but is not required — the Pi is fully self-contained.

### Model Files

| Model | Size | Source |
|-------|------|--------|
| all-MiniLM-L6-v2 ONNX | 90MB | [onnx-models/all-MiniLM-L6-v2-onnx](https://huggingface.co/onnx-models/all-MiniLM-L6-v2-onnx) |
| Qwen3-Embedding-0.6B RKLLM | 935MB | Pre-installed with rkllama |
| Qwen3-Reranker-0.6B RKLLM | 935MB | Pre-installed with rkllama |
| qmd-query-expansion 1.7B RKLLM | 2.4GB | Custom conversion |
| Qwen3-4B RKLLM | 4.6GB | [dulimov/Qwen3-4B-rk3588-1.2.1-base](https://huggingface.co/dulimov/Qwen3-4B-rk3588-1.2.1-base) |

## Platform-Specific Setup

### RK3588 NPU (Orange Pi / Rock 5 / Radxa)

```bash
# Install rkllama (serves models on the NPU)
# See: https://github.com/NotPunchnox/rkllama

# The setup script handles this automatically, or manually:
huggingface-cli download dulimov/Qwen3-4B-rk3588-1.2.1-base \
  Qwen3-4B-rk3588-w8a8-opt-1-hybrid-ratio-0.0.rkllm \
  --local-dir ~/.rkllama/models/qwen3-4b-chat
```

### Optional: GPU Worker (x86 + NVIDIA)

Not required — the Pi is fully self-contained. A GPU worker gives ~10x speed on LLM tasks:

```bash
# On your GPU machine
ollama pull qwen3:4b  # Same model as the Pi — same quality

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

# LoCoMo (1540 QAs, multi-session conversations)
python benchmarks/locomo_runner.py --model gemma4:e2b

# LoCoMo with tunable retrieval levers (see docs/benchmarks.md)
python benchmarks/locomo_runner.py --model gemma4:e2b \
  --retrieval-top-k 20 --adjacent-turns 2 --llm-query-expansion
```

All benchmark numbers in `docs/benchmarks.md` pin the commit they were measured on.

## Support

If taOSmd is useful to you:

- **Star this repo** — it helps others find it
- **Donate:** [Buy Me a Coffee](https://buymeacoffee.com/jaylfc)
- **Contact:** jaylfc25@gmail.com
- **Hardware donations/loans:** We test on real hardware. If you have spare SBCs, GPUs, or dev boards and want to help expand compatibility, reach out.

## License

MIT

## Dependencies & Acknowledgements

**Core taOSmd (the 97.0% benchmark) is fully self-contained** — it uses only standard packages (SQLite, numpy, ONNX Runtime) plus the MiniLM embedding model. No external servers or forked repos needed.

**Optional integrations for the full taOS stack:**

| Component | Source | Notes |
|-----------|--------|-------|
| QMD (reranking + query expansion) | [jaylfc/qmd](https://github.com/jaylfc/qmd) (fork) | Adds rkllama NPU backend and `qmd serve` mode. Upstream [tobi/qmd](https://github.com/tobi/qmd) doesn't have NPU support yet. |
| rkllama (NPU model serving) | [NotPunchnox/rkllama](https://github.com/NotPunchnox/rkllama) | Upstream with minor patches for rerank endpoint |
| ONNX MiniLM | [onnx-models/all-MiniLM-L6-v2-onnx](https://huggingface.co/onnx-models/all-MiniLM-L6-v2-onnx) | Standard pre-exported model |
| Qwen3-4B RKLLM | [dulimov/Qwen3-4B-rk3588-1.2.1-base](https://huggingface.co/dulimov/Qwen3-4B-rk3588-1.2.1-base) | Community RK3588 conversion |

## Credits

Built by [jaylfc](https://github.com/jaylfc). Part of the [taOS](https://github.com/jaylfc/tinyagentos) ecosystem.

Benchmark dataset: [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (ICLR 2025)
Embedding model: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
