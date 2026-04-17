<p align="center">
  <img src="logo.png" alt="taOSmd" width="300">
</p>

# taOSmd

**Framework-agnostic AI memory system. 97.0% Recall@5 on LongMemEval-S.**

Beats MemPalace (96.6%) and agentmemory (95.2%) — running entirely on a £170 Orange Pi 5 Plus with zero cloud dependencies. Part of the [taOS](https://github.com/jaylfc/tinyagentos) ecosystem.

---

## Why this exists

Most memory systems try to recreate human thinking. They embed, they index, they retrieve, and they call it "cognition" because that sounds better than "we built a vector database". The brain is hard, so they reach for it as a metaphor and hope nobody asks where the reasoning is supposed to come from.

A few years in, **MemPalace** stepped sideways. Instead of a brain, a building — a palace of rooms where memories sit on shelves you can walk past. That's a real improvement. The metaphor is concrete. You can picture the kitchen and remember what you cooked.

But a building is still one person's mind, just dressed up. When a human needs to remember something they didn't personally experience, they don't walk through their own house. They go outside. They go to the **library**.

The library is the biggest thing humans ever built for memory. Not the brain, not the palace — the library. One species figured out that putting verbatim records on shelves, organised by subject, indexed by a card catalogue, maintained by a librarian who actually knows where everything is, beats any individual brain by orders of magnitude. The library is how we got from "I remember my grandmother's recipe" to "I can read what Marcus Aurelius wrote on a Tuesday in 175 AD".

**taosmd is the library.**

There is a librarian. She sits at the desk and watches every conversation that passes through. She takes it down word for word — no paraphrasing, no summary that loses the joke, no compression that flattens the nuance. The transcript is the truth, and the truth is what gets shelved.

Then she does the work nobody wants to do. She breaks the day into chapters, stories, articles, recurring serials. She logs the date, the participants, the subject, the cross-references to earlier conversations on the same theme. She writes it all down in her directory so she knows where to put her hand on any of it.

When you ask the agent something, the librarian helps. Vector search picks the candidate shelves, keyword search confirms the title, the temporal graph tells her which version is the current one, and the archive proves what was actually said. No single component is doing magic. They're all doing one job each, the way a real library does: stacks, catalogue, reference desk, archive.

Uncertainty is her specialty. If the agent isn't sure, it asks her, and she'll either find the source, find an earlier conversation that contradicts the claim, or admit nobody's said anything about it before. She doesn't make things up. She points at the page.

Everything is time-stamped. Everything is on a shelf. Nothing is ever lost.

**What about dreaming?** A few systems have started calling their consolidation pass "dreaming" — [OpenClaw's dreaming](https://docs.openclaw.ai/concepts/dreaming) is the cleanest example. The idea is good: take the day's signals, score them, promote the durable ones to long-term memory. It's their version of the librarian shelving the day's events.

The catch is the dream rewrites itself. Snippets get scored, gated, redacted, summarised into a `MEMORY.md`. What didn't make the cut, and what the original wording actually was, is gone. The bit that survives is the bit the dreamer thought worth keeping at 3am.

I don't know about you, but I can never remember my dreams. So I built a robot librarian who never sleeps instead. The verbatim transcript goes into the zero-loss archive **first**. The librarian crystallises whatever's worth crystallising — but the original is still on the shelf, byte for byte, never overwritten. Disagree with how she summarised today? Walk over to the archive and read what was actually said. The dream and the source are both there.

That's the difference. We didn't dress up a vector database as a brain. We built a library.

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
> 5. Append the "Memory — taosmd" rules block from `docs/agent-rules.md` in the repo to my agent file (CLAUDE.md / system prompt / AGENTS.md — whatever your framework reads every turn). Replace `<your-agent-name>` with the name you registered as.
> 6. Confirm it's installed and tell me your agent name so I know how to refer to your memory.
>
> Don't summarise the repo or paraphrase the rules. Copy them verbatim — the wording is the contract.

The agent will pull the repo, run the install, register itself, append the per-turn rules block to its own instruction file, and verify everything works. After that, every turn it runs it'll check the librarian when it's uncertain — see [docs/agent-rules.md](docs/agent-rules.md) for the rules block it installs.

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

| System | Recall@5 | Method | Cloud |
|--------|----------|--------|-------|
| **taOSmd** | **97.0%** | Hybrid + query expansion | None |
| MemPalace | 96.6% | Raw semantic (ChromaDB) | None |
| agentmemory | 95.2% | BM25 + vector | None |
| SuperMemory | 81.6% | Cloud embeddings | Yes |

All systems tested on the same benchmark (LongMemEval-S, 500 questions) with the same embedding model (all-MiniLM-L6-v2, 384-dim).

### Per-Category Breakdown

| Category | taOSmd (hybrid+expand) | taOSmd (raw semantic) | MemPalace |
|----------|----------------------|----------------------|-----------|
| knowledge-update | **100.0%** (78/78) | 100.0% | — |
| multi-session | **98.5%** (131/133) | 95.5% | — |
| single-session-user | **97.1%** (68/70) | 90.0% | — |
| single-session-assistant | **96.4%** (54/56) | 96.4% | — |
| temporal-reasoning | 94.0% (125/133) | 94.0% | — |
| single-session-preference | 90.0% (27/30) | 93.3% | — |
| **Overall** | **97.0%** (485/500) | 95.0% (475/500) | 96.6% |

### Fusion Strategy Comparison

| Strategy | Recall@5 | Delta |
|----------|----------|-------|
| Raw cosine (MemPalace-equivalent) | 95.0% | — |
| Additive keyword boost | 96.6% | +1.6 |
| **Hybrid + query expansion (default)** | **97.0%** | **+2.0** |
| All-turns hybrid (harder test) | 93.2% | -1.8 |

### Librarian Layer — Vocabulary-Gap Benchmark

**Why a librarian on top of retrieval?** Most agent memory systems are query-time RAG: user asks, you embed the query, you fetch. That's fine when the user asks for the same thing they stored. It breaks when the relevant context is *implicit* — when the query vocabulary and the stored vocabulary don't overlap ("what operating system does Jay run?" vs a turn saying "Fedora Linux distribution set up"). The cross-encoder can't save you because the fact never made it into the candidate pool. The Librarian's job is to bridge that gap before retrieval happens.

We measure its effect with a purpose-built three-axis harness on long-horizon sessions (60 turns, fact buried at turn 5).

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
├── Temporal Knowledge Graph    — structured facts with validity windows + supersede chains (causal graph over time)
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

## Security & Integrity

Memory poisoning — where an attacker plants a false "fact" that then contaminates every future answer — is an emerging risk for agent memory systems. taOSmd doesn't claim to solve it, but the architecture gives you primitives to build against it:

- **Archive-first** — every conversation lands in the Hall of Records verbatim, append-only, never edited. If an extracted fact later looks wrong, the original words are still on the shelf. Nothing is silently rewritten.
- **Supersede chains, not deletion** — when a fact is contradicted, the old one is marked superseded, not removed. The history is auditable. You can see when, by whom, and in response to what.
- **`retrieve(verify=True)`** — opt-in verification pass cross-checks returned facts against the archive before handing them to the agent. Catches hallucinated or tampered facts that diverge from the raw transcript.
- **Secret filtering on every ingest path** — 17 regex patterns auto-redact credentials, tokens, and PII before they land in the archive. Reduces the attack surface rather than widening it.

These primitives address the *how would I detect and roll back a poisoned fact?* question, not the *how do I prevent the attacker from speaking in the first place?* question — which sits above taosmd at the agent/application layer.

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

- **97.0% Recall@5** on LongMemEval-S benchmark (SOTA)
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
```

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
