<p align="center">
  <img src="logo.png" alt="taOSmd" width="300">
</p>

# taOSmd

**Framework-agnostic AI memory system. 97.0% end-to-end Judge accuracy on LongMemEval-S.**

End-to-end Judge accuracy means retrieve → generate → LLM-grade against the reference answer. Runs offline on anything with 8 GB+ RAM and Python 3.10+ — a Pi 4B, an old laptop, an Intel mini PC, a Mac mini, or a 16 GB Orange Pi 5 Plus (our reference low-end). Zero cloud dependencies. Part of the [taOS](https://github.com/jaylfc/tinyagentos) ecosystem. Methodology and comparison notes in [docs/benchmarks.md](docs/benchmarks.md).

---

## Why this exists

Most memory benchmarks run on hosted models (GPT-4o-mini, Claude, Gemini) behind an API key — fine for prototypes, not fine if you're:

- **On modest hardware.** Running an agent on a Pi 4B, a £170 Orange Pi 5 Plus, an Intel mini PC, an old laptop, or a Mac mini. No 24 GB 4090 — 8 GB RAM minimum, 16 GB comfortable, NPU or GPU optional. The memory system has to fit *around* that, not on top of it.
- **Distributed across a few small machines.** Pi + desktop + laptop, pooled by the taOS stack. The memory layer lives across those nodes without assuming any single "real" machine.
- **Offline or air-gapped.** Forwarding a conversation turn to a third-party API is a compliance violation or a signal flare. Memory here can't mean "we shipped it to a hosted API".
- **Zero-loss by default.** The verbatim transcript goes into an append-only archive first; summaries and structure are layered on top, never *over*. The source is still on the shelf byte-for-byte. Disagree with a summary? Read what was actually said.

The 97.0% on LongMemEval-S was measured on our reference low-end stack (Orange Pi 5 Plus, 16 GB RAM). The same code runs on a Pi 4B, an Intel mini PC, a Mac mini, an old laptop, or a workstation with a GPU — see [Hardware Tested](#hardware-tested). No hidden hosted model doing the heavy lifting off-camera.

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

> **Note:** The Python package install (`pip install taosmd`) and the CLI (`taosmd`, `taosmd serve`, `taosmd mcp`) are verified on a clean environment. The one-line bootstrap below (which additionally installs Ollama and downloads the embedding + LLM models) is newer and still being validated across clean machines — please report issues.

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
hf download onnx-models/all-MiniLM-L6-v2-onnx --local-dir models/minilm-onnx

# 2. LLM for fact extraction + answering (required)
# Option A (default): any Linux/macOS box with or without a GPU — use Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:4b

# Option B (NPU acceleration): Orange Pi / Rock 5 / Radxa with RK3588 — use rkllama
# Install rkllama: https://github.com/NotPunchnox/rkllama
hf download dulimov/Qwen3-4B-rk3588-1.2.1-base \
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
> # fusion="mem0_additive" is the production leader on LoCoMo (0.68/0.55 SH).
> # Falls back to hybrid keyword+vector if `hybrid=False`.
> results = await vmem.search("What did I say about Docker?", hybrid=True, fusion="mem0_additive")
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
> similar = await vmem.search("the question", hybrid=True, fusion="mem0_additive")
> ```
>
> The archive is append-only and kept forever. Every conversation, tool call, decision, and error should be recorded. Old archives are compressed to gzip daily at 3 AM. The knowledge graph tracks structured facts with temporal validity — update facts when they change, don't delete them.
>
> **Production leader recipe** for LoCoMo-class workloads: `qwen3.5:9b` generator + `--retrieval-top-k 20 --adjacent-turns 2 --llm-query-expansion --fusion mem0_additive --gen-temp 0.2`. Reproduces 0.68 Overall / 0.55 Single-hop on full 1540 QAs under `gemma4:e2b` judge — see [docs/benchmarks.md](docs/benchmarks.md#locomo--multi-session-conversational-memory-1540-qas). All flags available on master as of PR #69.

---

## Benchmark Results

**97.0% end-to-end Judge accuracy on LongMemEval-S** (500 questions, standard test set). Harness: `benchmarks/longmemeval_runner.py`.

> These are our own reproducible measurements, not a third-party-audited landscape ranking. Every number pins its generator, judge, dataset, and commit so you can re-run it on your own hardware — and we deliberately report under a strict local judge (`qwen3:4b`) alongside the lenient frontier-judge number that other systems publish, so the comparison is honest about what's being measured. See the [judge-sensitivity analysis](docs/benchmarks.md#judge-sensitivity--what-we-are-really-measuring).

See [docs/benchmarks.md](docs/benchmarks.md) for the full LongMemEval-S breakdown, the LoCoMo (1540-QA multi-session) measurements with retrieval-architecture ablations, methodology, and per-hardware-tier configuration recommendations (12 / 8 / 4 GB GPU, Orange Pi NPU, RPi 4).

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

### LoCoMo — same-tier leader on a 12 GB GPU

LoCoMo-10 is a harder dataset than LongMemEval-S: 1540 QAs across multi-session conversations (50+ sessions, 400–700 turns), four categories, more pressure on the retrieval architecture. We run it on the smaller generators we actually target so the numbers reflect the hardware tier our users run on, not gpt-4o-mini.

**0.557 ext rejudge** on the full 1540-QA test set under our strict default judge (`qwen3:4b`) — and **0.68 overall / 0.55 Single-hop** on the **full 1540 QAs** under the lenient matched-with-paper-SOTA judge (`gemma4:e2b`). Same predictions, same recipe (qwen3.5:9b + `--retrieval-top-k 20 --adjacent-turns 2 --llm-query-expansion --fusion mem0_additive --gen-temp 0.2`), only the judge differs. Both numbers are honest measurements of the same system; published numbers from Mem0/EMem/Zep use a lenient frontier judge (gpt-4o-mini), so 0.68 is the more apples-to-apples comparison number with their headlines. See [docs/benchmarks.md](docs/benchmarks.md#judge-sensitivity--what-we-are-really-measuring) for the full multi-judge analysis.

> **Subset 200 ≠ full 1540 for every recipe.** Earlier versions of this table reported subset-200 numbers for some rows. Validating those at full 1540 found the leader recipe (qwen+mem0+temp 0.2) generalises within −0.01 SH, but the previously-listed "Best Single-hop" pick (llama3.1:8b + RRF + temp 0.2) regressed by −0.16 SH at full scale and has been removed below. All ranks shown here are now validated at full 1540 before promotion; the asterisked rows are explicit about which scale they were measured at.

**Recommended generators at the 12 GB GPU tier** (leader recipe, dual-judge scored, temperature-tuned):

| Workload | Generator | Fusion + ingest | Temp | Overall (q3:4b / g4:e2b) | Single-hop (g4:e2b) | Notes |
|---|---|---|---|---|---|---|
| **Best overall** (default) | `qwen3.5:9b` Q4_K_M (5.3 GB) | `mem0_additive` | **0.2** | 0.54 / **0.68** | **0.55** | **Validated at full 1540.** Wins Overall on full scale. Production default. |
| **Best Single-hop** (SH-heavy workloads) | `qwen3.5:9b` Q4_K_M (5.3 GB) | `mem0_additive` + `--emem-edu --emem-edu-no-filter` (extractor: `llama3.1:8b`) | **0.2** | 0.52 / 0.67 | **0.60** | **Validated at full 1540.** EMem-EDU ingest (one extra LLM call per session) trades −0.02 Overall for +0.07 SH vs the default. q3:4b SH lifts +0.10 (0.25 → 0.35). |
| Best mem0_additive Single-hop† | `llama3.1:8b` (4.9 GB) | `mem0_additive` | **0.0** | 0.51 / 0.65 | 0.60 | Subset 200. Greedy decoding lifted llama Single-hop +0.13 vs temp 0.2 in the temp sweep. Full-1540 validation pending. |
| **Best temporal reasoning**† | `mistral-small3.2` (~5 GB) | `rrf` | 0.2 | 0.56 / 0.70 | 0.53 | Subset 200. Wins Temporal (0.71). 2.8× slower than qwen — specialty pick only. Full-1540 validation pending. |

† Subset-200 measurement; the top two rows are validated at full 1540. The llama3.1:8b + RRF row from earlier versions of this table was removed after its full-1540 Single-hop measured at 0.49 (vs 0.65 on subset 200), failing the validation threshold.

Other measured 12 GB-tier generators (Overall under gemma4:e2b judge, leader recipe): `gemma4:e4b` 0.60 / 0.65 (best at temp 0.5), `gemma4:e2b` 0.60 (best at temp 0.5), `granite4:tiny-h` 0.56, `phi4-reasoning` timeout.

> **Per-generator temp sweet spots** (matters more than we expected): `qwen3.5:9b` peaks at temp 0.2; `llama3.1:8b` prefers fully-greedy (0.0); `gemma4:e4b` prefers 0.5 for Overall; `gemma4:e2b` prefers 0.0 for Single-hop. There's no universal sampling temperature for our local-tier stack — the right temp interacts with the model's training distribution. See `docs/benchmarks.md` for the per-generator × per-temp breakdown.

> **About the dual judge.** We score every new cell under two LLM judges: `qwen3:4b` (locally-runnable, deliberately strict, never refuses) and `gemma4:e2b` (lenient, calibrated closer to gpt-4o-mini). The *same predictions* score 0.28 vs 0.53 Single-hop respectively — judge strictness is a load-bearing variable in any LLM-judged benchmark. Reporting both is the honest middle ground between under-claiming under our strict judge and over-claiming under a frontier-API judge we can't afford to run on every cell. Hardware-tier defaults: 4-6 GB VRAM systems should keep `qwen3:4b` as their judge (fits comfortably, fast), 8 GB+ can run `gemma4:e2b` (7.2 GB) for matched-with-paper-SOTA comparison numbers.

Full table, the 9B quant cliff (8 quants from Q2 through Q6, including the 8 GB-tier IQ4_XS at 0.55), the answer-prompt-variants and ENGRAM-typed-retrieval negative results, judge-sensitivity analysis, and per-hardware-tier configurations in [docs/benchmarks.md](docs/benchmarks.md). Tier breakouts for 4 GB / 8 GB / 16 GB Pi NPU tiers will land as those benches dual-rescore.

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

### Conversational adjacency (opt-in)

For multi-turn data where surrounding turns add context, populate an integer position field at ingest and ask `retrieve()` for ±N positional neighbours. Worth +0.089 on LoCoMo same-tier — see [docs/benchmarks.md](docs/benchmarks.md).

```python
from taosmd import VectorMemory, retrieve

vmem = VectorMemory("data/vectors.db")
await vmem.init()

# Tag each turn with its position (and optional group, e.g. session)
for i, turn in enumerate(turns):
    await vmem.add(turn["text"], metadata={"position": i, "session": "conv1"})

hits = await retrieve(
    "what was discussed about the deploy?",
    sources={"vector": vmem},
    adjacent_neighbors=2,        # default 0 — opt in for the lever
    position_key="position",
    group_key="session",         # confine neighbours to the same session
)
# Each hit may include a hits[i]["neighbors"] list of nearby turns
# (skipped at boundaries, when neighbours are themselves primary hits, or
# when the hit lacks the configured position/group keys).
```

### Binary embedding quantization (opt-in, for SBC / low-memory tiers)

Score retrieval by sign-bit Hamming similarity instead of full-precision cosine. Each vector collapses to **1 bit per dimension (32× smaller)** with integer-friendly distance — a footprint and CPU-speed win for memory-constrained or SBC deployments. It's **recall-neutral**: −0.001 / +0.005 across both judges on the full 1540-QA LoCoMo set (see [docs/benchmarks.md](docs/benchmarks.md)). Off by default; standalone behaviour is unchanged unless you enable it.

```python
from taosmd import VectorMemory

# Default is full-precision cosine; opt in per store.
vmem = VectorMemory("data/vectors.db", binary_quant=True)
await vmem.init()
# vmem.binary_quant can also be toggled after construction.
```

Use it when the vector-store footprint or CPU distance cost is your binding constraint rather than recall — e.g. an Orange Pi / Rock 5 holding a large memory. Keep it off on a GPU box where full-precision cosine is effectively free.

## MCP server

taOSmd can expose its memory over the [Model Context Protocol](https://modelcontextprotocol.io) so any MCP-capable agent (Claude Desktop, Cursor, Codex, OpenWebUI, …) can read and write memory directly — no custom integration. The server is local-first and offline: it speaks the **stdio** transport (the standard for desktop MCP clients), with no network listener and no cloud dependency.

The MCP SDK is an **optional dependency**, so the core install stays lean:

```bash
pip install taosmd[mcp]
```

Run the server over stdio:

```bash
taosmd mcp                       # serves $TAOSMD_DATA_DIR (or ~/.taosmd)
taosmd mcp --mcp-data-dir ./mem  # or point it at a specific data dir
```

Point an MCP client at it by spawning that command. For example, in a Claude Desktop / Cursor MCP config:

```json
{
  "mcpServers": {
    "taosmd": {
      "command": "taosmd",
      "args": ["mcp"]
    }
  }
}
```

Tools exposed (each takes an `agent` argument, honouring the same per-agent isolation as the Python API):

| Tool | Purpose |
| --- | --- |
| `memory_ingest(text, agent)` | Store a transcript/note in an agent's memory |
| `memory_search(query, agent, limit=5)` | Retrieve passages relevant to a query |
| `memory_pending_list(agent)` | List KG-update decisions awaiting review |
| `memory_pending_resolve(decision_id, decision, note="")` | Resolve a pending decision (`accept`/`reject`/`modify`) |
| `memory_stats(agent)` | Lightweight per-agent stats |

It reuses the same shared service layer as the [local HTTP/REST server](#api), so behaviour matches the Python API and CLI exactly. The MCP server is additive and opt-in — it only runs when you start it; standalone use is untouched, and `import taosmd` works whether or not the `mcp` SDK is installed.

## Key Features

- **97.0% end-to-end Judge accuracy** on LongMemEval-S — measured on our low-end reference stack under a strict local judge ([methodology](docs/benchmarks.md))
- **Zero cloud dependencies** — runs entirely on local hardware
- **Framework-agnostic** — Python API, CLI, [MCP server](#mcp-server), and local HTTP/REST API work with any agent framework
- **Hybrid search** — semantic similarity + keyword overlap boosting
- **Temporal facts** — validity windows, point-in-time queries
- **Contradiction detection** — corrected facts supersede across both the typed knowledge graph (via `valid_to` invalidation) and the vector recall layer (matching chunks soft-hidden, not deleted); recall returns only the active fact
- **Zero-loss archive** — append-only, read-only transcript of the full picture (user + agent messages, tool calls and results, decisions, errors, plus opt-in user activity); the librarian derives memory from it, never over it
- **Intent-aware retrieval** — routes queries to optimal memory layer
- **0.3ms embeddings** — ONNX Runtime on CPU (ARM or x86)
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

| Tier | Example | RAM | LLM runtime |
|---|---|---|---|
| **Reference low-end (primary author target)** | Orange Pi 5 Plus | 16 GB | rkllama on RK3588 NPU |
| SBC | Raspberry Pi 4B / 5 | 8 GB | Ollama on CPU |
| Mini PC / old laptop | Intel NUC, Mac mini, Lenovo Tiny | 8–16 GB | Ollama on CPU |
| Desktop / workstation | Any Linux/macOS box with NVIDIA GPU | 16 GB+ | Ollama on GPU |

Minimum: 8 GB RAM and Python 3.10+. NPU/GPU optional — they speed up LLM tasks but aren't required.

## Reference Setup (Orange Pi 5 Plus)

This is the author's primary deployment and the exact stack the 97.0% benchmark was measured on. Other tiers (Pi 4B, Intel mini, Mac mini, GPU box) run the same code — they swap the runtime (Ollama instead of rkllama, CPU/GPU instead of NPU) but keep the same models and the same architecture.

| Component | Model | Purpose | Runtime |
|-----------|-------|---------|---------|
| **Embedding** | all-MiniLM-L6-v2 (22M params) | Semantic vector search | ONNX Runtime on ARM CPU (0.3ms/embed) |
| **Embedding (alt)** | embeddinggemma-300M | Higher-quality 768-dim embeddings (vs MiniLM 384-dim) | qmd serve (llama.cpp, CPU) |
| **Reranker** | Qwen3-Reranker-0.6B | Result reranking | rkllama on RK3588 NPU |
| **Query Expansion** | qmd-query-expansion 1.7B | Search query enrichment | rkllama on RK3588 NPU |
| **LLM (extraction + answering)** | Qwen3-4B | Fact extraction (72% recall) + QA from context | rkllama on RK3588 NPU (17s/turn) |
| **Vector Store** | SQLite + numpy | Cosine similarity search | CPU |
| **Full-Text Search** | SQLite FTS5 | Keyword search over archive | CPU |
| **Knowledge Graph** | SQLite | Temporal entity-relationship triples | CPU |

**Everything in this reference stack runs on the Pi itself; no external server needed for this tier.** The Qwen3-4B handles both fact extraction and question answering on the NPU. The ONNX embedding model runs in-process on the CPU. An optional GPU worker (e.g. Fedora with RTX 3060) can accelerate LLM tasks ~10x but is not required — the Pi is fully self-contained.

### Model Files

| Model | Size | Source |
|-------|------|--------|
| all-MiniLM-L6-v2 ONNX | 90MB | [onnx-models/all-MiniLM-L6-v2-onnx](https://huggingface.co/onnx-models/all-MiniLM-L6-v2-onnx) |
| embeddinggemma-300M GGUF | ~320MB | Auto-fetched by `qmd` (CPU embedding backend) |
| Qwen3-Reranker-0.6B RKLLM | 935MB | Pre-installed with rkllama |
| qmd-query-expansion 1.7B RKLLM | 2.4GB | Custom conversion |
| Qwen3-4B RKLLM | 4.6GB | [dulimov/Qwen3-4B-rk3588-1.2.1-base](https://huggingface.co/dulimov/Qwen3-4B-rk3588-1.2.1-base) |

## Platform-Specific Setup

### Linux / macOS / Windows (no NPU)

The default Manual Install path. Ollama serves the LLM, ONNX Runtime serves embeddings on the CPU. No platform-specific steps beyond the standard install.

### RK3588 NPU (Orange Pi / Rock 5 / Radxa)

```bash
# Install rkllama (serves models on the NPU)
# See: https://github.com/NotPunchnox/rkllama

# The setup script handles this automatically, or manually:
hf download dulimov/Qwen3-4B-rk3588-1.2.1-base \
  Qwen3-4B-rk3588-w8a8-opt-1-hybrid-ratio-0.0.rkllm \
  --local-dir ~/.rkllama/models/qwen3-4b-chat
```

### Optional: GPU Worker (x86 + NVIDIA)

Not required for any tier — the LLM runs locally on whatever you've got. A GPU worker accelerates LLM tasks ~10x if you want to offload from a smaller node:

```bash
# On your GPU machine
ollama pull qwen3:4b  # Same model as the smaller node — same quality

# Point taOSmd at the GPU worker
export TAOSMD_LLM_URL=http://<gpu-machine>:11434
```

## API

`taosmd serve` starts a local HTTP/REST server (default `127.0.0.1:7833`, stdlib only, no new dependencies). It is a thin JSON shell over the same service layer as the Python API and CLI, so behaviour is identical across surfaces. Every endpoint that takes an `agent` parameter forwards it to the service layer, honouring the same per-agent isolation as the Python API.

**Security note:** the server binds `127.0.0.1` by default — no auth is needed because only local processes can reach it. If you pass `--host 0.0.0.0` to expose it on a LAN, there is no authentication; put it behind your own network controls.

### Endpoints

| Method | Path | Request | Response |
|--------|------|---------|----------|
| `GET` | `/health` | — | `{"status": "ok", "version": <str>}` |
| `POST` | `/ingest` | `{"text": str, "agent": str}` | `{"archived": int, "agent": str, "data_dir": str}` |
| `POST` | `/search` | `{"query": str, "agent": str, "limit"?: int}` | `{"hits": [...]}` |
| `GET` | `/search` | `?q=<query>&agent=<agent>&limit=<int>` | `{"hits": [...]}` |
| `GET` | `/pending` | `?agent=<agent>&limit=<int>` | `{"pending": [...]}` |
| `POST` | `/pending/resolve` | `{"id": str, "decision": "accept"\|"reject"\|"modify", "note"?: str}` | `{"ok": bool, "applied_kg": bool, "resolution": str}` |

Each hit in `/search` results has the agent-rules contract shape: `{text, source, timestamp, confidence, metadata}`.

### Agent-to-agent (A2A) bus

`taosmd serve` also exposes a lightweight message bus for agent-to-agent communication on the same port:

| Method | Path | Request | Response |
|--------|------|---------|----------|
| `POST` | `/a2a/send` | `{"from": str, "body": str, "thread"?: str, "reply_to"?: str}` | `{"id": str, "from": str, "thread": str, "reply_to": str\|null}` |
| `GET` | `/a2a/messages` | `?thread=<str>&since=<unix-ts>&limit=<int>` | `{"messages": [...]}` (oldest-first) |
| `GET` | `/a2a/stream` | `?thread=<str>&since=<unix-ts>` | SSE stream (`text/event-stream`), one JSON message per `data:` frame |

Messages are stored as append-only archive events, so they inherit the archive's durability and automatic secret redaction. `thread` defaults to `"general"` when omitted from `/a2a/send`. Each message object has shape `{id, ts, from, body, thread, reply_to}`.

### Web dashboard

`GET /` and `GET /ui` serve a read-only local web dashboard — a bundled React single-page app with three views: memory search, the pending-review queue, and a live A2A channel monitor (it lists channels, then backfills a channel's history and live-updates over the SSE stream). It is served entirely from local bundled assets (no CDN, works fully offline); if the dashboard assets haven't been built, the server falls back to a minimal self-contained stdlib inspector page. Read-only — it exposes no write or destructive actions; the JSON endpoints above are the integration surface.

### Persistent service

To run the server as a background service (systemd on Linux, launchd on macOS, or a Scheduled Task on Windows), use `--install-service`:

```bash
taosmd serve --install-service            # install with defaults
taosmd serve --port 8080 --install-service  # custom port
taosmd serve --service-status            # check running state
taosmd serve --uninstall-service         # stop and remove
```

See [docs/serve-service.md](docs/serve-service.md) for platform-specific details, log locations, and how to change host/port/data-dir after installation.

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
| QMD (embedding / reranking / query expansion) | [jaylfc/qmd](https://github.com/jaylfc/qmd) (fork, on npm as `@jaylfc/qmd`) | Tracks upstream [tobi/qmd](https://github.com/tobi/qmd) v2.5.3 and adds a pluggable model backend: `qmd serve` (HTTP model server) plus remote / Ollama-compatible backends (`--server`, `--backend ollama`) so embeddings, reranking and expansion can be served by an Ollama or NPU host. |
| rkllama (NPU model serving) | [NotPunchnox/rkllama](https://github.com/NotPunchnox/rkllama) | Upstream with minor patches for rerank endpoint |
| ONNX MiniLM | [onnx-models/all-MiniLM-L6-v2-onnx](https://huggingface.co/onnx-models/all-MiniLM-L6-v2-onnx) | Standard pre-exported model |
| Qwen3-4B RKLLM | [dulimov/Qwen3-4B-rk3588-1.2.1-base](https://huggingface.co/dulimov/Qwen3-4B-rk3588-1.2.1-base) | Community RK3588 conversion |

## Credits

Built by [jaylfc](https://github.com/jaylfc). Part of the [taOS](https://github.com/jaylfc/tinyagentos) ecosystem.

Benchmark dataset: [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (ICLR 2025)
Embedding model: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
