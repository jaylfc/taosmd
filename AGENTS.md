# taOSmd — Agent Integration Guide

This file is for AI agents (Claude Code, OpenClaw, Cursor, Hermes, or any coding assistant).

## Before You Install

**Do not overwrite an existing `AGENTS.md` or `CLAUDE.md`.** If one already exists in this project:

1. Read it first and check for any existing memory system instructions (Mem0, Zep, MemGPT/Letta, OpenClaw's built-in memory, LangChain memory modules, or any other `remember()`/`memorize()` calls).
2. If you find an existing memory system, **stop and ask the user** whether it should be disabled, removed, or kept alongside taOSmd. Running two memory systems in parallel without coordination will cause duplicate storage and conflicting retrieval results.
3. Only proceed once the user has confirmed what to do with any existing memory integrations.

If no `AGENTS.md` exists, or the existing one has no memory system instructions, you can safely append this file's contents to it.

---

## What taOSmd Does

taOSmd is a persistent memory system. It gives you structured, searchable memory across sessions:

- **Knowledge Graph** — structured facts with validity windows (`User prefers local models`)
- **Vector Memory** — hybrid semantic + keyword search (0.3ms per query on CPU)
- **Zero-Loss Archive** — every conversation turn, verbatim, forever. Never summarised, never deleted.
- **Session Catalogue** — LLM-derived timeline directory over the archive, organised by topic
- **Crystal Store** — compact session digests with lessons learned

97.0% Recall@5 on LongMemEval-S (a retrieval metric, like-for-like with MemPalace 96.6% and agentmemory 95.2%), measured locally on a low-end reference stack. The end-to-end Judge baseline (corrected, post-PR #176) is 42.8% under a strict Qwen judge / 51.2% under llama3.1:8b with the shipped qwen3.5:9b generator; the `factual-recall` generator profile (gemma4:12b at 12 GB) reaches 53.8 / 61.4, parity with MemOS-lossless (54.0 / 61.2). Runs entirely locally, no cloud dependencies.

---

## Setup

### One-line install

```bash
curl -fsSL https://raw.githubusercontent.com/jaylfc/taosmd/master/scripts/setup.sh | bash
```

This installs the embedding models (all-MiniLM-L6-v2 ONNX, 90MB, plus snowflake-arctic-embed-s ONNX, ~130MB, the shipped dense default), pulls Qwen3-4B via Ollama (2.6GB), creates all data stores, and installs the nightly librarian cron (3 AM: compresses old archive files via `compress_old_files()` and runs `CatalogPipeline.index_yesterday()` to catalogue, crystallize, and update the knowledge graph for the previous day). On RK3588 (Orange Pi, Rock 5) it uses the NPU-optimised model instead.

### Manual install

```bash
git clone https://github.com/jaylfc/taosmd.git
cd taosmd && pip install -e .
huggingface-cli download onnx-models/all-MiniLM-L6-v2-onnx --local-dir models/minilm-onnx
ollama pull qwen3:4b
```

---

## Your Core Responsibilities

Once set up, you must do three things in every session:

### 1. Archive every conversation turn (mandatory)

Archive EVERYTHING — user messages, your responses, tool calls, errors. The archive is append-only and kept forever. This is not optional: missed turns cannot be recovered.

```python
import os
from taosmd import Archive

archive = Archive(
    archive_dir=os.path.expanduser("~/.taosmd/archive"),
    index_path=os.path.expanduser("~/.taosmd/archive-index.db"),
)
await archive.init()

# Call this for every turn — both user messages and your responses
await archive.record(
    "conversation",
    {"role": "user", "content": user_message, "timestamp": time.time()},
    summary=user_message[:80]
)
await archive.record(
    "conversation",
    {"role": "assistant", "content": your_response, "timestamp": time.time()},
    summary=your_response[:80]
)

# Also archive tool calls and errors
await archive.record("tool_call", {"tool": tool_name, "input": tool_input, "output": tool_output})
await archive.record("error", {"message": error_message, "context": context})
```

**Why:** The archive is your fallback when structured retrieval fails. It also future-proofs your data — when better models ship, they can reingest the raw transcripts and extract richer structure than what's possible today. You are never starting from scratch.

### 2. Extract and store facts

After each user message, extract facts and store them in the knowledge graph and vector memory:

```python
from taosmd import KnowledgeGraph, VectorMemory
from taosmd.memory_extractor import process_conversation_turn

kg = KnowledgeGraph(os.path.expanduser("~/.taosmd/knowledge-graph.db"))
vmem = VectorMemory(os.path.expanduser("~/.taosmd/vector-memory.db"), embed_mode="onnx", onnx_path="<taosmd-dir>/models/minilm-onnx")
await kg.init()
await vmem.init()

# This runs regex extraction (fast, 15ms) and optionally LLM extraction (slower, richer)
await process_conversation_turn(
    user_message,
    agent_name="assistant",
    kg=kg,
    archive=archive,
    vmem=vmem
)
```

For explicit facts you learn, store them directly:

```python
await kg.add_triple("User", "prefers", "dark mode")
await kg.add_triple("User", "works on", "taOS project")
await vmem.add("User is building taOS, a personal AI operating system on Orange Pi hardware")
```

### 3. Search before answering

Before answering any question about the user's history, preferences, or prior work:

```python
# Fast path: structured retrieval
facts = await kg.query_entity("User")
kw_results = await archive.search_fts("the topic")
semantic_results = await vmem.search("the question", hybrid=True)

# If results are thin or you're uncertain, also check session crystals
from taosmd import CrystalStore
crystals = CrystalStore(os.path.expanduser("~/.taosmd/crystals.db"))
await crystals.init()
recent = await crystals.recent(limit=10)
crystal_hits = await crystals.search("the topic")
```

If structured retrieval returns nothing useful and you are still uncertain, say so explicitly and offer to search the full archive:

```python
archive_hits = await archive.search_fts("broader search terms")
```

**Do not guess.** If the archive doesn't have it, it didn't happen in a recorded session.

---

## Session Boundaries

At the end of every session, crystallise it:

```python
import os
from taosmd import CrystalStore

crystals = CrystalStore(os.path.expanduser("~/.taosmd/crystals.db"))
await crystals.init()

crystal = await crystals.crystallize(
    session_id=session_id,
    turns=session_turns,          # list of {role, content, timestamp} dicts
    agent_name="assistant",
    llm_url="http://localhost:11434",
    model="qwen3:4b",
    kg=kg                          # lessons get written back to the KG
)
```

The crystal stores a narrative summary, outcomes, and lessons learned. Lessons are automatically added to the KG as `agent learned <lesson>` triples.

---

## Organising Sessions into Projects

taOSmd's Session Catalogue automatically derives a timeline directory from the archive. It classifies each session by topic and links related sessions:

```python
import os
from taosmd import CatalogPipeline

d = os.path.expanduser("~/.taosmd")
pipeline = CatalogPipeline(
    archive_dir=os.path.join(d, "archive"),
    sessions_dir=os.path.join(d, "sessions"),
    catalog_db=os.path.join(d, "session-catalog.db"),
    crystals_db=os.path.join(d, "crystals.db"),
    kg_db=os.path.join(d, "knowledge-graph.db"),
    llm_url="http://localhost:11434",
)
await pipeline.init()

# The pipeline indexes archive days (split, enrich, crystallize), not live turns:
await pipeline.index_yesterday()                      # what the nightly cron runs
await pipeline.index_day("2026-07-01", force=True)    # re-index one day
await pipeline.index_range("2026-06-01", "2026-06-30")
await pipeline.rebuild()                              # full re-index of the archive

# The catalogue itself is pipeline.catalog (a SessionCatalog)
sessions = await pipeline.catalog.search_topic("the topic")

await pipeline.close()
```

The catalogue is what the shipped Librarian layer uses to route uncertain queries. Keep it current.

---

## When You Are Uncertain

If you can't find something in structured retrieval (KG + vector + crystals), follow this escalation:

1. **Archive FTS** — full-text search the raw archive: `await archive.search_fts("query")`
2. **Broader FTS** — try related terms, adjacent dates, or entity variants
3. **Session catalogue** — check `await pipeline.catalog.search_topic("topic")` for relevant session IDs, then retrieve those sessions from the archive directly
4. **Acknowledge the gap** — if nothing is found, tell the user clearly: "I don't have a record of this from our sessions." Do not guess or fabricate.

Never return a fabricated answer when retrieval fails. The archive is complete for everything that was recorded — if it's not there, it wasn't captured.

---

## Transcript Capture for Standalone Frameworks

If you're running in a framework that doesn't automatically expose transcript access (OpenClaw, Hermes, CrewAI, LangChain, etc.), set up transcript capture manually:

### OpenClaw / Claude Code (CLAUDE.md)

Add to your `CLAUDE.md` or `AGENTS.md`:

```
After every response, archive the conversation turn to taOSmd:
  python3 -c "
import asyncio, os, sys, time
from taosmd import Archive
async def record():
    a = Archive(os.path.expanduser('~/.taosmd/archive'), os.path.expanduser('~/.taosmd/archive-index.db'))
    await a.init()
    await a.record('conversation', {'role': sys.argv[1], 'content': sys.argv[2], 'timestamp': time.time()}, summary=sys.argv[2][:80])
asyncio.run(record())
" "$ROLE" "$CONTENT"
```

For production use, integrate the Python archive calls directly into your agent's response handler rather than shelling out.

### LangChain / LangGraph

```python
from taosmd import Archive
import asyncio, os, time

archive = Archive(os.path.expanduser("~/.taosmd/archive"), os.path.expanduser("~/.taosmd/archive-index.db"))
asyncio.run(archive.init())

class TaOSmdCallbackHandler(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs):
        content = response.generations[0][0].text
        asyncio.run(archive.record(
            "conversation",
            {"role": "assistant", "content": content, "timestamp": time.time()},
            summary=content[:80]
        ))

    def on_human_message(self, message, **kwargs):
        asyncio.run(archive.record(
            "conversation",
            {"role": "user", "content": message, "timestamp": time.time()},
            summary=message[:80]
        ))
```

### Hermes / OpenAI-compatible APIs

Wrap the API call:

```python
import httpx, asyncio, os, time
from taosmd import Archive

archive = Archive(os.path.expanduser("~/.taosmd/archive"), os.path.expanduser("~/.taosmd/archive-index.db"))

async def chat_with_memory(messages: list[dict]) -> str:
    await archive.init()

    # Archive the user turn
    user_msg = messages[-1]["content"]
    await archive.record("conversation", {"role": "user", "content": user_msg, "timestamp": time.time()}, summary=user_msg[:80])

    # Call the model
    async with httpx.AsyncClient() as client:
        resp = await client.post("http://localhost:11434/api/chat", json={"model": "qwen3:4b", "messages": messages})
    reply = resp.json()["message"]["content"]

    # Archive the response
    await archive.record("conversation", {"role": "assistant", "content": reply, "timestamp": time.time()}, summary=reply[:80])

    return reply
```

---

## Data Location

All data is stored in `~/.taosmd/` by default:

| File | Contents |
|------|---------|
| `knowledge-graph.db` | Temporal fact triples |
| `vector-memory.db` | Embedded text chunks |
| `archive/` | Raw JSONL conversation logs |
| `archive-index.db` | FTS5 index over the archive |
| `session-catalog.db` | Session catalogue with topic classification |
| `crystals.db` | Session digests and lessons |

The archive directory contains one `.jsonl.gz` file per day. The nightly cron installed by setup (3 AM) compresses old archive files (`compress_old_files()`) and runs `CatalogPipeline.index_yesterday()` to catalogue, crystallize, and update the knowledge graph for the previous day. Never delete these files — they are the source of truth.

---

## GPU Worker (Optional)

A GPU machine (e.g. x86 + NVIDIA) gives ~10x speed on LLM extraction. Not required — the Pi or any CPU handles everything:

```bash
# On the GPU machine
ollama pull qwen3:4b
```

Then point taOSmd at it by passing the URL where you construct the LLM-backed pieces (there is no `TAOSMD_LLM_URL` environment variable):

```python
pipeline = CatalogPipeline(..., llm_url="http://<gpu-machine>:11434")
crystal = await crystals.crystallize(..., llm_url="http://<gpu-machine>:11434")
```

---

## Summary

| Must do | When |
|---------|------|
| `archive.record(...)` for user message | Every turn |
| `archive.record(...)` for your response | Every turn |
| `process_conversation_turn(...)` | Every user message |
| `crystals.crystallize(...)` | End of session |
| `pipeline.index_yesterday()` (or the nightly cron) | Daily |
| Search before answering history questions | On demand |
| Escalate to archive FTS when uncertain | On demand |
