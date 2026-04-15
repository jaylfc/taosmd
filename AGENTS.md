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

97.0% Recall@5 on LongMemEval-S (SOTA). Runs entirely locally, no cloud dependencies.

---

## Setup

### One-line install

```bash
curl -fsSL https://raw.githubusercontent.com/jaylfc/taosmd/master/scripts/setup.sh | bash
```

This installs the embedding model (90MB), pulls Qwen3-4B via Ollama (2.6GB), creates all data stores, and sets up a daily archive compression cron. On RK3588 (Orange Pi, Rock 5) it uses the NPU-optimised model instead.

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
from taosmd import Archive

archive = Archive(archive_dir="~/.taosmd/archive", index_path="~/.taosmd/archive-index.db")
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

kg = KnowledgeGraph("~/.taosmd/knowledge-graph.db")
vmem = VectorMemory("~/.taosmd/vector-memory.db", embed_mode="onnx", onnx_path="<taosmd-dir>/models/minilm-onnx")
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
crystals = CrystalStore("~/.taosmd/crystals.db")
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
from taosmd import CrystalStore

crystals = CrystalStore("~/.taosmd/crystals.db")
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
from taosmd import SessionCatalog, CatalogPipeline

catalog = SessionCatalog("~/.taosmd/catalog.db")
pipeline = CatalogPipeline(catalog, archive, llm_url="http://localhost:11434", model="qwen3:4b")
await catalog.init()

# Run after each session to keep the catalogue current
await pipeline.run(session_id=session_id, turns=session_turns)
```

The catalogue is what the Librarian layer (coming) uses to route uncertain queries. Keep it current.

---

## When You Are Uncertain

If you can't find something in structured retrieval (KG + vector + crystals), follow this escalation:

1. **Archive FTS** — full-text search the raw archive: `await archive.search_fts("query")`
2. **Broader FTS** — try related terms, adjacent dates, or entity variants
3. **Session catalogue** — check `await catalog.search("topic")` for relevant session IDs, then retrieve those sessions from the archive directly
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
import asyncio, sys, time
from taosmd import Archive
async def record():
    a = Archive('~/.taosmd/archive', '~/.taosmd/archive-index.db')
    await a.init()
    await a.record('conversation', {'role': sys.argv[1], 'content': sys.argv[2], 'timestamp': time.time()}, summary=sys.argv[2][:80])
asyncio.run(record())
" "$ROLE" "$CONTENT"
```

For production use, integrate the Python archive calls directly into your agent's response handler rather than shelling out.

### LangChain / LangGraph

```python
from taosmd import Archive
import asyncio, time

archive = Archive("~/.taosmd/archive", "~/.taosmd/archive-index.db")
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
import httpx, asyncio, time
from taosmd import Archive

archive = Archive("~/.taosmd/archive", "~/.taosmd/archive-index.db")

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
| `catalog.db` | Session catalogue with topic classification |
| `crystals.db` | Session digests and lessons |

The archive directory contains one `.jsonl.gz` file per day (compressed by the daily cron at 3 AM). Never delete these files — they are the source of truth.

---

## GPU Worker (Optional)

A GPU machine (e.g. x86 + NVIDIA) gives ~10x speed on LLM extraction. Not required — the Pi or any CPU handles everything:

```bash
# On the GPU machine
ollama pull qwen3:4b

# Point taOSmd at it
export TAOSMD_LLM_URL=http://<gpu-machine>:11434
```

---

## Summary

| Must do | When |
|---------|------|
| `archive.record(...)` for user message | Every turn |
| `archive.record(...)` for your response | Every turn |
| `process_conversation_turn(...)` | Every user message |
| `crystals.crystallize(...)` | End of session |
| `pipeline.run(...)` | End of session |
| Search before answering history questions | On demand |
| Escalate to archive FTS when uncertain | On demand |
