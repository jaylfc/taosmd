# Running multiple agents on one taosmd install

The taosmd service is one process. Agents share embedding compute, the LLM, the cross-encoder, and the catalog runtime, but each one has its own shelf, its own knowledge graph scope, and its own archive rows. Reading another agent's memory is opt-in and explicit.

If you've got one OpenClaw process running five agents, or one LangChain harness running a researcher and a support bot, this is the doc for you.

## The shape

```
data/
├── agents.json          ← registry: who's who (bookkeeping only)
├── agent-memory/
│   ├── alice/           ← registry dir for alice (bookkeeping + delete support)
│   └── bob/             ← registry dir for bob
├── vector-memory.db     ← shared vector store (all agents)
├── knowledge-graph.db   ← shared knowledge graph (all agents)
└── archive-index.db     ← shared archive index (all agents)
```

One taosmd process holds the embedding model, the LLM client, and the catalog pipeline in memory. There is **one shared set of stores** for the whole data directory: `vector-memory.db`, `knowledge-graph.db`, and `archive-index.db` are used by every agent. Per-agent isolation is enforced by an `agent` field stored in each row's metadata and filtered at query time, not by separate files.

The `data/agents.json` registry and `agent-memory/{name}/` directories exist for bookkeeping (agent list, display names, stats) and to support `taosmd agent rm <name> --drop-data` (the `delete_agent(name, drop_data=True)` API). They do not route reads or writes. Every search and ingest call goes to the same shared stores and is scoped by the `agent` parameter.

## Naming convention

Agent names must match `^[a-z][a-z0-9_-]{0,62}$`: lowercase, start with a letter, dashes and underscores allowed, max 63 chars. Pick stable names. They become directory names on disk and they're cited in every search/ingest call.

The convention I'd recommend: `{framework}-{role}` or `{tenant}-{role}`.

| Framework / context | Suggested name |
|---|---|
| Standalone OpenClaw research agent | `openclaw-research` |
| OpenClaw support bot | `openclaw-support` |
| Two LangChain agents on the same harness | `langchain-writer`, `langchain-editor` |
| Multi-tenant SaaS, customer ACME | `acme-assistant` |
| Personal Claude Code project | your handle, e.g. `jay` |

Avoid names that clash with reserved-feeling defaults (`default`, `agent`, `user`). Those are easy to type by accident in another script and end up sharing.

## Registering an agent

Explicit (recommended for any install with more than one agent):

```python
import taosmd

taosmd.register_agent("openclaw-research", display_name="Research Assistant")
```

Or via the CLI:

```bash
taosmd agent add openclaw-research --display-name "Research Assistant"
```

Lazy (back-compat for solo installs):

```python
# First call auto-registers with display_name == name. Fine for one
# agent; risky when you have several because typos become permanent
# new agents. Prefer explicit registration in multi-agent setups.
taosmd.search("hello", agent="openclaw-research")
```

A second `register_agent("openclaw-research", ...)` raises `AgentExistsError` unless you pass `clobber=True`. The install agent uses this to detect name clashes and prompt for a different name.

## Cross-agent reads (opt-in)

By default agents only see their own rows. If you need one agent to read another's memory, typically a supervisor reading what its workers logged, use `also_include`. Cross-agent reads require a shared `project` id: memories tagged with a different project (or tagged with no project at all, from a different agent) are excluded unless they match the calling agent's own rows.

```python
import taosmd

# Get the shared project fingerprint (auto-detected from git remote or cwd).
pid = taosmd.get_project_id()

# Supervisor reads its own memory PLUS the worker shelves, all within the same project.
results = await taosmd.search(
    query,
    agent="supervisor",
    project=pid,
    also_include=["worker-1", "worker-2"],  # explicit names, no wildcards
)
```

Rules:

- `also_include` is only honoured when `project` is also set. Without a `project`, the list is ignored and the search returns only the calling agent's own rows.
- The `also_include` list takes explicit names only. No wildcards, no globs, no "all". Name every shelf you want to read.
- The supervisor's own writes still go only to its own rows. `ingest()` ignores `also_include`.
- Cross-agent reads work for `search()`. They don't apply to `KnowledgeGraph.add_triple()` or any other write path.
- The `source` field on returned passages always names the shelf it came from, so you can attribute "I learned this from worker-1" in your output.

If you want bidirectional sharing (worker-1 also reads supervisor's memory), each agent declares the other in its own `also_include`. There is no group concept; pairwise opt-in keeps the model simple and the access surface narrow.

## Project-scoped memory

Projects let multiple agents working on the same codebase or task share memory without leaking it to unrelated agents.

```python
import taosmd

# Auto-detect a 12-character project fingerprint from the git remote origin.
# Falls back to .taosmd/project.toml, then a hash of the cwd.
pid = taosmd.get_project_id()

# Tag memories with a project when ingesting.
await taosmd.ingest(transcript, agent="supervisor", project=pid)

# Scope a search to that project.
hits = await taosmd.search(query, agent="supervisor", project=pid)

# Cross-agent reads within the project (see Cross-agent reads above).
hits = await taosmd.search(query, agent="supervisor", project=pid, also_include=["worker-1"])

# Discover which projects have stored memories.
projects = await taosmd.list_projects()

# List the agent shelves that have memories in a specific project.
shelves = await taosmd.list_shelves(project=pid)
```

These work across every surface. The Python API takes `project=`/`also_include=` on `ingest`/`search`; the HTTP server accepts `project` on `POST /ingest` and `project`/`also_include` on `/search`, plus `GET /projects` and `GET /shelves?project=`; the MCP server adds `project`/`also_include` to `memory_ingest`/`memory_search` and the `memory_list_projects`/`memory_list_shelves` tools; and the CLI has `taosmd projects` and `taosmd shelves --project <id>`.

## Migration scenarios

### Renaming an agent

Because per-agent data lives in shared stores (keyed by the `agent` field, not by directory), renaming is a two-step operation: update the registry, then update the agent name in the instruction file.

```bash
# 1. Update the registry.
taosmd agent rm old-name           # removes the registry entry (does not delete data)
taosmd agent add new-name          # registers the new name

# 2. Update the agent's instruction file: replace the agent name in
# the per-turn rules block. Without this the agent will keep calling
# search(agent="old-name"), which auto-registers a fresh empty shelf.
```

Note: existing rows in the shared stores still carry `agent="old-name"` in their metadata. A future `taosmd agent rename` command will handle the in-place metadata migration. For now, you can re-ingest the old agent's archive under the new name using the same split-style script above, or simply accept that old memories will only be found when searching with the old agent name.

### Splitting one agent into two

You've been running everything as `jay` and want to split personal life out into `jay-personal`. Because all agents share the same stores (isolation is by `agent` field, not by file), splitting is done by re-ingesting the relevant rows under the new agent name. There's no built-in mover; write a one-off script:

```python
import asyncio
from datetime import datetime, timezone
import taosmd
from taosmd.archive import ArchiveStore

async def split():
    taosmd.register_agent("jay-personal")
    # Open the shared archive store.
    store = ArchiveStore(
        archive_dir="data/archive",
        index_path="data/archive-index.db",
    )
    await store.init()

    # Locate the source agent's events through the index. query() filters on
    # agent_name / event_type / time window / FTS search and returns index
    # rows; raise limit (or page with offset) to cover the full history.
    rows = await store.query(agent_name="jay", event_type="conversation", limit=100000)
    days = sorted({
        datetime.fromtimestamp(r["timestamp"], tz=timezone.utc).strftime("%Y-%m-%d")
        for r in rows
    })

    # export_day() returns the full raw events for a day (it reads the JSONL,
    # compressed days included), so the re-ingest uses the verbatim text.
    for day in days:
        for event in await store.export_day(day):
            if event.get("agent_name") != "jay":
                continue
            content = event.get("data", {}).get("content", "")
            if "personal" in content:                    # your filter
                # Re-ingest under the new agent name to create tagged copies.
                await taosmd.ingest(content, agent="jay-personal")

asyncio.run(split())

# Re-index so vector + KG stay in sync with the new agent.
# (Run your normal ingest flow against jay-personal for any turns missed above.)
```

### Merging two agents

Reverse of the split. Append the source's archive into the destination, then re-index. Delete the source registry entry once verified.

## Resource isolation

The job queue is single-process. All agents queue against the same embedding service. There are no per-agent rate limits today; a chatty agent can saturate the queue and slow down its peers. If that becomes a problem, the right fix is a per-agent token budget at the queue layer, tracked under #2-style follow-ups.

Disk: per-agent dirs grow independently. Run `taosmd agent list` to see chunk counts and spot the agents that are running away.

Backups: back up the entire `data/` tree as one unit. The shared stores (`vector-memory.db`, `knowledge-graph.db`, `archive-index.db`) and the archive directory must stay consistent with each other; partial backups risk dangling references.

## Worked example: single OpenClaw, five agents

You're running one OpenClaw process. You want a research agent, a writing agent, a code-review agent, a customer-support agent, and a personal assistant. All five share the same Pi.

```bash
# 1. Install taosmd once.
curl -fsSL https://raw.githubusercontent.com/jaylfc/taosmd/master/scripts/setup.sh | bash

# 2. Register each agent.
taosmd agent add openclaw-research --display-name "Research"
taosmd agent add openclaw-writer   --display-name "Writer"
taosmd agent add openclaw-review   --display-name "Code Review"
taosmd agent add openclaw-support  --display-name "Support"
taosmd agent add openclaw-personal --display-name "Personal Assistant"

taosmd agent list
# NAME                  DISPLAY              CREATED            LAST INGEST  CHUNKS
# openclaw-research     Research             2026-04-14 15:00   (none)            0
# openclaw-writer       Writer               2026-04-14 15:00   (none)            0
# ...
```

In each agent's instruction file (OpenClaw's per-agent system prompt), append the rules block from [`taosmd/docs/agent-rules.md`](../taosmd/docs/agent-rules.md), with `<your-agent-name>` replaced by that agent's name. The writer's prompt cites `openclaw-writer`, the support bot's prompt cites `openclaw-support`, etc.

Optional cross-reads: if the writer should be able to see what the researcher logged, add `also_include=["openclaw-research"]` to the writer's search calls (or wrap the search call in OpenClaw's tool layer so the agent doesn't have to remember).

That's the whole setup. Five shelves, one taosmd process, five differently-instructed agents that never accidentally share a memory.
