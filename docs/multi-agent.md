# Running multiple agents on one taosmd install

The taosmd service is one process. Per-agent indexes are separate. Two agents share embedding compute, the LLM, the cross-encoder, and the catalog runtime — but each one has its own shelf, its own knowledge graph, its own archive. Reading another agent's memory is opt-in and explicit.

If you've got one OpenClaw process running five agents, or one LangChain harness running a researcher and a support bot, this is the doc for you.

## The shape

```
data/
├── agents.json                              ← registry: who's who
└── agent-memory/
    ├── alice/
    │   └── index.sqlite                     ← alice's shelf
    ├── bob/
    │   └── index.sqlite                     ← bob's shelf
    └── support/
        └── index.sqlite                     ← support's shelf
```

One taosmd process holds the embedding model, the LLM client, and the catalog pipeline in memory. When `alice.search()` runs, it routes to `agent-memory/alice/index.sqlite`. When `bob.search()` runs, it routes to `agent-memory/bob/index.sqlite`. The compute is shared; the data isn't.

## Naming convention

Agent names must match `^[a-z][a-z0-9_-]{0,62}$` — lowercase, start with a letter, dashes and underscores allowed, max 63 chars. Pick stable names. They become directory names on disk and they're cited in every search/ingest call.

The convention I'd recommend: `{framework}-{role}` or `{tenant}-{role}`.

| Framework / context | Suggested name |
|---|---|
| Standalone OpenClaw research agent | `openclaw-research` |
| OpenClaw support bot | `openclaw-support` |
| Two LangChain agents on the same harness | `langchain-writer`, `langchain-editor` |
| Multi-tenant SaaS, customer ACME | `acme-assistant` |
| Personal Claude Code project | your handle, e.g. `jay` |

Avoid names that clash with reserved-feeling defaults (`default`, `agent`, `user`) — those are easy to type by accident in another script and end up sharing.

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

By default agents only see their own shelf. If you need one agent to read another's memory — typically a supervisor reading what its workers logged — call search with an explicit `also_include` list:

```python
# Supervisor reads its own memory PLUS the worker shelves it owns.
results = taosmd.search(
    query,
    agent="supervisor",
    also_include=["worker-1", "worker-2"],  # explicit names, no wildcards
)
```

Rules:

- The `also_include` list is **explicit names only**. No wildcards, no globs, no "all". You name every shelf you want to read.
- The supervisor's own writes still go only to its own shelf. `ingest()` ignores `also_include`.
- Cross-agent reads work for `search()`, `vsearch()`, `hybrid_search()`, and `ContextAssembler.assemble()`. They don't apply to `KnowledgeGraph.add_triple()` or any other write path.
- The `source` field on returned passages always names the shelf it came from, so you can attribute "I learned this from worker-1" in your output.

If you want bidirectional sharing (worker-1 also reads supervisor's memory), each agent declares the other in its own `also_include`. There's no group concept — pairwise opt-in keeps the model simple and the access surface narrow.

## Migration scenarios

### Renaming an agent

```bash
# 1. Rename the registry record + the directory.
mv data/agent-memory/old-name data/agent-memory/new-name
taosmd agent rm old-name           # removes the registry entry, preserves data
taosmd agent add new-name          # registers the new name against the moved dir

# 2. Update the agent's instruction file: replace the agent name in
# the per-turn rules block. Without this the agent will keep calling
# search(agent="old-name"), which auto-registers a fresh empty shelf.
```

### Splitting one agent into two

You've been running everything as `jay` and want to split personal life out into `jay-personal`. There's no built-in mover — write a one-off script:

```python
import taosmd
from taosmd.archive import ArchiveStore

src = ArchiveStore("data/agent-memory/jay/index.sqlite")
dst = ArchiveStore("data/agent-memory/jay-personal/index.sqlite")
taosmd.register_agent("jay-personal")

for record in src.iter_archive():
    if record.metadata.get("topic") == "personal":   # your filter
        dst.append(record.text, metadata=record.metadata)
        src.delete(record.id)

# Re-index the new agent so vector + KG stay in sync with the moved archive.
# (Run your normal ingest flow against jay-personal.)
```

### Merging two agents

Reverse of the split. Append the source's archive into the destination, then re-index. Delete the source registry entry once verified.

## Resource isolation

The job queue is single-process. All agents queue against the same embedding service. There are no per-agent rate limits today — a chatty agent can saturate the queue and slow down its peers. If that becomes a problem, the right fix is a per-agent token budget at the queue layer, tracked under #2-style follow-ups.

Disk: per-agent dirs grow independently. Run `taosmd agent list` to see chunk counts and spot the agents that are running away.

Backups: back up the entire `data/` tree as one unit. Per-agent dirs reference each other through the catalog and crystals stores; partial backups risk dangling references.

## Worked example — single OpenClaw, five agents

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
# openclaw-research     Research             2026-04-14 15:00   —                 0
# openclaw-writer       Writer               2026-04-14 15:00   —                 0
# ...
```

In each agent's instruction file (OpenClaw's per-agent system prompt), append the rules block from [`docs/agent-rules.md`](agent-rules.md), with `<your-agent-name>` replaced by that agent's name. The writer's prompt cites `openclaw-writer`, the support bot's prompt cites `openclaw-support`, etc.

Optional cross-reads: if the writer should be able to see what the researcher logged, add `also_include=["openclaw-research"]` to the writer's search calls (or wrap the search call in OpenClaw's tool layer so the agent doesn't have to remember).

That's the whole setup. Five shelves, one taosmd process, five differently-instructed agents that never accidentally share a memory.
