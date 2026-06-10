# Task Graph: Design Spec

Status: PROPOSAL (joint taOS/taosmd rec, awaiting greenlight)
Date: 2026-06-10
Scope: New `taosmd/tasks.py` module, HTTP endpoints, CLI subcommands, and MCP tools.

---

## Problem

Multi-agent handoffs today have no shared, durable record of what work exists, what
is blocked, and what an agent should pick up next when it starts a new session. The
A2A bus moves messages between agents in real time; taosmd memory stores
conversation knowledge. Neither answers the question: "what tasks are open right now,
which ones are ready to run, and what context does a fresh agent session need to get
started fast?"

Without a dependency-aware task layer there are two failure modes. First, two agents
create duplicate work because there is no authoritative list of in-progress items.
Second, an agent that just joined the bus has to read back long conversation history
to reconstruct what to do next, burning tokens on archaeology rather than execution.

This spec defines a minimal task-graph component that closes both gaps: collision-free
task IDs safe for concurrent multi-agent creation, an edge table that expresses
blocking relationships, a ready-queue view that derives from those edges, and a prime
endpoint that produces a token-budgeted briefing for session bootstrap.

## Prior Art and Credit

While specifying this, we evaluated [beads](https://github.com/gastownhall/beads) by
gastownhall: a Go-based task/project CLI built on Dolt (a versioned SQL database).
beads is mature, MIT-licensed, and its core concepts are well-designed. Three ideas
from beads directly shaped this spec: the dependency graph as the source of truth for
readiness, the ready-queue as a derived view (not a separate queue), and the `bd prime`
command as a session-bootstrap briefing that summarises what needs doing right now.

The beads binary itself was rejected for integration. Dolt is too heavy for the Pi
tier (the primary deployment target) and would introduce a second daemon alongside
taosmd serve, which violates the lean-core principle. Instead, this spec takes the
concepts and builds an independent minimal implementation on taosmd's existing SQLite
substrate.

Credit line: "Concept credit: the dependency-graph, ready-queue, and prime ideas
come from beads (github.com/gastownhall/beads); this is an independent minimal
implementation for the taosmd substrate."

## Scope and Non-goals

This component covers task-graph storage, a ready-queue view, and a prime briefing
endpoint. It is explicitly not:

- A messaging layer. The A2A bus exists for that.
- A memory or knowledge layer. taosmd memory remains the source of truth for
  conversation knowledge and retrieved facts.
- A git-like branching database. No Dolt-style versioning. A single live server
  is the target; archive replay covers rebuild if needed.
- A web UI. A dashboard surface may come later; v1 is API, CLI, and MCP only.
- A memory-decay system for tasks. The memory layer owns knowledge expiry;
  tasks in this layer are archived, not decayed.

The A2A bus and taosmd memory are unchanged by this component.

## Core Design Principle: Archive-Backed Projection

Tasks follow the same zero-loss principle that governs taosmd memory. Every mutation
(create, status change, edge add, edge remove, assign) is appended to the archive as
an event with `event_type = "task"`. The SQLite tables described below are a queryable
projection built from those events.

This has two practical consequences. First, the full audit trail is free: you can see
who created, blocked, or closed any task and when. Second, if the projection tables
are ever lost or corrupted, replaying the archive events rebuilds them exactly. The
future consolidation engine will also be able to learn from task history without any
special instrumentation.

## Schema

The schema lives in a new module `taosmd/tasks.py` and is created via the existing
`_db` migration pattern used throughout the codebase.

### Table: `tasks`

```
id          TEXT  PRIMARY KEY
title       TEXT  NOT NULL
body        TEXT
status      TEXT  CHECK(status IN ('open','in_progress','blocked','closed','superseded'))
                  DEFAULT 'open'
assignee    TEXT  NULL    -- agent handle or canonical_id
project     TEXT  NULL    -- git-derived project id, same vocabulary as memory shelves
priority    INTEGER DEFAULT 0
created_ts  REAL
updated_ts  REAL
closed_ts   REAL  NULL
created_by  TEXT
metadata    TEXT          -- JSON blob for extension fields
```

The `id` is a content hash: `"t-" + sha256(title | project | created_ts)[:12]`. Using
a hash derived from content rather than a server-side sequence means two agents that
concurrently create tasks with different titles will never collide, even if they call
the endpoint within the same millisecond. The `created_ts` component ensures that
re-creating a superseded task with the same title yields a new ID.

The `project` field uses the same vocabulary as memory shelves, aligning tasks with
the git-derived project-id that the taOS grants contract (issue #744) will verify.
When that contract lands, project scoping on tasks will bind to the verified claim
without any schema change.

### Table: `task_edges`

```
from_id     TEXT  REFERENCES tasks(id)
to_id       TEXT  REFERENCES tasks(id)
type        TEXT  CHECK(type IN ('blocks','parent','relates','duplicates'))
created_ts  REAL
created_by  TEXT
removed_ts  REAL  NULL
```

Edges are never deleted. Removing an edge records a `removed_ts` timestamp instead.
This means edge history is preserved in the projection table and is also replayed from
the archive. The four edge types cover the common cases: `blocks` for hard dependency,
`parent` for hierarchy, `relates` for soft reference, and `duplicates` for dedup
management.

### View: `ready_tasks`

```sql
CREATE VIEW ready_tasks AS
SELECT t.*
FROM tasks t
WHERE t.status = 'open'
  AND NOT EXISTS (
    SELECT 1 FROM task_edges e
    JOIN tasks blocker ON blocker.id = e.from_id
    WHERE e.to_id = t.id
      AND e.type = 'blocks'
      AND e.removed_ts IS NULL
      AND blocker.status NOT IN ('closed', 'superseded')
  )
ORDER BY t.priority DESC, t.created_ts ASC;
```

The ready queue is a SQL view, not a separate queue or polling mechanism. A task is
ready when it is open and has no active (non-removed) blocking edge whose source task
is still live (not closed or superseded). The ordering puts high-priority tasks first,
breaking ties by creation time ascending (oldest first). This view is queried directly
by the `/tasks/ready` endpoint and by `taosmd tasks ready`.

## HTTP Endpoints

All endpoints follow the same style as the existing endpoint table in
`taosmd/http_server.py`: stdlib HTTP server, same token gate, same JSON response
envelope. Project scoping will bind to the verified `project_id` claim from the taOS
grants contract once that is available.

| Method | Path | Body / Query | Response |
|--------|------|--------------|----------|
| POST | `/tasks` | `{title, body?, project?, assignee?, priority?, depends_on?: [task_id]}` | task object |
| GET | `/tasks` | `?status=&project=&assignee=&limit=` | task list |
| GET | `/tasks/ready` | `?project=&assignee=&limit=` | ready-queue ordered list |
| GET | `/tasks/prime` | `?project=&assignee=` | prime briefing object |
| POST | `/tasks/{id}` | `{status?, assignee?, priority?, body?}` | updated task object |
| POST | `/tasks/{id}/edges` | `{to_id, type}` | edge record |
| POST | `/tasks/{id}/edges/remove` | `{to_id, type}` | edge record with removed_ts |

### Prime Briefing

`GET /tasks/prime` returns a JSON object with two keys:

- `"text"`: a plain-text briefing targeting roughly 1-2k tokens, suitable for direct
  injection into an agent's system prompt or first user message as session context.
- `"tasks"`: the raw task objects referenced in the briefing.

The briefing covers four sections in order: ready tasks (top N by priority), tasks
currently in progress (with their assignees), blocked tasks (with the blocking task
title as the reason), and recently closed tasks (last 24 hours by default). The token
budget is enforced by counting approximate tokens (word count * 1.3) and truncating
lower-priority sections before ready tasks.

This is the beads `bd prime` concept applied to the taosmd context. An agent that
calls `/tasks/prime` at session start gets a complete picture of the task graph state
without scanning message history. taOS will wire this into the handoff bootstrap
flow; Claude Code sessions can call it via MCP.

The STATUS.md on-arrival checklist will gain `taosmd tasks ready` as the canonical
what-next query once this ships.

## CLI

New subcommand group under `taosmd tasks`:

```
taosmd tasks add    --title TEXT [--body TEXT] [--project ID] [--assignee HANDLE]
                    [--priority INT] [--depends-on TASK_ID...]
taosmd tasks list   [--status STATUS] [--project ID] [--assignee HANDLE] [--limit N]
taosmd tasks ready  [--project ID] [--assignee HANDLE] [--limit N]
taosmd tasks prime  [--project ID] [--assignee HANDLE]
taosmd tasks start  TASK_ID [--assignee HANDLE]
taosmd tasks close  TASK_ID
taosmd tasks block  TASK_ID --blocked-by TASK_ID
```

`start` is a convenience wrapper for `POST /tasks/{id}` with `status=in_progress`.
`close` sets `status=closed`. `block` creates a `blocks` edge from the blocking task
to the target task.

## MCP Tools

Four MCP tools exposed through the existing MCP server surface:

- `task_add`: create a task (mirrors POST /tasks)
- `task_ready`: list the ready queue (mirrors GET /tasks/ready)
- `task_prime`: fetch the prime briefing (mirrors GET /tasks/prime)
- `task_update`: update status, assignee, priority, or body (mirrors POST /tasks/{id})
- `task_edge`: add or remove an edge (mirrors POST /tasks/{id}/edges and /edges/remove)

These give Claude Code sessions and any MCP-connected agent direct access to the task
graph without going through the HTTP API explicitly.

## Consumers

**taOS Tasks app and handoff bootstrap**: taOS will wire `/tasks/prime` into agent
session startup and provide a task management UI. That integration is being built on
the taOS side (@taOS is coordinating).

**Claude Code sessions via MCP or CLI**: Any Claude Code session working on the
taosmd codebase can call `taosmd tasks ready` to find the next open task and
`taosmd tasks prime` to get the full session briefing.

**STATUS.md**: Once shipped, the on-arrival checklist gains `taosmd tasks ready` as
the canonical first command for any agent resuming work on the project.

## Testing

The test plan follows the existing `live_server` pattern for HTTP surface tests and
standalone unit tests for pure logic.

**Hash ID stability**: given the same `(title, project, created_ts)` triple, the ID
must be identical across two calls. Given any distinct triple, the IDs must differ.
This covers the concurrent multi-agent collision-free guarantee.

**Ready-view correctness**: unit tests covering (a) a task with no edges is ready,
(b) a task blocked by an open task is not ready, (c) a task blocked only by closed
tasks is ready, (d) a transitive chain A blocks B blocks C where A is open means B
and C are both not ready, (e) removing an edge (setting removed_ts) restores
readiness of the downstream task.

**Prime token budget**: the briefing text must stay within the configured token ceiling
under a range of task counts; truncation must drop lower-priority sections before
ready tasks; the `"tasks"` array must include only tasks referenced in the text.

**Archive event round-trip**: create a set of tasks and edges, capture the live table
state, delete the projection tables, replay the archive events, and assert the
rebuilt tables diff-equal the captured state. This validates the archive-backed
projection guarantee end to end.

**HTTP surface**: using the existing `live_server` fixture, test each endpoint for
correct status codes, response shape, token-gate rejection, and project scoping.

## Rollout

This component is purely additive. It introduces new tables, new endpoints, new CLI
subcommands, and new MCP tools. No existing interface changes. It ships as one PR
containing the module, tests, doc updates, and a CHANGELOG entry. No feature flag is
needed.

The PR will include:
- `taosmd/tasks.py` with schema, migration, and business logic
- HTTP endpoint registrations in `taosmd/http_server.py`
- CLI subcommand group in the existing CLI entrypoint
- MCP tool registrations
- Unit and integration tests
- This spec linked from `docs/superpowers/specs/`
- CHANGELOG entry under a new "Task Graph" section
