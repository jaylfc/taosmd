# Coding-agent capture hooks

Status: approved for build (Jay, 2026-10-02). Phase 1 is Claude Code only.

## Problem

Zero-loss is only as good as capture. Today a coding agent gets into the archive only if
it follows AGENTS.md and shells out to `archive.record` on every turn. Agents skip turns,
forget after a compaction, or never read the file. Nothing in the repo hooks the harness
itself: `SessionStart`, `UserPromptSubmit`, `PreCompact` and `"hooks"` have no hits in
`taosmd/`, and the only installer is `taosmd install-skill` (copies the A2A skill).

Most memory systems in the field now capture through harness hooks rather than agent
compliance (Hindsight, agentmemory, letta-code, gbrain, funes and others). This spec
does the same, built on pieces we already have.

## Goals

1. Every user prompt and assistant reply in a hooked session lands in the archive with no
   cooperation from the model.
2. Capture is idempotent and catches up: a missed or failed hook loses nothing, the next
   hook firing (or a manual `taosmd hooks sync`) backfills it.
3. A hook never blocks or breaks the agent. It fails open and stays fast.
4. A new session starts with a short, relevant briefing injected as context.
5. Works local-only (in-process) and against `taosmd serve` (remote), with no new
   dependencies.

Non-goals for phase 1: Codex and Cursor (phase 2), per-prompt retrieval injection on by
default, procedural memory (gap #7, which builds on this).

## Design

### Capture source: the transcript, not the hook payload

Claude Code passes every hook a JSON object on stdin that includes `session_id`,
`transcript_path`, `cwd` and `hook_event_name`. The transcript is a JSONL file the harness
appends to. Capture reads the transcript from a saved cursor instead of trusting any one
hook's payload. That is what makes it lossless: whichever hook fires next picks up
everything since the last successful read.

- Cursor state: one row per `session_id` (`transcript_path`, `cwd`, project id, byte
  offset, last entry id, updated_at; `cwd` from the hook payload; project id computed with `taosmd.project.get_project_id(cwd=cwd)` and stored, so `hooks sync --all` can scope a backfill without a live hook) in a small SQLite file under the data dir (`capture-cursors.db`, opened via
  `taosmd._db.connect`).
- Read from the offset to the last complete line only. A trailing line without `\n` is
  being written; leave it for next time.
- If the file is now shorter than the offset (rewritten or truncated), re-read from 0.
  Dedup (below) makes that safe.
- Keep user and assistant message entries. Tool calls and tool results are kept too, in
  phase 1 as ordinary `ingest_batch` items (which `taosmd.api.ingest_batch` always records
  as archive event type `conversation`; there is no event-type parameter and phase 1 does
  not add one), marked by `metadata.kind` = `"tool_use"` or `"tool_result"`, with content
  capped (configurable, default 8 KB per
  field, the cap recorded in metadata as `truncated: true`) so a huge file read does not
  bloat the archive. Other entry types (summaries, system, attachments metadata) are
  skipped and counted.
- The implementer MUST confirm the transcript entry shape against a real current Claude
  Code transcript and commit a scrubbed fixture of it. The parser keys off fields seen in
  that fixture, not off this document.

### Write path: `ingest_batch` with stable ids

Each kept entry becomes one `ingest_batch` item, a dict `{"text", "id", "metadata"}`
(`text` is required and non-empty; an entry that renders to empty text is skipped and
counted). `metadata.kind` is `"message"`, `"tool_use"` or `"tool_result"`. Its `id` is
`claude-code:<session_id>:<entry uuid>` (fall back to a sha256 of the line's byte offset in the transcript plus the raw line if an entry
has no uuid). `ingest_batch` skips ids it has stored (#25 contract), so replays, truncation re-reads
and overlapping hooks do not duplicate except as noted below. Caveat, measured at `taosmd/api.py`: the stored-id
set is read from the VECTOR store (`existing_source_ids`), so an item whose vector write
failed has an archive row but an unseen id. The batch then returns `degraded: true` and
`vector_failures`. Rule: a returned result, degraded or not, ADVANCES the cursor (the
archive rows exist and `reconcile()` re-embeds them); only a raised exception or a timeout
holds it. A timeout can strand at most the in-flight item (archive row written, vector row not);
the retry re-writes that one archive row. Acceptance counts distinct source_id, so this is tolerated;
`taosmd hooks sync` runs `reconcile()` after a sync that timed out, and batches are chunked so a timeout is rare. Never retry a degraded batch, or its archive rows duplicate. Metadata carries `source:
"hook:claude-code"`, `session_id`, `role`, `cwd`, entry timestamp and `transcript_path`.

- Agent name: `claude-code` by default, overridable in the hook config.
- Project: `taosmd.project.get_project_id(cwd=cwd)`, so sessions in the same repo share
  project-scoped memory.
- Local vs remote: call `taosmd.service.ingest_batch`, which already forwards to the
  remote client when a server URL is configured (`taosmd config set-server`) and calls
  `taosmd.api.ingest_batch` in-process otherwise. Do not re-implement that switch.
- Secrets: the archive already runs `redact_secrets` on record. Nothing extra here.

### Never block the agent

- Every hook command wraps its work in a hard timeout (default 5 s for capture, 3 s for
  injection) and exits 0 on any error, logging to `<data dir>/logs/hooks.log`.
- If the write raises (server down, database locked) or times out, the cursor does NOT
  advance, so the next firing retries. A degraded result is not a failure (see above). There is no separate spool: the transcript is the spool.
- Capture hooks print nothing to stdout, so they cannot inject noise into context.

### Which hooks

| Hook | Action |
|------|--------|
| `SessionStart` | sync, then print the briefing (below) |
| `Stop` | sync |
| `PreCompact` | sync (the last chance before the context is summarised) |
| `SessionEnd` | sync |
| `UserPromptSubmit` | sync; inject retrieval only if enabled |

`taosmd hooks sync [--session ID | --all]` runs the same code by hand, and `--all` walks
every cursor row so a crashed session can be backfilled.

### Injection

- `SessionStart` prints a briefing capped at a token budget (default 1500 tokens, counted
  approximately at 4 chars per token): open tasks from the existing `task_prime` path, the
  last N crystals for this project, and a one-line note that capture is automatic. If
  nothing is found it prints nothing.
- `UserPromptSubmit` retrieval injection is OFF by default. When enabled, it runs
  `search()` on the prompt scoped to the project, keeps hits above a score floor, and
  prints them under the budget. Off by default because embedding on every prompt costs
  latency on a Pi, and bad injection is worse than none.
- Injected text is framed as retrieved memory, not instructions ("Possibly relevant notes
  from earlier sessions, verify before relying on them"). This is the minimum guard until
  write-time quarantine (gap #6) exists.

### Installer

`taosmd hooks install --agent claude-code [--scope user|project] [--inject-prompts]`

- Writes the hook entries into `~/.claude/settings.json` (user) or
  `.claude/settings.local.json` (project; the local file, because the hook command holds
  an absolute path that is machine-specific and must not land in a committed
  `.claude/settings.json`). It merges: existing hooks, including other tools'
  entries for the same events, are kept. Our entries are recognisable by their command
  (`taosmd hooks run ...`) so install is idempotent and uninstall removes only ours.
- Before any write, copy the settings file to `settings.json.taosmd-bak-<UTC ts>`. Never
  delete a user's file.
- Refuse to write if the existing file is not valid JSON; print the parse error.
- `taosmd hooks uninstall --agent claude-code [--scope ...]` and `taosmd hooks status`
  (installed where, cursor rows, last sync time, last error from the log, count of degraded batches since the last `reconcile()`).
- The hook command must resolve `taosmd` by absolute path at install time (the hook runs
  in a non-interactive shell where a venv may not be on PATH).
- Each installed hook entry sets an explicit `timeout` (seconds) matching its budget: 5 for capture hooks, 3 for injection.

### AGENTS.md

When hooks are installed, manual `archive.record` calls are redundant. Update AGENTS.md
and the README to say: install the hooks if your harness supports them, and only fall
back to manual recording if it does not. Dedup by id means a session doing both does not
double-write hook-captured turns, but manual records carry no entry id, so they are
duplicates in content. The docs must say this plainly.

## Phase 2 (separate cards, not now)

- Codex: transcript files under `~/.codex/sessions/`. Same cursor + `ingest_batch`
  design, driven by Codex's notify hook or a `taosmd hooks sync --agent codex` timer.
- Cursor: its hooks API. Confirm the event names against the current Cursor docs first.
- A generic `taosmd hooks sync --jsonl PATH --format <name>` for any harness that writes
  transcripts.

## Tests (phase 1)

All with `tmp_path`, no network, no models (use the existing test embedder/fakes):

1. Fixture transcript in, expected archive rows out (count and content), roles right.
2. Idempotency: sync twice, the row count does not change; truncate the file and sync,
   no duplicates.
3. Partial trailing line is not consumed; once completed, it is.
4. Write failure leaves the cursor where it was; the next sync writes the rows
   A degraded result (fake vector store returning -1) advances the cursor, and a second
   sync adds no archive rows.
5. Hook entry point exits 0 and prints nothing when the data dir is unwritable, and it
   returns within the timeout when the write hangs (simulate with a sleeping fake).
6. Installer: merges into a settings file that already has another tool's hooks for the
   same events and keeps them; second install is a no-op; uninstall removes only ours;
   a backup file is written; invalid JSON is refused.
7. Briefing respects the budget and prints nothing when there is nothing to say.

## Acceptance

A real Claude Code session with the hooks installed: after the session, the archive rows
tagged `hook:claude-code` for that `session_id`, counted by distinct `source_id`, equal
the entries the parser kept from its transcript (user and assistant messages plus tool
use and tool result entries; skipped types excluded), and the PR also shows the per-kind
split from `metadata.kind`, and a second session in the same repo starts with the briefing. Paste
both counts in the PR.
