# A2A delivery contract v2 — mechanical delivery, convergent alarms, wake economics

Status: DRAFT for review (Jay). Authored 2026-08-24 from the token-burn audit of
the week of 2026-08-16..23, during which the shared account pinned at 97-99% of
its weekly allowance within ~2 days of the reset, two weeks in a row.

## Why (measured, not argued)

- The bus has carried **3.7MB of message bodies in its entire life** (3,089
  messages). The two busiest agent sessions wrote **~120MB of transcript in
  three days** (Aug 16-18). The cost of A2A is therefore not bandwidth or
  message volume: **it is LLM wakes times context size**. Every optimisation
  below targets wakes, re-reads, or alarm noise, never bytes.
- Alarm precision measured on live data: at the taosmd watcher's frozen
  watermark, its rule counted **20 "unhandled" messages of which 4 were real**
  (80% false). The false 16 included the *other* watcher's alarms and the
  lead's acks - two dead-session watchers feeding each other's alarm
  condition. Convergence was structurally impossible.
- `GET /a2a/messages?since=<id>` (an id, not the required epoch-ts) was
  **silently tolerated** by the deployed bus and returned the last-N page, so
  every naive consumer re-read the whole window and fed it to a model. The
  400-guard exists on master but is not deployed (Pi is 139 commits behind).
- All delivery state lives client-side: **five independent watermark/filter
  stacks** (lead bash+python, taosmd python, taosc python, website python,
  fleet_health inline) each re-derive "addressed to me" and "unhandled",
  each with its own bugs. Server docs already concede the gap: "read receipts
  are not yet implemented (tsk-fhltad); unread_count omitted".

## Design law

A message is read into a model context **at most once**, and a wake happens
**only for information that changed**. Everything the bus can decide
mechanically, it decides server-side; a harness (Claude Code, opencode, kilo,
grok CLI, anything MCP) gets identical semantics because the intelligence is
in the bus, not the client stack.

## Contract

### 1. Message taxonomy at write time
`POST /a2a/send` gains `kind: chat | alarm | ack | digest | receipt | review |
system` (default `chat`). Machine traffic is labelled by its author at write
time, never inferred from body prefixes at read time (today three watchers
grep for "[AUTOMATED" - it works until one author changes a prefix).
Backfill: a one-shot migration tags historical messages by the same prefixes.

### 2. Server-side consumer cursors + inbox
- `GET /a2a/inbox?consumer=<principal>&limit=` returns only messages past the
  consumer's cursor that are **addressed to it** (mention of its handle, its
  owned threads, or a direct `to`), excluding its own posts and excluding
  `kind in (alarm, ack, receipt, digest)` unless `include_kinds=` says
  otherwise. Oldest-first, cursor does NOT advance on read.
- `POST /a2a/inbox/advance {consumer, to_id}` advances the cursor explicitly
  (the consumer decides when something is *handled*, not merely fetched).
- One implementation on the bus replaces five client stacks. Watermark files
  remain only as a transport-failure fallback.

### 3. Acks are state, not traffic
`POST /a2a/messages/{id}/ack {by}`. An ack today is a new bus message, which
other watchers then count as unhandled traffic (measured above). As state it
is queryable (`acked_by` on the message envelope) and generates no wake for
anyone. "Unhandled for X" becomes ONE server query - mentions of X past X's
cursor minus acks - instead of four divergent client definitions.

### 4. Alarms converge, server-enforced
Alarm-kind messages carry `alarm_key` (stable subject+condition string, e.g.
`dead-session:@taOSmd-dev`). The bus refuses to store a same-key alarm unless
its payload materially changed or the key was cleared (`POST
/a2a/alarms/{key}/clear`), answering `{deduped: true}` instead. Cooldowns
become bus policy (per-key min interval), deleting N per-watcher cooldown
files. An alarm that repeats identical information is the mechanism by which
alarms train their readers to ignore them; the bus makes that impossible
rather than discouraged.

### 5. Strict parameters everywhere
Unknown or invalid query parameters on every /a2a endpoint are a 400, never
silently ignored (master's since<1e9 rule generalised). Silent tolerance is
what converted a typo'd cursor into a full-page re-read fed to a model.

### 6. Wake payload carries the messages
The SSE stream and any push path deliver the filtered inbox entries
themselves (id, from, kind, body), so a woken session never issues a second
read for content it was woken FOR. Non-mention traffic batches into a
30-minute digest server-side (the client-side wake-diet of 2026-08-17,
promoted to bus policy so every harness inherits it).

## Deployment order
1. Deploy current master to the Pi first (already queued: 139 commits behind;
   carries the 400-guard and thread pagination).
2. Ship taxonomy (1) + strict params (5) - small, additive.
3. Inbox/cursor (2) + acks (3); migrate the three backup watchers and
   fleet_health onto `/a2a/inbox` and delete their filter stacks.
4. Alarm keys (4) + digest-in-bus (6); delete client digest machinery.

## Interim mitigations already landed (2026-08-24, this host)
- The three backup watchers now count only handle-mentions, exclude
  automation-class bodies, and refire only when the unhandled set changes
  (proven on live bus data: 20 claimed -> 4 real).
- `~/.taos-usage/watch.sh` judges usage windows per-window (an idle five_hour
  no longer blanks a valid seven_day - that defect blinded band governance
  for 32h on Aug 23-24) and retries once on the 8-hourly OAuth-refresh race.
- The six A2A cron lines on the lead host are disarmed pending this contract
  (`#A2A-HOLD-20260824#` markers in crontab).
