# Audit cron (re-arm every session)

Project rule: every working session arms an HOURLY monitor cron whose prompt is
"run `.claude/audit-cron-prompt.md`" (in Claude Code: **CronCreate**, schedule
`13 * * * *`). The :13 slot stays off the :00/:30 herd and away from the Pi sweep
at :15/:45. Session crons die with the session and auto-expire after 7 days,
which is why this file exists: the prompt is canonical here, not in anyone's
memory. (The original wording also passed `durable`; that flag is a documented
no-op, every cron is session-only, so do not rely on it.)

**The WHY for every rule below lives in `.claude/audit-cron-rationale.md`.** Read
that file before CHANGING a rule or when a guard fires and you want to know what
it caught. Do not simplify a rule here on the grounds that it looks arbitrary:
several were re-derived wrongly two or three times before landing.

Keep each run PROPORTIONATE: if nothing changed since the last run, say so in one
line and stop.

---

## STEP 0a - USAGE GATE (first)

Read the REAL shared account usage, never a local estimate (a local ccusage
estimate read 16 percent when the true shared figure was 78).

- **Source: `/home/jay/.taos-usage/current.json`**, written every 10 min by
  `~/.taos-usage/watch.sh` from the system crontab at minutes 6,16,26,36,46,56.
  Shape `{five_hour:{utilization,resets_at}, seven_day:{...}}`.
- **NEVER read `/home/jay/.taos-usage.json`** (the old Pi path) and never report
  its numbers even with a staleness caveat: a stale number quoted with a caveat
  still anchors a decision. If you want to retire this, name which sense you are
  falsifying (it is a CONSUMER-side claim; a producer-side fact cannot retire it).
- **Validity check 1 (mtime):** if `fetched_at` is older than 35 minutes, treat
  the file as dead and fall back to the direct API.
- **Validity check 2 (window), ALWAYS:** compare `resets_at` to `date -u`. If
  `resets_at` is in the PAST, the number describes a window that has ALREADY
  ROLLED and must not gate anything, no matter how recent the mtime is. File
  freshness is not window validity.
- **If you re-publish, gate on `~/.taos-team/usage_publish.sh` STDOUT ONLY. Never
  re-read `current.json` to confirm it.** That script does not write that file
  (in 39 lines it has exactly two writes: `printf > /tmp/usage.json` and an `scp`
  of that file to the Pi path; `current.json` is written by a DIFFERENT
  component, `~/.taos-usage/watch.sh --once`), so the natural re-read returns
  the IDENTICAL stale numbers and reads as independent confirmation of them.
  **This stdout rule is the correctness fix**; the resume-arm offset below is
  only a latency and legibility property.
- **Direct API fallback:** OAuth token from the Mac Keychain
  (`security find-generic-password -s "Claude Code-credentials" -w`, JSON field
  `.claudeAiOauth.accessToken`; on the Pi `~/.claude/.credentials.json`), then
  `GET https://api.anthropic.com/api/oauth/usage` with `Authorization: Bearer
  <token>` and header `anthropic-beta: oauth-2025-04-20`. **NEVER print or store
  the token.** **NEVER use the Pi-local `~/.claude/.credentials.json` as the
  fallback SOURCE** - it belongs to a different account and reports wrong numbers.
- The file is a **FLOOR**, not the truth: it lags active burn by up to half an
  hour. Before CITING a utilization number or gating mid-work, fetch the live
  figure; use the file only as the cheap first read.

### Bands (Jay, Jun 13; supersedes the old 70-90 finish-stage rule)

- **Below 90:** work normally. Do not finish-stage or hold early. Re-check at
  stage boundaries but only ACT on the bands below.
- **~90 = SOFT WIND-DOWN:** stop new work, run the handoff sweep, arm the resume
  pair, go quiet. All crons become MONITOR-ONLY: publish usage, skip
  non-essential sweep commits and retriggers, act only if essential.
- **98 = HARD STOP.**
- **`seven_day` crossing 90-95 before the weekly pause:** refresh the
  hand-holding job pack for Jay's weaker coding agents (1-2 years outdated
  knowledge, no design judgment) so work continues while strong sessions are
  dark. Format in `docs/agent-jobs/README.md`: absolute-rules README + repo facts
  they will not know + one fully spelled-out file per job (exact branch, commit
  message, PR title, allowed files, step-by-step with code, verification, STOP
  conditions). Contained low-risk jobs only; they PR to master and NEVER merge.
- **A HOLD SUSPENDS PRODUCING, NEVER RECEIVING.** At EVERY band including the 98
  hard stop, STEP 4's bus sweep, the heartbeat touch and the ADDRESSED drain
  STILL RUN. Only the producing steps (reviews, commits, merges, cards, probes)
  stop. **Do not collapse this pass to STEP 0a because usage is high** - that
  cost 15 hours of unread bus across the 2026-08-14/16 stop. If the band forbids
  acting on what the sweep finds, still READ it, record what is owed, and hand it
  to the next wake. A stop that cannot hear "do not stop" is a disconnection.

### Handoff-doc sweep (mandatory at EVERY wind-down and pause)

Bring the durable docs current before going quiet: (a) `STATUS.md` - Current
state, In flight, Queued next, Last updated stamp; (b) the `MEMORY.md` dev-status
index line + the `project_taosmd_dev_status.md` body; (c) `CHANGELOG.md`;
(d) `docs/research-report.md` per its skill if an experiment landed; (e)
README/AGENTS only if a user-facing surface or architecture fact changed. Commit
and push straight to master BEFORE silencing monitors. Never go dark with an
uncommitted tree or a stale STATUS.md.

### Resume pair protocol (every pause, no exceptions)

1. **Delete any still-pending resume/retry crons from earlier pauses first.**
2. **Create the RETRY one-shot FIRST**, at the RETRY CRON line printed by
   `python3 ~/.taos-team/resume_arm_time.py <resets_at>`. Its prompt starts by
   checking whether the primary already ran (tail of the conversation + the
   taosmd-progress channel) and exits in one line if so.
3. **Create the PRIMARY one-shot** at the primary time from **the SAME
   invocation**, embedding the retry cron id; its first step is to delete the
   retry now that the wake succeeded.
   - **Take BOTH cron lines from ONE invocation, never two**, or the pair's
     ordering rests on the crontab not changing in between.
   - **Never a constant.** +2, +7 and +11 were each tried and each was wrong in
     the same way; a flat retry beside a derived primary can fire FIRST, making
     the safety net the primary at a time derived from nothing. `resume_arm_time.py`
     is correct for any phase and any cadence and fails loud if the watcher line
     is absent.
4. **Alert the humans and the sibling BEFORE going quiet:** one `[WINDDOWN]` line
   to the general channel tagging @hermes (Jay's assistant, can reach him by
   phone) and the sibling agent, stating utilization, reset time, and the armed
   pair. The allowance is SHARED. Treat an incoming `[WINDDOWN]` as your own
   early warning and check the gate at your next stage boundary.
5. **LAST, silence the live monitors** (TaskStop the bus watcher and any
   result/log monitors): every monitor event wakes the session and burns tokens
   even when ignored, and everything they watch lands durably anyway. The PRIMARY
   re-arms them on wake per `a2a-session-setup.md` and reads the slept-through
   backlog.

## STEP 0a-bis - ARM-AT-START RESUME PAIR (every fire, idempotent)

A session that dies at a hard limit never reaches its wind-down, so the pair must
exist BEFORE that happens. Read the current window's `resets_at`; if no pair is
armed for THIS `resets_at`, arm one per the protocol above. Delete stale pairs
from prior windows. Both one-shots auto-delete on firing, so only an unfired
sibling ever needs cleanup. (Canonical on the taOS side in `AGENT_HANDOFF.md`;
this is the taOSmd mirror.)

## STEP 0b - RATE-LIMIT RECOVERY CHECK

Look for work a rate limit interrupted since the last audit, and salvage it
before anything else: background agents that returned only a session-limit
message; worktrees under `.claude/worktrees/` with uncommitted changes or
unpushed local commits (`git status --porcelain` + `git log origin/<branch>..HEAD`
per worktree); pushed branches with no PR; bus handoff lines without completions.
Test/commit/push/PR each, or redispatch a continuation agent with the original
spec.

## STEP 1 - DOCS FRESHNESS

`git fetch origin`; `git log --since="40 minutes ago" --oneline origin/master`;
`gh pr list --state merged --limit 5`. Check `README.md`, `CHANGELOG.md`,
`AGENTS.md`, `STATUS.md`, `docs/*.md` (esp. `benchmarks.md`), the
`http_server.py` endpoint docstring table, and `taosmd/skills/taosmd-a2a/`
against those changes. Fix small drift straight to master; branch + PR for big
rewrites. Commit rules: plain commits as jaylfc, no AI attribution, no em dashes,
"taOSmd" in prose (lowercase only for package/CLI/repo refs), never commit IPs or
credentials.

## STEP 2 - RESEARCH REPORT

If `docs/research-report.md` exists, check it against new results in
`benchmarks/results` and `benchmarks.md` since its last revision-log entry:
unrecorded findings, pre-registered experiments whose results have landed (move
them with their kill criterion quoted verbatim), index rows missing or stale.
Follow `.claude/skills/research-report/SKILL.md` exactly. If the report does not
exist yet, note that its first edition is pending and skip.

## STEP 3 - MEMORY

Update the session memory dev-status block and its index line if stale.

## STEP 4 - BUS (ALL CHANNELS)

- **HEARTBEAT FIRST, every fire:** touch `~/.taosmd-agent/heartbeat`.
- **Then verify the backup itself is alive:** `crontab -l | grep
  taosmd-agent/backup_watch` must return the `51 */2` line. If it is missing,
  re-add it with **PATH-PRECISE dedup** (`grep -v "taosmd-agent/backup_watch"`,
  **never the bare basename** - three leads share this user account and all three
  scripts are named `backup_watch.py`, and a peer's basename-filtered reinstall
  silently wiped this entry once), then report the wipe on the agent-rules
  thread. An independent poller (`51 */2 * * *`, `~/.taosmd-agent/backup_watch.py`,
  no model tokens) posts ONE alert to agent-rules (6h cooldown) if it finds
  unhandled traffic while this heartbeat is >90 min stale. If you are reading
  this after such an alert, read the backup's watermark file for where handling
  stopped.
- **Sweep with `~/.taos-team/a2a_catchup.sh`** (since-watermark, filtered, always
  ends in one summary line). **To find out what a thread IS, use
  `a2a_catchup.sh --threads`** - metadata only, no bodies, watermark untouched.
  **Never read a raw page with bodies printed**; that is the burn these tools
  replace (one such read cost ~8k tokens to learn a thread was irrelevant).
- **API facts (verified):** the bus has no `channel` field. Channels live in
  `thread`, so the filter is `?thread=<name>`; **`?channel=` is silently
  dropped**, which makes a per-channel sweep return the unfiltered global feed
  once per channel. The timestamp field is **`ts`, NOT `created_at`** - a
  client-side `created_at` filter gets None, defaults to 0, and returns a
  confident EMPTY result. Prefer `?since=<float epoch>&limit=200` (measured 199x
  smaller than `limit=500` for identical coverage). **ALWAYS pass an explicit
  `limit`:** a BARE read returns only the most recent ~50 and looks IDENTICAL to
  a complete one (a peer lead silently ate a 37-message window this way).
  Explicit limits of 50/60/200/500 are all honoured exactly.
- **Completeness check:** compare the oldest returned id against the last-handled
  id from the previous fire. If the oldest is beyond it, messages fell off the
  end and the read is NOT complete - say so, rather than reporting quiet.
- Scan for anything addressed to or relevant to @taOSmd / @taOSmd-dev (handoffs,
  decisions, requests, `[WINDDOWN]`s, and replies that landed in the wrong
  thread). Answer trivial acks; log substantive work to STATUS and report it,
  never drop it silently. Record docs-relevant decisions. Do not derail a monitor
  run into long bus threads.

## STEP 5 - REPO (gh)

We auto-merge overnight, so bot reviews can land post-merge and must be swept.
Check for new or updated PRs, new issues, and BOT reviews + comments
(coderabbitai, kilo-code-bot, qodo-code-review) on open AND recently-merged PRs.
Triage genuine findings vs nitpicks; fix small real ones straight to master or a
follow-up PR, flag the rest in STATUS.

**External-contributor backstop:** contributors not on the bus raise contract
questions as `contract-question` issues on the PRIVATE repo
`jaylfc/taos-agent-commons`. @taOS-dev sweeps that repo and relays, so we do not
poll it properly and do not read the bodies - but their relay is a single point
of failure, and a channel that silently stops while still looking like a route is
worse than no route. So glance at the **TITLES ONLY**: `gh issue list --repo
jaylfc/taos-agent-commons --limit 10 --json number,title,labels`. If a title is
clearly aimed at taOSmd and no relay has arrived on the bus within a couple of
hours, assume the relay is down and answer directly on the issue. Temporary
scaffolding; retires when external contributors can hold a bus identity. Worth
telling contributors: `jaylfc/taosmd` is PUBLIC with issues enabled and this step
already sweeps it hourly, so a question opened there reaches us within the hour
with no relay and no single point of failure.

## STEP 6 - FORK FRESHNESS (at most once a day: only on the 09:xx fire)

Skip on every other hour. **Surface, do NOT action infra ourselves** - the Pi and
its NPU runtime are @taOS-dev's to deploy; our job is to notice, flag and
coordinate, not to touch their live host.

- **rknn-llm:** `gh api repos/airockchip/rknn-llm/releases/latest` - flag
  anything newer than release-v1.3.0 (2026-06-17). NPU model support rides these
  releases (gemma4 + qwen3.5 RK3588 support landed in 1.3.0), so a new one may
  unlock more of taosmd's models on NPU.
- **rkllama:** is `jaylfc/rkllama` main behind upstream `notpunchnox/rkllama`,
  and is the Pi deployment behind our main? Respect the per-model-locking history.
- **qmd:** `gh api repos/tobi/qmd/tags` - a tag newer than v2.5.3 (especially
  v2.5.4) triggers a rebase of `@jaylfc/qmd` onto it and a republish, then rotate
  the npm token.

Something moved: one-line surface to the bus for @taOS-dev (their infra) or a
STATUS note (our forks/packages), and tell Jay. Nothing moved: one line.

---

**Output:** one short line when nothing changed; otherwise recovered, checked,
fixed, skipped.
