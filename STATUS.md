# STATUS

Working snapshot for whoever picks this repo up next. Updated whenever meaningful state changes. Shipped changes live in [CHANGELOG.md](CHANGELOG.md); measured results live in [docs/benchmarks.md](docs/benchmarks.md); the task list of record is GitHub issues.

Last updated: 2026-06-12 (weekly dark window; both agents down until the seven-day reset at 2026-06-14T02:00:00Z)

## Current state

- **master** is healthy: about 850 tests passing. Shipped today (Jun 12): the temporal date-range lever (`retrieve(temporal=...)`, default off, PR #157), pause/resume checkpointing for the LoCoMo runner (`--ckpt`/`--resume`/`--pause-flag`, PR #158), compact mode on `GET /a2a/messages` (`fields` projection + `ndjson`), and an auth fix (`faaaf3c`): the data-endpoint grants gate now fires for every verified token, not only project-bound ones, plus ISO-8601 grant-expiry parsing. Earlier this cycle: project-scoped grants (#151), task-graph (#150), batch ingest + bm25 mode (#149), TTL filter + three-number summaries (#155), admin surface (#153), research report first edition (#156).
- **taOS #744 is CLOSED, e2e-verified 3/3** against the real registry feeds (live grant pass with claim-binding, expired-grant 403, revoked 403). The exercise found and fixed the gate-bypass bug above, the argument for e2e over unit coverage.
- **Benchmarks: all pre-registered experiments resolved today, three negatives by their own written criteria** (docs/research-report.md rev 1.10): N-008 (projected ColBERT spaces, both 0.730, do not beat the answerai backbone 0.760 subset-200 gemma, no recipe upgrade), N-009 (surprise-boundary chunking was a coverage artifact: matched-turn-budget baseline at k=120 beats chunks at k=20 by 0.15), N-010 (write-skip floor is safe but fires on only 2.3 percent of LoCoMo turns, under the 5 percent shipping floor). Headlines unchanged: 97.0 percent LongMemEval-S, LoCoMo leader 0.748/0.394/0.659.

## In flight

- **Nothing running.** All experiments closed, Fedora drained and powered down (565 result files archived to `~/Development/taosmd-bench-archive/fedora-results-20260612/` on the Mac), the VPS is the only available bench target. Both agents dark for the weekly limit; the lead session wakes shortly after the 2026-06-14T02:00:00Z reset.
- **Interim crew:** a low-risk job pack is live at `docs/agent-jobs/` (README of absolute rules + three jobs: benchmarks.md em-dash sweep, cross-encoder cwd-path fix, dead-import cleanup). Weaker agents PR these to master; they NEVER merge. The waking lead session reviews each PR against its job file's verification section.
- **qmd fork exit path:** upstream PR tobi/qmd#728 open (tsc cleanup 51 to 3, zero behavior change); #663 (brettdavies, continues our #511 with credit, includes /search + /vsearch) has two ready branches linked from us (`dim-guard-on-663`, `fix-663-test-types`). When #663 merges and a release cuts, our fork can be dropped. Watch both for maintainer replies.

## Queued next (for the post-reset wake)

1. **v2 P1-vs-P2 fold decision (Jay's call).** Every retrieval-side use of surprisal is now dead (N-009, N-010, and the earlier flat prior), so the re-scoped P1 plan (`~/tinyagentos-private/plans/taosmd-v2-p1-surprisal-traces-plan.md`: provider stack + TraceStore + one-trace-per-turn encoder) rides entirely on the P2 bet that surprisal-seeded strength helps consolidation priority. Decide whether to execute the slim P1 now or fold it into the P2 plan.
2. **Pausable benchmarks phases 2-3 (#25):** Fedora PAUSE-on-boot orchestration + the hermes rebootwin runbook. Phase 1 (runner checkpointing) is merged.
3. **Review the interim crew's PRs** against docs/agent-jobs/, and the qmd upstream PRs (#728, #663).
4. snowflake-arctic-embed-s/xs probe as a MiniLM ONNX drop-in for low-power tiers (long-standing backlog).

## Working agreements

- Durable state lives in three places: GitHub issues (tasks), this file (snapshot), and the A2A bus (the taOSmd-progress channel is the running log).
- Small doc fixes go straight to master. Features, refactors, and redesigns go through a branch and PR.
- Benchmark policy: real datasets, external judges, publish methodology, record negative results too.

## On arrival (incoming agent or contributor)

1. Read this file top to bottom.
2. ARM THE SESSION INFRASTRUCTURE (project rule, every session, no exceptions): (a) the 30-minute audit cron using the canonical prompt in `.claude/audit-cron-prompt.md` (schedule 7,37 * * * *), covering rate-limit recovery, docs freshness, the research report, memory, and the bus; (b) the A2A connection per `.claude/a2a-session-setup.md`: live watcher, durable poll floor verified, history caught up, identity rules observed.
2. `git fetch origin` and review recent commits on master plus any open branches.
3. Check open GitHub issues and PRs.
4. Tail the A2A bus: taOSmd-progress, general, integration channels.
5. Read the project memory index if you have one.
Then take the top unblocked item, or continue whatever "In flight" points at.

## On rate limit or handoff (the moment you see the warning)

1. Commit and push WIP on a branch. Never leave uncommitted work.
2. Update this file: move your task to "In flight" with the branch name, exactly where you stopped, and the next concrete step.
3. Post one line to taOSmd-progress: finished X, mid-flight Y on branch Z, next step W.
The next agent recovers from last push + open PRs + this file + issues.
