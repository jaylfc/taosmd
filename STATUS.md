# STATUS

Working snapshot for whoever picks this repo up next. Updated whenever meaningful state changes. Shipped changes live in [CHANGELOG.md](CHANGELOG.md); measured results live in [docs/benchmarks.md](docs/benchmarks.md); the task list of record is GitHub issues.

Last updated: 2026-06-13 (back online, limits lifted; two experiments running in parallel)

## Current state

- **master** is healthy: about 850 tests passing. Shipped today (Jun 12): the temporal date-range lever (`retrieve(temporal=...)`, default off, PR #157), pause/resume checkpointing for the LoCoMo runner (`--ckpt`/`--resume`/`--pause-flag`, PR #158), compact mode on `GET /a2a/messages` (`fields` projection + `ndjson`), and an auth fix (`faaaf3c`): the data-endpoint grants gate now fires for every verified token, not only project-bound ones, plus ISO-8601 grant-expiry parsing. Earlier this cycle: project-scoped grants (#151), task-graph (#150), batch ingest + bm25 mode (#149), TTL filter + three-number summaries (#155), admin surface (#153), research report first edition (#156).
- **taOS #744 is CLOSED, e2e-verified 3/3** against the real registry feeds (live grant pass with claim-binding, expired-grant 403, revoked 403). The exercise found and fixed the gate-bypass bug above, the argument for e2e over unit coverage.
- **Benchmarks: all pre-registered experiments resolved today, three negatives by their own written criteria** (docs/research-report.md rev 1.10): N-008 (projected ColBERT spaces, both 0.730, do not beat the answerai backbone 0.760 subset-200 gemma, no recipe upgrade), N-009 (surprise-boundary chunking was a coverage artifact: matched-turn-budget baseline at k=120 beats chunks at k=20 by 0.15), N-010 (write-skip floor is safe but fires on only 2.3 percent of LoCoMo turns, under the 5 percent shipping floor). Headlines unchanged: 97.0 percent LongMemEval-S, LoCoMo leader 0.748/0.394/0.659.

## In flight

- **E-007 arctic-embed CONFIRMED at full-1540 (F-010): the win to ship.** R@K +0.157, judged +0.0565 full-1540 (arctic-s 0.7305 vs MiniLM 0.6740), same 384 dim and latency (report rev 1.15, section 3.7). **PR #160 (enabler) and PR #161 (recipe-aware store seam) both MERGED.** Arctic ship is **PR #162 open** (feat/arctic-low-tier-default): config-level embed_model honored in _ensure_stores, embedder in the store-mode guard (switch = fail loud), setup.sh fetch, lite-pi+fast-8b recipes select arctic, benchmarks.md updated, 862 tests green. REMAINING before #162 merge: a gentle Pi-CPU speed sample + @taOS store-path verification (msg 417 sent: a recommended model is not shipped until its taOS-store download path is verified, now ship-the-win step 6).
- **E-008 RESOLVED to N-011: surprisal is the WORST retention signal (below random at every budget); length wins.** The v2 surprisal pillar is conclusively dead across retrieval (N-009/N-010) and consolidation (N-011), so the v2 spine PIVOTS to the provenance/claims layer (anchored by F-009, 18.8 percent extraction-hallucination rate, unmatched), per the pre-registered criterion and Jay's plan (report rev 1.14). Surprisal P1 plan shelved. **v2 claims-layer (Provable Memory) BUILT, PR #163 open (default-off, not self-merged).** taosmd/claims/ package: zero-loss ClaimStore, pure demote-not-delete gate, cross-family fail-closed entailment verifier, extract-with-archive-span-provenance, async verify_pass, ingest+search wiring (search(prefer_verified) default off), CLI (verify/claims status). 886 tests pass (+24), clean isolation. Turns F-009 (18.8 percent hallucination) into a live always-on gate. E-009 pre-registered (rev 1.16). FOLLOW-UPS gating the default flip: build benchmarks/claims_gate_probe.py + run E-009. Spec/plan private at ~/tinyagentos-private/{specs,plans}/2026-06-13-v2-claims-layer-*. Side-finding: length is a strong free retention heuristic.
- **qmd fork exit path:** upstream PR tobi/qmd#728 open (tsc cleanup 51 to 3, zero behavior change); #663 (brettdavies, continues our #511 with credit, includes /search + /vsearch) has two ready branches linked from us (`dim-guard-on-663`, `fix-663-test-types`). When #663 merges and a release cuts, our fork can be dropped. Watch both for maintainer replies.

## Queued next

1. **v2 spine direction (Jay confirmed the plan):** give surprisal the one E-008 consolidation shot (running), then pivot to the provenance/claims layer regardless. F-009 (18.8 percent extraction-hallucination rate, cross-family verified) is the strongest v2-relevant result and is unmatched by competitors; it anchors that layer.
2. **E-007 follow-ups if the confirm holds:** full-1540 judged accuracy (Fedora/VPS) and a small Pi-CPU speed sample, then a default-switch proposal for the low tier.
3. **Interim job pack** at `docs/agent-jobs/` (three low-risk jobs) remains for weaker agents; review any PRs against each job's verification section, never auto-merge.
4. snowflake-arctic-embed result feeds the hardware-tier defaults matrix once the confirm lands.

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
