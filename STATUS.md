# STATUS

Working snapshot for whoever picks this repo up next. Updated whenever meaningful state changes. Shipped changes live in [CHANGELOG.md](CHANGELOG.md); measured results live in [docs/benchmarks.md](docs/benchmarks.md); the task list of record is GitHub issues.

Last updated: 2026-06-14 (E-009 RESOLVED to F-011: prefer_verified claims gate PASSES, zero served-hallucination at no accuracy cost; default flip pending Jay sign-off + tri-judge confirm)

## Current state

- **master** is healthy: about 886 tests passing. Shipped today (Jun 13): **PR #162 arctic-embed-s as the low-tier dense default** (ships F-010 -- R@K +0.157, judged +0.0565 full-1540, same 384-dim and latency; existing installs keep MiniLM via the config default, only fresh setups fetch arctic) and **PR #163 the v2 claims layer "Provable Memory"** (default-off: `taosmd/claims/` -- zero-loss ClaimStore, cross-family fail-closed entailment verifier, demote-not-delete recall gate, archive-span provenance, async verify-pass, `taosmd verify`/`claims status` CLI). Earlier this cycle (Jun 12): temporal date-range lever (PR #157), pause/resume checkpointing for the LoCoMo runner (PR #158), compact mode on `GET /a2a/messages`, the grants-gate auth fix (`faaaf3c`), project-scoped grants (#151), task-graph (#150), batch ingest + bm25 mode (#149), TTL filter + three-number summaries (#155), admin surface (#153), research report first edition (#156).
- **taOS #744 is CLOSED, e2e-verified 3/3** against the real registry feeds (live grant pass with claim-binding, expired-grant 403, revoked 403). The exercise found and fixed the gate-bypass bug above, the argument for e2e over unit coverage.
- **Benchmarks: all pre-registered experiments resolved today, three negatives by their own written criteria** (docs/research-report.md rev 1.10): N-008 (projected ColBERT spaces, both 0.730, do not beat the answerai backbone 0.760 subset-200 gemma, no recipe upgrade), N-009 (surprise-boundary chunking was a coverage artifact: matched-turn-budget baseline at k=120 beats chunks at k=20 by 0.15), N-010 (write-skip floor is safe but fires on only 2.3 percent of LoCoMo turns, under the 5 percent shipping floor). Headlines unchanged: 97.0 percent LongMemEval-S, LoCoMo leader 0.748/0.394/0.659.

## In flight

- **E-009 RESOLVED to F-011: the prefer_verified claims gate PASSES its kill criterion.** Full 10-conv LoCoMo sweep (200 QAs, 1853 claims, llama3.1:8b judge, `benchmarks/results/e009.json`): served-hallucination 0.040 -> 0.000 (eliminated), judged accuracy 0.425 -> 0.445 (no drop, +0.020 within noise at n=200), R@K 0.615 -> 0.610 (-0.005, inside the 0.02 bound). strict FAILS (judge -0.065, R@K -0.065, trades accuracy for purity) and stays opt-in. Robust claim: **prefer_verified eliminates served-hallucination at NO measured accuracy cost** (the +0.02 is noise; do not claim a gain). Recorded report rev 1.18 + F-011. **REMAINING before the default flip: Jay sign-off + a tri-judge larger-n confirm** (per "never silently enabled" + ship-the-win); until then the gate stays default-off with the validated trade documented. Harness on branch bench/e009-claims-gate-probe (PR it). NOTE: first launch with qwen3:14b judge hung on a 12GB VRAM model-swap deadlock; relaunched with llama3.1:8b co-resident (see [[feedback_gpu_contention]]).
- **E-007 arctic-embed SHIPPED (F-010), PR #162 MERGED.** R@K +0.157, judged +0.0565 full-1540 (arctic-s 0.7305 vs MiniLM 0.6740), same 384 dim and latency (report rev 1.15). Enablers #160/#161 also merged. Existing installs keep MiniLM (config default); fresh setups fetch arctic via the tested HF path (Pi 13ms/embed). **OPEN FOLLOW-UP (not a blocker): @taOS owes arctic store-registration + verified OS-store download path** (integration msg 417; bus 422 reframed it as a follow-up). A gentle Pi-CPU speed sample is the remaining nice-to-have.
- **E-008 RESOLVED to N-011: surprisal is the WORST retention signal; length wins.** The v2 surprisal pillar is dead across retrieval (N-009/N-010) and consolidation (N-011); the v2 spine is the provenance/claims layer (anchored by F-009, 18.8 percent extraction-hallucination, unmatched). **Claims layer (Provable Memory) MERGED, PR #163** (default-off). E-009 (above) is the validation that decides whether `prefer_verified` ever flips default-on. Spec/plan private at ~/tinyagentos-private/{specs,plans}/2026-06-13-v2-claims-layer-*. Side-finding: length is a strong free retention heuristic (own pre-registration deferred).
- **qmd fork exit path:** upstream PR tobi/qmd#728 open (tsc cleanup 51 to 3, zero behavior change); #663 (brettdavies, continues our #511 with credit, includes /search + /vsearch) has two ready branches linked from us (`dim-guard-on-663`, `fix-663-test-types`). When #663 merges and a release cuts, our fork can be dropped. Watch both for maintainer replies.

## Queued next

1. **E-009 DONE (F-011).** Follow-ups: (a) **PR the harness branch** bench/e009-claims-gate-probe to master; (b) **the prefer_verified default flip needs Jay sign-off + a tri-judge larger-n confirm** before flipping (n=200 single-judge is enough to validate "no cost + zero hallucination" but not to flip a shipped default silently); (c) consider documenting prefer_verified as a recommended opt-in integrity mode in the README/recipes now (it is validated as no-cost), even ahead of the default flip.
2. **@taOS Fedora model leaderboard RUNNING (their own job, GPU busy).** @taOS resolved the Fedora endpoint from their own Tailscale peer list (I declined to post it per the no-addresses rule; nothing shared) and is running the tool-call leaderboard themselves across qwen3.5:9b (iq4_xs/q5_k_m/q6_k) + qwen3:14b + llama3.1:8b + gemma4:12b + unsloth-2507 + qwen3.6:35b-a3b, one job at a time. **So the Fedora GPU is occupied; E-011 (needs the GPU host) is blocked until they finish.** Nothing owed from me here unless they ask.
3. **E-010 embedding-model bake-off** (pre-registered rev 1.17, queued): judge-free subset-200 R@K screen of the small/fast/ONNX embedding field against arctic-embed-s, per-model pooling+prefix validity gate, license + ONNX-<=384dim gates, footprint tiebreak; judged full-1540 confirm for any winner. Shortlist + integration notes in [[reference_embedding_bakeoff_candidates]]. CPU-bound (no GPU), can run concurrently with GPU work or on the VPS. Needs a small loader build first (independent pooling/prefix flags + two-sided E5/Nomic prefixes).
4. **E-011 arctic on LongMemEval-S** (pre-registered rev 1.17, queued): does MiniLM->arctic move the 97.0% headline? MiniLM vs arctic arms on the full 500-Q LongMemEval-S, same generator/judge. Needs the GPU host (free after the @taOS leaderboard).
5. **@taOS arctic store-registration** (owed to us): register Snowflake/snowflake-arctic-embed-s with a verified OS-store download path; taOSmd-section README already refreshed on @taOS dev (msg 429).
6. **Interim job pack** at `docs/agent-jobs/` (three low-risk jobs) remains for weaker agents; review any PRs against each job's verification section, never auto-merge.

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
