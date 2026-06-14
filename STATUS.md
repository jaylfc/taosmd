# STATUS

Working snapshot for whoever picks this repo up next. Updated whenever meaningful state changes. Shipped changes live in [CHANGELOG.md](CHANGELOG.md); measured results live in [docs/benchmarks.md](docs/benchmarks.md); the task list of record is GitHub issues.

Last updated: 2026-06-13 (arctic + claims layer both merged to master; E-009 claims-gate probe built and running on Fedora)

## Current state

- **master** is healthy: about 886 tests passing. Shipped today (Jun 13): **PR #162 arctic-embed-s as the low-tier dense default** (ships F-010 -- R@K +0.157, judged +0.0565 full-1540, same 384-dim and latency; existing installs keep MiniLM via the config default, only fresh setups fetch arctic) and **PR #163 the v2 claims layer "Provable Memory"** (default-off: `taosmd/claims/` -- zero-loss ClaimStore, cross-family fail-closed entailment verifier, demote-not-delete recall gate, archive-span provenance, async verify-pass, `taosmd verify`/`claims status` CLI). Earlier this cycle (Jun 12): temporal date-range lever (PR #157), pause/resume checkpointing for the LoCoMo runner (PR #158), compact mode on `GET /a2a/messages`, the grants-gate auth fix (`faaaf3c`), project-scoped grants (#151), task-graph (#150), batch ingest + bm25 mode (#149), TTL filter + three-number summaries (#155), admin surface (#153), research report first edition (#156).
- **taOS #744 is CLOSED, e2e-verified 3/3** against the real registry feeds (live grant pass with claim-binding, expired-grant 403, revoked 403). The exercise found and fixed the gate-bypass bug above, the argument for e2e over unit coverage.
- **Benchmarks: all pre-registered experiments resolved today, three negatives by their own written criteria** (docs/research-report.md rev 1.10): N-008 (projected ColBERT spaces, both 0.730, do not beat the answerai backbone 0.760 subset-200 gemma, no recipe upgrade), N-009 (surprise-boundary chunking was a coverage artifact: matched-turn-budget baseline at k=120 beats chunks at k=20 by 0.15), N-010 (write-skip floor is safe but fires on only 2.3 percent of LoCoMo turns, under the 5 percent shipping floor). Headlines unchanged: 97.0 percent LongMemEval-S, LoCoMo leader 0.748/0.394/0.659.

## In flight

- **E-009 claims-gate probe RUNNING on Fedora (gates the prefer_verified default flip).** Harness `benchmarks/claims_gate_probe.py` on branch `bench/e009-claims-gate-probe`: drives the SHIPPED claims layer end to end on a LoCoMo slice (real ClaimStore + `claims_from_text` provenance + `LocalEntailmentVerifier` + `verify_pass` + pure `apply_claims_gate`), answers every QA under gate `off`/`prefer_verified`/`strict`, judged externally; reports judged accuracy, R@K (runner's macro definition), and served-hallucination rate per mode, then the pre-registered kill criterion. Code-reviewed (two faithfulness bugs fixed: micro vs macro R@K; gates the production-shaped top-k input pre-gate). Offline `--self-test` green. Smoke run on Fedora end-to-end OK (one conv -> 168 claims -> all verified, live extraction-hallucination 32.1%); needed qwen3 `/no_think` and the onnx model DIR path (file path doubled -> QMD fallback). NEXT: full LoCoMo sweep (10 conv) then record E-009 verdict in the report and propose/withhold the default flip. Verifier model on Fedora: `hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M`; generator qwen3:4b; judge qwen3:14b.
- **E-007 arctic-embed SHIPPED (F-010), PR #162 MERGED.** R@K +0.157, judged +0.0565 full-1540 (arctic-s 0.7305 vs MiniLM 0.6740), same 384 dim and latency (report rev 1.15). Enablers #160/#161 also merged. Existing installs keep MiniLM (config default); fresh setups fetch arctic via the tested HF path (Pi 13ms/embed). **OPEN FOLLOW-UP (not a blocker): @taOS owes arctic store-registration + verified OS-store download path** (integration msg 417; bus 422 reframed it as a follow-up). A gentle Pi-CPU speed sample is the remaining nice-to-have.
- **E-008 RESOLVED to N-011: surprisal is the WORST retention signal; length wins.** The v2 surprisal pillar is dead across retrieval (N-009/N-010) and consolidation (N-011); the v2 spine is the provenance/claims layer (anchored by F-009, 18.8 percent extraction-hallucination, unmatched). **Claims layer (Provable Memory) MERGED, PR #163** (default-off). E-009 (above) is the validation that decides whether `prefer_verified` ever flips default-on. Spec/plan private at ~/tinyagentos-private/{specs,plans}/2026-06-13-v2-claims-layer-*. Side-finding: length is a strong free retention heuristic (own pre-registration deferred).
- **qmd fork exit path:** upstream PR tobi/qmd#728 open (tsc cleanup 51 to 3, zero behavior change); #663 (brettdavies, continues our #511 with credit, includes /search + /vsearch) has two ready branches linked from us (`dim-guard-on-663`, `fix-663-test-types`). When #663 merges and a release cuts, our fork can be dropped. Watch both for maintainer replies.

## Queued next

1. **E-009 full LoCoMo sweep + verdict** (the live in-flight item): the 10-conv probe is RUNNING on Fedora (nohup, benchmarks/results/e009.json, log e009_run.log); record the result in the research report under pre-registered E-009 and either propose the `prefer_verified` default flip or document the trade and keep it default-off. Uncheckpointed: relaunch only if the proc is dead AND e009.json is absent. **WHEN IT LANDS: ping @taOS on the taOS-taOSmd-hermes-integration channel that e009.json is recorded and the Fedora box is FREE** — @taOS is waiting on box-time for their offline tool-calling eval (qwen3.5:9b is pulled, 4b is not; never overlap two Ollama jobs on the 3060). I committed to that ping.
2. **E-010 embedding-model bake-off** (pre-registered rev 1.17, queued): judge-free subset-200 R@K screen of the small/fast/ONNX embedding field against arctic-embed-s, per-model pooling+prefix validity gate, license + ONNX-<=384dim gates, footprint tiebreak; judged full-1540 confirm for any winner. Shortlist + integration notes in [[reference_embedding_bakeoff_candidates]]. CPU-bound (no GPU), can run concurrently with GPU work or on the VPS.
3. **E-011 arctic on LongMemEval-S** (pre-registered rev 1.17, queued): does MiniLM->arctic move the 97.0% headline? MiniLM vs arctic arms on the full 500-Q LongMemEval-S, same generator/judge. Needs the GPU host; runs after E-009 frees it.
4. **@taOS arctic store-registration** (owed to us): register Snowflake/snowflake-arctic-embed-s with a verified OS-store download path; plus the taOSmd-section README/website refresh (ready-to-drop copy sent on integration; @taOS acked msg 420, in progress).
5. **Interim job pack** at `docs/agent-jobs/` (three low-risk jobs) remains for weaker agents; review any PRs against each job's verification section, never auto-merge.

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
