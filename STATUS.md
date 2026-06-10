# STATUS

Working snapshot for whoever picks this repo up next. Updated whenever meaningful state changes. Shipped changes live in [CHANGELOG.md](CHANGELOG.md); measured results live in [docs/benchmarks.md](docs/benchmarks.md); the task list of record is GitHub issues.

Last updated: 2026-06-10 (night)

## Current state

- **master** is healthy: 702 tests passing. NEW: the dependency-aware task-graph component shipped (PR #150): `taosmd tasks` CLI, `/tasks` HTTP surface with a ready queue and a `prime` briefing endpoint, MCP tools, all mutations archive-backed and replayable. Beads concepts credited. `taosmd tasks ready` is now the canonical what-next query for incoming agents once deployed. The taOS user-memory unification surface (`POST /ingest/batch` with idempotent re-import, `mode=bm25` on search) shipped in PR #149 and is verified live by the downstream consumer. A post-merge review pass landed six hardening fixes (see CHANGELOG, Unreleased / Fixed).
- **Benchmarks:** the late-interaction retrieval lever is confirmed at full-1540 scale (+0.037 lenient / +0.044 strict-instruct over the dense baseline, with no reranker in the loop). A retrieval-only CPU probe shows the lever is viable without a GPU: evidence recall 0.641 (dense) to 0.854 (answerai backbone) at ~110ms per query on a 16-core CPU box. Full tables and caveats in docs/benchmarks.md.

## In flight

- answerai-colbert-small full-1540 DONE: 0.716 gemma4:e2b / 0.388 llama3.1:8b / 0.656 qwen3:4b-instruct-2507 (see docs/benchmarks.md late-interaction section). Still queued on the GPU box: (1) a qwen3:14b judge column over all full-1540 prediction files plus a gemma4:12b generator A/B; (2) a full LoCoMo-Refined run (generation with the leader recipe on the 1,382 refined questions, then their official Qwen3-14B judge pointed at local Ollama).
- Branch `feat/pylate-loader` (pushed): loads ColBERT models through pylate so projection heads actually apply; per-instance token dim; 221 tests green. Ready for a projected-space vs backbone comparison next GPU window.
- Branch `bench/retrieval-latency-probe`: retrieval-only R@K + latency harness (no LLM dependency). Candidate for merge after the colbert branch lands.
- Branch `feat/colbert-models-probe`: late-interaction and `--colbert-model` support, wrong-dimension guards added. Note: it was cut from a stale base, so rebase onto master before merging.

## Queued next

1. LoCoMo-Refined harness run. The dataset is CC BY-NC: run it and publish scores, never vendor the dataset into this repo.
2. pylate loader for ColBERT projection heads. Today's sentence-transformers token path uses backbone embeddings only, so answerai numbers are backbone MaxSim, not the trained ColBERT space. LateOn and ColBERT-Zero probes follow once the loader exists.
3. snowflake-arctic-embed-s/xs probe as a MiniLM ONNX drop-in for low-power tiers.
4. Project-scoped grants (taOS #744): project_id as a verified JWT claim and per-grant rows; GrantsVerifier matches (canonical_id, project_id); append-only reattach. Contract agreed, taosmd implementation queued.

## Working agreements

- Durable state lives in three places: GitHub issues (tasks), this file (snapshot), and the A2A bus (the taosmd-progress channel is the running log).
- Small doc fixes go straight to master. Features, refactors, and redesigns go through a branch and PR.
- Benchmark policy: real datasets, external judges, publish methodology, record negative results too.

## On arrival (incoming agent or contributor)

1. Read this file top to bottom.
2. `git fetch origin` and review recent commits on master plus any open branches.
3. Check open GitHub issues and PRs.
4. Tail the A2A bus: taosmd-progress, general, integration channels.
5. Read the project memory index if you have one.
Then take the top unblocked item, or continue whatever "In flight" points at.

## On rate limit or handoff (the moment you see the warning)

1. Commit and push WIP on a branch. Never leave uncommitted work.
2. Update this file: move your task to "In flight" with the branch name, exactly where you stopped, and the next concrete step.
3. Post one line to taosmd-progress: finished X, mid-flight Y on branch Z, next step W.
The next agent recovers from last push + open PRs + this file + issues.
