# STATUS

Working snapshot for whoever picks this repo up next. Updated whenever meaningful state changes. Shipped changes live in [CHANGELOG.md](CHANGELOG.md); measured results live in [docs/benchmarks.md](docs/benchmarks.md); the task list of record is GitHub issues.

Last updated: 2026-06-10 (evening)

## Current state

- **master** is healthy: 655 tests passing. The taOS user-memory unification surface (`POST /ingest/batch` with idempotent re-import, `mode=bm25` on search) shipped in PR #149 and is verified live by the downstream consumer. A post-merge review pass landed six hardening fixes (see CHANGELOG, Unreleased / Fixed).
- **Benchmarks:** the late-interaction retrieval lever is confirmed at full-1540 scale (+0.037 lenient / +0.044 strict-instruct over the dense baseline, with no reranker in the loop). A retrieval-only CPU probe shows the lever is viable without a GPU: evidence recall 0.641 (dense) to 0.854 (answerai backbone) at ~110ms per query on a 16-core CPU box. Full tables and caveats in docs/benchmarks.md.

## In flight

- GPU bench box: answerai-colbert-small full-1540 confirmation plus a stacked answerai + reranker probe, followed by a qwen3:14b judge column (the LoCoMo-Refined community judge) and a gemma4:12b generator A/B.
- Branch `bench/retrieval-latency-probe`: retrieval-only R@K + latency harness (no LLM dependency). Candidate for merge after the colbert branch lands.
- Branch `feat/colbert-models-probe`: late-interaction and `--colbert-model` support, wrong-dimension guards added. Note: it was cut from a stale base, so rebase onto master before merging.

## Queued next

1. LoCoMo-Refined harness run. The dataset is CC BY-NC: run it and publish scores, never vendor the dataset into this repo.
2. pylate loader for ColBERT projection heads. Today's sentence-transformers token path uses backbone embeddings only, so answerai numbers are backbone MaxSim, not the trained ColBERT space. LateOn and ColBERT-Zero probes follow once the loader exists.
3. snowflake-arctic-embed-s/xs probe as a MiniLM ONNX drop-in for low-power tiers.
4. Project-scoped grants (taOS #744): project_id as a verified JWT claim and per-grant rows; GrantsVerifier matches (canonical_id, project_id); append-only reattach. Contract agreed, taosmd implementation queued.

## Working agreements

- Durable state lives in three places: GitHub issues (tasks), this file (snapshot), and the A2A bus (the taosmd-progress channel is the running log).
- On a rate limit or handoff: commit and push WIP, update this file, post one line to the bus. The next person recovers from last push + issues + STATUS.md.
- Small doc fixes go straight to master. Features, refactors, and redesigns go through a branch and PR.
- Benchmark policy: real datasets, external judges, publish methodology, record negative results too.
