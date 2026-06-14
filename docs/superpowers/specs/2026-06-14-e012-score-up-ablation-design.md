# E-012 score-up ablation design

**Status:** approved (Jay, 2026-06-14). Build now, run the matrix when the GPU is claimed.

**Goal:** lift the honest end-to-end Judge score on LongMemEval-S above the F-012
baseline of 47.2% (236/500), and attribute any gain to a specific lever via a
clean ablation, not a single tangled config.

## Background

F-012 established the real Judge number: 47.2% on the 500-question oracle set,
with a qwen3.5:9b generator, a strict external Qwen3-4B-Instruct judge, MiniLM
embeddings, and the depth-tuned evidence pipeline. The diagnosis was that
retrieval is solved (97.0% Recall@5) but reasoning is the bottleneck: temporal
17.3% and multi-session 28.6%, against single-session and knowledge 71 to 89%.
The generator was also evidence-starved at the original depth.

The levers below target the reasoning bottleneck and the evidence quality, in
the order Jay chose: reasoning scaffolds and context-release first (levers 1 to
3), the bigger generator as a separate arm afterwards.

## The levers

Each lever is an independent toggle on `benchmarks/longmemeval_runner.py`
(branch `bench/e012-judge-harness`), gated by an environment variable, default
off so the baseline is unchanged.

**Lever 1 - context release (already built).** Retrieve more, then rerank and
keep a clean subset before generation. Implemented: depth knobs
(`TAOSMD_ASSEMBLE_TOKENS`, `TAOSMD_RETRIEVE_LIMIT`, `TAOSMD_FTS_LIMIT`,
`TAOSMD_CONTEXT_CHARS`) and the `TAOSMD_RERANK` cross-encoder toggle
(bge-reranker-v2-m3 ONNX, `TAOSMD_RERANK_TOP_K`). No new code; it is an arm in
the matrix.

**Lever 2 - query decomposition with iterative retrieval (port).** Split the
question into 2 to 3 focused sub-queries, retrieve for each, union the hits
deduplicated by text, cap at the retrieve limit. Port `_decompose_query` and
`MULTIHOP_PROMPT` from `benchmarks/locomo_runner.py` (a small gemma4:e2b
utility call, temperature 0, falls back to the original question on failure or
fewer than two lines). New env toggle `TAOSMD_DECOMPOSE=1`. Targets
multi-session questions that need facts from more than one place.

**Lever 3 - answer self-verification, CoVe-style (fresh build).** After the
first answer, run one verification pass: given the context and the draft
answer, ask the generator to flag any part not supported by the context and
return a corrected answer, or the draft unchanged if fully supported. This is a
lightweight Chain-of-Verification adapted to QA; there is no existing
implementation to port, so it is built fresh. New env toggle
`TAOSMD_SELF_VERIFY=1`. Uses the same generator and the `/no_think` discipline
already in the runner. Targets answers that drift from the evidence.

**Separate arm - bigger generator.** qwen3.6:35b-a3b in place of qwen3.5:9b, via
the existing `TAOSMD_OLLAMA_MODEL` env. No code change. Run after levers 1 to 3
so we know what the scaffolds buy on the smaller, faster generator first.

## Ablation matrix

Same fixed substrate for every arm: oracle 500-question set, MiniLM embeddings,
the hybrid plus query-expansion retrieval, the strict external
Qwen3-4B-Instruct judge, `/no_think`, temperature 0. Only the lever under test
changes.

1. baseline (no levers) - reproduce 47.2% as the control in this run
2. lever 1 alone (context release)
3. lever 2 alone (decomposition)
4. lever 3 alone (self-verification)
5. levers 1 + 2
6. levers 1 + 3
7. levers 2 + 3
8. levers 1 + 2 + 3 (all scaffolds)
9. best-of-1-to-8 with the bigger generator

Each arm reports overall Judge accuracy and the per-category breakdown, so a
lever that helps temporal or multi-session but not overall is still visible.
Read each arm against the baseline control, not against the prior 47.2% (judge
and sampling noise differ across runs; the control absorbs that).

## Judge protocol

The ablation deltas use the single strict Qwen3-4B-Instruct judge for
comparability and cost. The winning arm gets a tri-judge firm-up (gemma4:e2b
lenient, llama3.1:8b strict, qwen3:4b-instruct-2507 strict) so the reported
gain is not judge-specific, the same discipline E-009 uses before a default
flip. A lever only counts as a win if it clears the baseline by more than
plausible run-to-run noise and survives the firm-up.

## GPU and run protocol

The matrix needs the 3060 (9B and 35B generators plus the judge). Coordinate
with @taOS over the lease protocol: CHECK plus nvidia-smi, post [GPU CLAIM]
before running, [GPU RELEASE] when done, their demo has priority. Size models so
co-resident generator plus judge stay under 12 GB to avoid the swap deadlock
that hung E-009 (qwen3.5:9b plus Qwen3-4B-Instruct judge fits; the 35B arm runs
with the judge pass sequenced, not co-resident, if it would exceed the budget).
Build is GPU-free and happens first; the GPU is claimed only when an arm is
ready to run.

## Success criteria

- A per-lever attribution table: each lever's overall and per-category delta
  versus the baseline control.
- At least the all-scaffolds arm and the bigger-generator arm measured.
- The winning configuration confirmed under the tri-judge firm-up.
- A recorded research-report finding (F or N) with provenance, whichever way it
  lands. A null result (scaffolds do not move the Judge number) is a publishable
  outcome and gets recorded as such.

## Out of scope (YAGNI)

- No new embedder work (E-010 settled arctic as the dense default; this run
  holds the embedder fixed for comparability).
- No full Provence-style learned pruner; the cross-encoder rerank is the
  context-release mechanism for this round.
- No distractor-set run in this matrix; the oracle set is the headline-
  comparable surface. A distractor run is a later, separate arm.
- No changes to shipped defaults from this harness; any win flows to defaults
  through the normal branch, full-scale confirm, and Jay sign-off.
