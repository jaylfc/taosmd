# E-012 score-up ablation results (F-013)

Provenance for finding F-013. Honest end-to-end Judge on LongMemEval-S oracle
(500 questions). Generator qwen3.5:9b with `/no_think`, MiniLM embedder, external
strict judge Qwen3-4B-Instruct-2507 (non-thinking). Harness
`benchmarks/e012_ablation.sh` + `benchmarks/longmemeval_runner.py` on branch
`bench/e012-judge-harness`. Raw run logs: bench host `/tmp/e012_screen2.log`
(screen) and `/tmp/e012_confirm.log` (full-500 confirm).

Lever env toggles: 1a depth = the tuned defaults (`TAOSMD_ASSEMBLE_TOKENS=4000`,
`RETRIEVE_LIMIT=12`, `FTS_LIMIT=5`, `CONTEXT_CHARS=16000`); 1b rerank =
`TAOSMD_RERANK=1` (bge-reranker-v2-m3); 2 decompose = `TAOSMD_DECOMPOSE=1`;
3 self-verify = `TAOSMD_SELF_VERIFY=1`. The starved anchor uses 2000/5/3/3000 to
reproduce the F-012 floor.

## Screen (n=100 representative, seed 42, all six question types)

| arm | Judge | vs tuned baseline |
|---|---|---|
| F012_anchor_starved | 24/100 | starved floor |
| baseline_tuned_depth | 54/100 | depth lever 1a |
| L1b_rerank | 60/100 | +6 |
| L2_decompose | 54/100 | +0 (dud) |
| L3_self_verify | 74/100 | +20 |
| L1b+L2 | 57/100 | decompose drags rerank down |
| **L1b+L3 (winner)** | **76/100** | best combination |
| L2+L3 | 70/100 | decompose drags self-verify down |
| all_L1b+L2+L3 | 74/100 | decompose drags the winner down |

Decomposition lowers every combination it joins, so it is dropped.

## Full-500 confirm (oracle, strict Qwen3-4B-Instruct judge)

| arm | overall | knowledge | multi-session | ss-assistant | ss-preference | ss-user | temporal |
|---|---|---|---|---|---|---|---|
| baseline_tuned_depth | 284/500 (56.8%) | 83.3% | 39.8% | 96.4% | 53.3% | 80.0% | 30.1% |
| **rerank+self-verify (winner)** | **373/500 (74.6%)** | 91.0% | 66.9% | 98.2% | 76.7% | 85.7% | 56.4% |

Reference: F-012 published baseline (starved config) was 236/500 = 47.2%. The
chain is 47.2% (starved) -> 56.8% (depth tuned) -> 74.6% (rerank + self-verify),
a +27.4pp gain over the published baseline, no model change.

Self-verification carries the win and lands on the diagnosed bottlenecks:
temporal-reasoning 30.1 to 56.4%, multi-session 39.8 to 66.9%, single-session-
preference 53.3 to 76.7%; the already-strong categories stay near ceiling.

## Firm-up

Cross-family llama3.1:8b judge re-run of baseline + winner is in progress to
confirm the delta is not specific to the Qwen judge. The lenient gemma4:e2b judge
does not fit co-resident with the 9B generator (12 GB budget), so it is deferred
to a prediction-cache re-judge.
