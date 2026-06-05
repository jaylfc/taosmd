#!/usr/bin/env bash
# Retrieval-lever probe chain (subset-200, early-abortable).
#
# Runs four levers on top of the vector-only leader recipe, in priority order,
# as 200-QA probes with --checkpoint-every 20 so we can read trends live and
# kill losers before they finish. Generator = qwen3.5:9b (self-judge inline)
# for all but P3, which swaps in the 35B-A3B generator.
#
# EARLY SIGNAL:
#   * Retrieval levers (P1a/P1b/P4a): watch `recall` (R@K) in the checkpoints —
#     objective, judge-independent. If R@K doesn't beat the baseline trend,
#     abort; no point running to completion or rescoring.
#   * Generator lever (P3): retrieval (R@K) is unchanged vs B0 — only the
#     answer changes, so its inline self-judge uses the 35B and is NOT
#     comparable to the 9b configs. Judge it via the dual-judge rescore, and
#     gate on THROUGHPUT first (abort if projected > ~6h for 200 QAs).
#
# Survivors only get escalated to full-1540 + dual-judge rescore (separately).
set -u
cd "$(dirname "$0")/.."

TS="$(date +%Y%m%d_%H%M%S)"
OUT="benchmarks/results"
LOG="${OUT}/retrieval_probes_${TS}.log"
mkdir -p "$OUT"

# Vector-only leader recipe (the binq baseline), shared across probes.
BASE="--strategy vector-only --fusion mem0_additive --top-k 10 --adjacent-turns 2 --embed-mode onnx --limit 200 --checkpoint-every 20 --no-think-prefix"

run () {  # name  extra-args...
  local name="$1"; shift
  local json="${OUT}/${TS}_${name}.json"
  echo "===================================================================" | tee -a "$LOG"
  echo "[$(date +%H:%M:%S)] PROBE ${name}  ::  $*" | tee -a "$LOG"
  echo "===================================================================" | tee -a "$LOG"
  python3 benchmarks/locomo_runner.py $BASE "$@" --out "$json" 2>&1 | tee -a "$LOG"
  echo "[$(date +%H:%M:%S)] PROBE ${name} DONE -> ${json}" | tee -a "$LOG"
}

echo "RETRIEVAL PROBE CHAIN start ${TS}" | tee -a "$LOG"

# B0 — baseline (cosine, narrow pool, no rerank). Anchors R@K + inline judge.
run baseline       --model qwen3.5:9b --retrieval-top-k 20

# P1a — coarse-to-fine rerank only (wide cosine pool -> BGE cross-encoder -> 10).
run rerank_only    --model qwen3.5:9b --retrieval-top-k 50 --reranker bge-v2-m3

# P1b — binq + rerank (wide BINARY pool -> BGE cross-encoder -> 10). The pitch:
#       binq makes the wide first stage cheap; the cross-encoder recovers precision.
run binq_rerank    --model qwen3.5:9b --retrieval-top-k 50 --reranker bge-v2-m3 --binary-quant

# P4a — stronger lexical arm in the fusion (lemma BM25 RRF).
run bm25_lemma     --model qwen3.5:9b --retrieval-top-k 20 --fusion bm25_lemma_rrf

# P3 — generator bump (35B-A3B). Same retrieval as B0; slow (partial CPU
#      offload on 12 GB). Throughput-gated: read the first checkpoint timing.
run gen_35b        --model qwen3.6:35b-a3b-q4_K_M --retrieval-top-k 20

echo "RETRIEVAL PROBE CHAIN DONE ${TS}" | tee -a "$LOG"
