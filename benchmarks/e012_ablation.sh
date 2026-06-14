#!/usr/bin/env bash
# E-012 score-up ablation. Sweeps the lever arms SEQUENTIALLY (one Ollama job
# at a time per the GPU-contention rule) and records each arm's overall and
# per-type Judge accuracy.
#
# Usage: benchmarks/e012_ablation.sh [LIMIT]
#   LIMIT defaults to 100 (screening pass). Use 500 for the full oracle set.
#
# Required env (export before running):
#   TAOSMD_OLLAMA_MODEL   generator (e.g. qwen3.5:9b)
#   TAOSMD_JUDGE_MODEL    external judge (e.g. qwen3:4b-instruct-2507)
#   TAOSMD_OLLAMA_URL     Ollama base URL (default http://localhost:11434)
# Fixed substrate (depth + rerank context-release = lever 1) is set per arm.
set -u
LIMIT="${1:-100}"
PY=".venv/bin/python"
OUT="benchmarks/results/e012_ablation_$(date -u +%Y%m%d_%H%M%S).log"
RUNNER="benchmarks/longmemeval_runner.py"

run_arm () {
  local name="$1"; shift
  echo "==================== ARM: $name (limit=$LIMIT) ====================" | tee -a "$OUT"
  # Reset all levers off, then apply this arm's env (passed as VAR=VAL args).
  env -u TAOSMD_DECOMPOSE -u TAOSMD_SELF_VERIFY -u TAOSMD_RERANK \
      TAOSMD_ASSEMBLE_TOKENS="${TAOSMD_ASSEMBLE_TOKENS:-4000}" \
      TAOSMD_RETRIEVE_LIMIT="${TAOSMD_RETRIEVE_LIMIT:-12}" \
      TAOSMD_FTS_LIMIT="${TAOSMD_FTS_LIMIT:-5}" \
      TAOSMD_CONTEXT_CHARS="${TAOSMD_CONTEXT_CHARS:-16000}" \
      "$@" \
      "$PY" "$RUNNER" --llm --limit "$LIMIT" 2>&1 | tee -a "$OUT"
  echo "" | tee -a "$OUT"
}

# Arms. Lever 1 = TAOSMD_RERANK=1, lever 2 = TAOSMD_DECOMPOSE=1, lever 3 = TAOSMD_SELF_VERIFY=1.
run_arm "baseline"
run_arm "L1_context_release"  TAOSMD_RERANK=1
run_arm "L2_decompose"        TAOSMD_DECOMPOSE=1
run_arm "L3_self_verify"      TAOSMD_SELF_VERIFY=1
run_arm "L1+L2"               TAOSMD_RERANK=1 TAOSMD_DECOMPOSE=1
run_arm "L1+L3"               TAOSMD_RERANK=1 TAOSMD_SELF_VERIFY=1
run_arm "L2+L3"               TAOSMD_DECOMPOSE=1 TAOSMD_SELF_VERIFY=1
run_arm "L1+L2+L3"            TAOSMD_RERANK=1 TAOSMD_DECOMPOSE=1 TAOSMD_SELF_VERIFY=1

echo "Ablation complete. Full log: $OUT" | tee -a "$OUT"
echo "Per-arm overall accuracy:" | tee -a "$OUT"
grep -E "ARM:|Overall:" "$OUT" | tee -a "$OUT"
