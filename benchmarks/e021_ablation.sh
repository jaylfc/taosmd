#!/usr/bin/env bash
# E-021 generator-prompt ablation. Sweeps the four answer-prompt arms
# SEQUENTIALLY (one Ollama job at a time per the GPU-contention rule) on top of
# the E-012 recommended substrate: rerank + self-verify ON as the baseline.
# Each arm runs at the n=100 screen size against the fixed strict judge.
#
# DO NOT run this while another job holds the GPU.
#
# Usage: benchmarks/e021_ablation.sh [LIMIT]
#   LIMIT defaults to 100 (screening pass).
#
# Required env (export before running):
#   TAOSMD_OLLAMA_MODEL   generator (qwen3.5:9b)
#   TAOSMD_JUDGE_MODEL    external strict judge
#   TAOSMD_OLLAMA_URL     Ollama base URL (default http://localhost:11434)
set -u
LIMIT="${1:-100}"
PY=".venv/bin/python"
OUT="benchmarks/results/e021_ablation_$(date -u +%Y%m%d_%H%M%S).log"
RUNNER="benchmarks/longmemeval_runner.py"

# Fixed strict judge + generator for the screen (overridable from the env).
export TAOSMD_JUDGE_MODEL="${TAOSMD_JUDGE_MODEL:-hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M}"
export TAOSMD_OLLAMA_MODEL="${TAOSMD_OLLAMA_MODEL:-qwen3.5:9b}"

run_arm () {
  local name="$1"; shift
  echo "==================== ARM: $name (limit=$LIMIT) ====================" | tee -a "$OUT"
  # Reset every E-012 and E-021 lever off, then apply this arm's env. The
  # E-012 substrate (rerank + self-verify) is the E-021 baseline, so it is set
  # on EVERY arm below; the E-021 prompt flags are what each arm toggles.
  env -u TAOSMD_DECOMPOSE -u TAOSMD_SELF_VERIFY -u TAOSMD_RERANK \
      -u TAOSMD_SELF_CONSISTENCY -u TAOSMD_EVIDENCE_GROUNDING -u TAOSMD_PERSONA \
      TAOSMD_ASSEMBLE_TOKENS="${TAOSMD_ASSEMBLE_TOKENS:-4000}" \
      TAOSMD_RETRIEVE_LIMIT="${TAOSMD_RETRIEVE_LIMIT:-12}" \
      TAOSMD_FTS_LIMIT="${TAOSMD_FTS_LIMIT:-5}" \
      TAOSMD_CONTEXT_CHARS="${TAOSMD_CONTEXT_CHARS:-16000}" \
      TAOSMD_RERANK=1 TAOSMD_SELF_VERIFY=1 \
      "$@" \
      "$PY" "$RUNNER" --llm --limit "$LIMIT" 2>&1 | tee -a "$OUT"
  echo "" | tee -a "$OUT"
}

# E-021 arms. Baseline = rerank + self-verify, no E-021 prompt flag (its delta
# is zero by construction; it anchors the screen). Each other arm flips exactly
# one E-021 flag on top of that substrate.
run_arm "baseline"            # rerank + self-verify only
run_arm "self_consistency"    TAOSMD_SELF_CONSISTENCY=3
run_arm "evidence_grounding"  TAOSMD_EVIDENCE_GROUNDING=1
run_arm "persona"             TAOSMD_PERSONA=1

echo "E-021 ablation complete. Full log: $OUT" | tee -a "$OUT"
echo "Per-arm overall accuracy:" | tee -a "$OUT"
grep -E "ARM:|Overall:" "$OUT" | tee -a "$OUT"
