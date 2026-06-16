#!/usr/bin/env bash
# BEAM score-up campaign. Sweeps configs SEQUENTIALLY (one GPU job at a time) on
# BEAM-100K to raise the score past the tight-retrieval baseline (47%).
# Each arm resets the levers, applies its own, and logs overall + per-type.
#
# Usage: benchmarks/beam_campaign.sh [SPLIT] [LIMIT]
#   SPLIT default 100K, LIMIT default 5 (100 questions per arm, a fast sweep).
# Required env (export before running): TAOSMD_OLLAMA_MODEL, TAOSMD_JUDGE_MODEL.
# Fixed substrate per arm: arctic embedder + tight retrieval (the settled winner).
set -u
SPLIT="${1:-100K}"
LIMIT="${2:-5}"
PY=".venv/bin/python"
RUNNER="benchmarks/beam_runner.py"
OUT="benchmarks/results/beam_campaign_$(date -u +%Y%m%d_%H%M%S).log"
ARC="$PWD/models/arctic-embed-s-onnx"

run_cfg () {
  local name="$1"; shift
  echo "==================== ARM: $name (split=$SPLIT limit=$LIMIT) ====================" | tee -a "$OUT"
  env -u TAOSMD_SELF_VERIFY -u TAOSMD_DECOMPOSE -u TAOSMD_CHUNK_MODE \
      TAOSMD_ONNX_PATH="$ARC" TAOSMD_ONNX_POOLING=cls \
      TAOSMD_ONNX_QUERY_PREFIX="Represent this sentence for searching relevant passages: " \
      TAOSMD_ONNX_DOC_PREFIX="" \
      TAOSMD_RETRIEVE_LIMIT="${TAOSMD_RETRIEVE_LIMIT:-6}" TAOSMD_FTS_LIMIT="${TAOSMD_FTS_LIMIT:-3}" \
      TAOSMD_RERANK="${TAOSMD_RERANK:-1}" TAOSMD_RERANK_TOP_K="${TAOSMD_RERANK_TOP_K:-4}" \
      TAOSMD_ASSEMBLE_TOKENS="${TAOSMD_ASSEMBLE_TOKENS:-2000}" TAOSMD_CONTEXT_CHARS="${TAOSMD_CONTEXT_CHARS:-6000}" \
      "$@" \
      "$PY" "$RUNNER" --split "$SPLIT" --limit "$LIMIT" --llm 2>&1 | grep -E "Overall:|^    [a-z_]+ +[0-9]+/" | tee -a "$OUT"
  echo "" | tee -a "$OUT"
}

# Base reference = tight retrieval + self-verify + chunk100 (the 47% config).
run_cfg "base_chunk100_sv"      TAOSMD_SELF_VERIFY=1 TAOSMD_CHUNK_WORDS=100
# Chunk-size sweep (does bigger/smaller chunking surface the needle better?).
run_cfg "chunk50_sv"            TAOSMD_SELF_VERIFY=1 TAOSMD_CHUNK_WORDS=50
run_cfg "chunk200_sv"          TAOSMD_SELF_VERIFY=1 TAOSMD_CHUNK_WORDS=200
run_cfg "chunk300_sv"          TAOSMD_SELF_VERIFY=1 TAOSMD_CHUNK_WORDS=300
# Sentence-aware chunking (keep facts intact, no mid-sentence splits).
run_cfg "sentences_sv"         TAOSMD_SELF_VERIFY=1 TAOSMD_CHUNK_MODE=sentences TAOSMD_CHUNK_WORDS=120
# Self-verify OFF (does the CoVe pass help or over-abstain on BEAM?).
run_cfg "chunk100_nosv"        TAOSMD_CHUNK_WORDS=100
# Decompose ON (multi-hop split for the reasoning categories).
run_cfg "chunk100_sv_decomp"   TAOSMD_SELF_VERIFY=1 TAOSMD_DECOMPOSE=1 TAOSMD_CHUNK_WORDS=100

echo "Campaign complete -> $OUT" | tee -a "$OUT"
echo "Per-arm overall:" | tee -a "$OUT"
grep -E "ARM:|Overall:" "$OUT" | tee -a "$OUT"
