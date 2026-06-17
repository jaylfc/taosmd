#!/usr/bin/env bash
# BEAM score-up campaign RESUME: the 5 arms still pending after the Jun 16 GPU pause.
# Already done (both 47.0%, skipped here): base_chunk100_sv, chunk50_sv.
# Same substrate as beam_campaign.sh (arctic + tight retrieval). One GPU job at a time.
# Usage: benchmarks/beam_campaign_resume.sh [SPLIT] [LIMIT]   (defaults 100K 5)
set -u
SPLIT="${1:-100K}"
LIMIT="${2:-5}"
PY=".venv/bin/python"
RUNNER="benchmarks/beam_runner.py"
OUT="benchmarks/results/beam_campaign_resume_$(date -u +%Y%m%d_%H%M%S).log"
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

run_cfg "chunk200_sv"          TAOSMD_SELF_VERIFY=1 TAOSMD_CHUNK_WORDS=200
run_cfg "chunk300_sv"          TAOSMD_SELF_VERIFY=1 TAOSMD_CHUNK_WORDS=300
run_cfg "sentences_sv"         TAOSMD_SELF_VERIFY=1 TAOSMD_CHUNK_MODE=sentences TAOSMD_CHUNK_WORDS=120
run_cfg "chunk100_nosv"        TAOSMD_CHUNK_WORDS=100
run_cfg "chunk100_sv_decomp"   TAOSMD_SELF_VERIFY=1 TAOSMD_DECOMPOSE=1 TAOSMD_CHUNK_WORDS=100

echo "Resume campaign complete -> $OUT" | tee -a "$OUT"
echo "Per-arm overall:" | tee -a "$OUT"
grep -E "ARM:|Overall:" "$OUT" | tee -a "$OUT"
