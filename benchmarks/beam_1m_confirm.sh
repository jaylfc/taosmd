#!/usr/bin/env bash
# Carry the BEAM-100K sweep winner to a firmer BEAM-1M number. Same arctic +
# tight-retrieval substrate as beam_campaign.sh; the winning chunk/self-verify
# combo comes in as args so there is no GPU idle between the sweep ending and
# this run starting.
#
# Usage: benchmarks/beam_1m_confirm.sh [CHUNK_WORDS] [CHUNK_MODE] [SELF_VERIFY] [LIMIT]
#   CHUNK_WORDS  default 100   (the winning arm's chunk size)
#   CHUNK_MODE   default words (or "sentences")
#   SELF_VERIFY  default 1     (1 on, 0 off, per the winning arm)
#   LIMIT        default 10    (10 convs = 200 Qs, apples-to-apples with the n=200 baseline; bump to 20 for n=400)
# Required env: TAOSMD_OLLAMA_MODEL, TAOSMD_JUDGE_MODEL.
set -u
CW="${1:-100}"
CM="${2:-words}"
SV="${3:-1}"
LIMIT="${4:-10}"
PY=".venv/bin/python"
RUNNER="benchmarks/beam_runner.py"
ARC="$PWD/models/arctic-embed-s-onnx"
OUT="benchmarks/results/beam_1m_confirm_$(date -u +%Y%m%d_%H%M%S).log"

echo "==================== BEAM-1M CONFIRM: chunk=$CW mode=$CM self_verify=$SV limit=$LIMIT ====================" | tee -a "$OUT"
env -u TAOSMD_DECOMPOSE \
    TAOSMD_SELF_VERIFY="$SV" TAOSMD_CHUNK_WORDS="$CW" TAOSMD_CHUNK_MODE="$CM" \
    TAOSMD_ONNX_PATH="$ARC" TAOSMD_ONNX_POOLING=cls \
    TAOSMD_ONNX_QUERY_PREFIX="Represent this sentence for searching relevant passages: " \
    TAOSMD_ONNX_DOC_PREFIX="" \
    TAOSMD_RETRIEVE_LIMIT=6 TAOSMD_FTS_LIMIT=3 \
    TAOSMD_RERANK=1 TAOSMD_RERANK_TOP_K=4 \
    TAOSMD_ASSEMBLE_TOKENS=2000 TAOSMD_CONTEXT_CHARS=6000 \
    "$PY" "$RUNNER" --split 1M --limit "$LIMIT" --llm 2>&1 | tee -a "$OUT"

echo "BEAM-1M confirm complete -> $OUT" | tee -a "$OUT"
