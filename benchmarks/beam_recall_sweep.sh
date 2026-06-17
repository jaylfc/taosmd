#!/usr/bin/env bash
# BEAM retrieval-recall push. Hypothesis: the 47% plateau is a RETRIEVAL-RECALL
# ceiling, not a generation ceiling. The tight config retrieves only ~6+3
# candidates, so the exact-fact needle is often never retrieved. Test WIDER
# retrieval (more candidates into the reranker) with a still-tight pruned
# generation context. Self-verify OFF throughout (the resume sweep's best on BEAM).
# Usage: benchmarks/beam_recall_sweep.sh [SPLIT] [LIMIT]   (defaults 100K 5)
set -u
SPLIT="${1:-100K}"; LIMIT="${2:-5}"
PY=".venv/bin/python"; RUNNER="benchmarks/beam_runner.py"
OUT="benchmarks/results/beam_recall_$(date -u +%Y%m%d_%H%M%S).log"
ARC="$PWD/models/arctic-embed-s-onnx"

run_cfg () {
  local name="$1"; shift
  echo "==================== ARM: $name (split=$SPLIT limit=$LIMIT) ====================" | tee -a "$OUT"
  env -u TAOSMD_SELF_VERIFY -u TAOSMD_DECOMPOSE -u TAOSMD_CHUNK_MODE \
      TAOSMD_ONNX_PATH="$ARC" TAOSMD_ONNX_POOLING=cls \
      TAOSMD_ONNX_QUERY_PREFIX="Represent this sentence for searching relevant passages: " \
      TAOSMD_ONNX_DOC_PREFIX="" TAOSMD_CHUNK_WORDS=100 \
      "$@" \
      "$PY" "$RUNNER" --split "$SPLIT" --limit "$LIMIT" --llm 2>&1 | grep -E "Overall:|^    [a-z_]+ +[0-9]+/" | tee -a "$OUT"
  echo "" | tee -a "$OUT"
}

# Anchor = the resume sweep's best nosv config (47% pass / avg 0.455).
run_cfg "anchor_nosv"  TAOSMD_RETRIEVE_LIMIT=6  TAOSMD_FTS_LIMIT=3  TAOSMD_RERANK=1 TAOSMD_RERANK_TOP_K=4 TAOSMD_ASSEMBLE_TOKENS=2000 TAOSMD_CONTEXT_CHARS=6000
# Wide recall, hard prune: many candidates for the reranker, tight gen context.
run_cfg "recall_wide"  TAOSMD_RETRIEVE_LIMIT=40 TAOSMD_FTS_LIMIT=15 TAOSMD_RERANK=1 TAOSMD_RERANK_TOP_K=6 TAOSMD_ASSEMBLE_TOKENS=2000 TAOSMD_CONTEXT_CHARS=6000
run_cfg "recall_wider" TAOSMD_RETRIEVE_LIMIT=60 TAOSMD_FTS_LIMIT=20 TAOSMD_RERANK=1 TAOSMD_RERANK_TOP_K=8 TAOSMD_ASSEMBLE_TOKENS=2400 TAOSMD_CONTEXT_CHARS=7000
# FTS-heavy: BEAM needles are exact facts; lexical BM25 may surface what dense misses.
run_cfg "fts_heavy"    TAOSMD_RETRIEVE_LIMIT=12 TAOSMD_FTS_LIMIT=25 TAOSMD_RERANK=1 TAOSMD_RERANK_TOP_K=6 TAOSMD_ASSEMBLE_TOKENS=2000 TAOSMD_CONTEXT_CHARS=6000

echo "Recall sweep complete -> $OUT" | tee -a "$OUT"
echo "Per-arm overall:" | tee -a "$OUT"
grep -E "ARM:|Overall:" "$OUT" | tee -a "$OUT"
