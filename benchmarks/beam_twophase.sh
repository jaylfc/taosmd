#!/usr/bin/env bash
# BEAM two-phase runner: generate with a BIG model, then judge separately so the
# two never co-reside in 12GB VRAM. Phase 1 loads only the generator and writes
# predictions; we `ollama stop` it to free VRAM; phase 2 loads only the judge and
# reads those predictions. This unblocks big generators (qwen3:14b, 9.3GB) and
# multi-judge runs that would otherwise deadlock a memory-tight GPU.
#
# Substrate matches beam_recall_sweep.sh / beam_campaign_resume.sh: arctic ONNX,
# cls pooling + arctic query prefix, tight retrieval (6+3, rerank top-4), pruned
# generation context, self-verify OFF. Judge = Qwen3-4B-Instruct.
#
# Usage: benchmarks/beam_twophase.sh <GEN_MODEL> [SPLIT] [LIMIT]   (defaults 100K 5)
#   GEN_MODEL  ollama tag for the generator, e.g. qwen3:14b
set -u

if [ "$#" -lt 1 ]; then
  echo "usage: benchmarks/beam_twophase.sh <GEN_MODEL> [SPLIT] [LIMIT]" >&2
  echo "  e.g. benchmarks/beam_twophase.sh qwen3:14b 100K 5" >&2
  exit 2
fi

GEN_MODEL="$1"
SPLIT="${2:-100K}"
LIMIT="${3:-5}"
PY=".venv/bin/python"
RUNNER="benchmarks/beam_runner.py"
JUDGE_MODEL="hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M"
ARC="$PWD/models/arctic-embed-s-onnx"

TS="$(date -u +%Y%m%d_%H%M%S)"
# Sanitise the model tag for a filename (slashes/colons to dashes).
GEN_SAFE="$(printf '%s' "$GEN_MODEL" | tr '/:' '--')"
PRED="benchmarks/results/beam_pred_${GEN_SAFE}_${TS}.json"
LOG="benchmarks/results/beam_twophase_${TS}.log"

# Shared tight-retrieval substrate, applied to BOTH phases via a wrapper.
run_phase () {
  local label="$1"; shift
  echo "==================== PHASE: $label (gen=$GEN_MODEL split=$SPLIT limit=$LIMIT) ====================" | tee -a "$LOG"
  env -u TAOSMD_SELF_VERIFY -u TAOSMD_DECOMPOSE -u TAOSMD_CHUNK_MODE \
      TAOSMD_ONNX_PATH="$ARC" TAOSMD_ONNX_POOLING=cls \
      TAOSMD_ONNX_QUERY_PREFIX="Represent this sentence for searching relevant passages: " \
      TAOSMD_ONNX_DOC_PREFIX="" TAOSMD_CHUNK_WORDS=100 \
      TAOSMD_RETRIEVE_LIMIT=6 TAOSMD_FTS_LIMIT=3 \
      TAOSMD_RERANK=1 TAOSMD_RERANK_TOP_K=4 \
      TAOSMD_ASSEMBLE_TOKENS=2000 TAOSMD_CONTEXT_CHARS=6000 \
      "$@" 2>&1 | tee -a "$LOG"
}

echo "BEAM two-phase: generator=$GEN_MODEL judge=$JUDGE_MODEL split=$SPLIT limit=$LIMIT" | tee -a "$LOG"
echo "  predictions -> $PRED" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Phase 1: GENERATE (only the big generator is loaded).
run_phase "1 GENERATE" \
  TAOSMD_OLLAMA_MODEL="$GEN_MODEL" \
  "$PY" "$RUNNER" --split "$SPLIT" --limit "$LIMIT" --llm --phase generate --predictions "$PRED"

# Free the generator's VRAM before the judge loads (avoids co-residency deadlock).
echo "" | tee -a "$LOG"
echo "-------------------- unloading generator: ollama stop $GEN_MODEL --------------------" | tee -a "$LOG"
ollama stop "$GEN_MODEL" 2>&1 | tee -a "$LOG" || true
echo "" | tee -a "$LOG"

# Phase 2: JUDGE (only the judge is loaded; reads the predictions file).
# --split/--limit are passed for consistency only; the judge phase reads $PRED.
run_phase "2 JUDGE" \
  TAOSMD_JUDGE_MODEL="$JUDGE_MODEL" \
  "$PY" "$RUNNER" --split "$SPLIT" --limit "$LIMIT" --llm --phase judge --predictions "$PRED"

echo "" | tee -a "$LOG"
echo "Two-phase run complete -> $LOG" | tee -a "$LOG"
echo "Overall:" | tee -a "$LOG"
grep -E "Overall:" "$LOG" | tail -1
