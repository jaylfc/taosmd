#!/usr/bin/env bash
# E-010 embedding bake-off (judge-free R@K screen). CPU-only, no GPU, no servers,
# no Docker, no system changes -> safe to run on the VPS alongside Coolify.
# For each candidate embedder: fetch its ONNX, set the correct pooling + prefix
# via the env-configurable loader (TAOSMD_ONNX_POOLING / _QUERY_PREFIX / _DOC_PREFIX),
# run the retrieval R@K probe on a LoCoMo subset, collect R@K. Compares against
# the shipped arctic-embed-s default.
#
# Usage: bash benchmarks/e010_bakeoff.sh [conversations]   (default 10)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PY="${PY:-/root/bench-venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
CONV="${1:-10}"
DATASET="${LOCOMO_DATASET:-data/locomo/data/locomo10.json}"
OUTDIR="benchmarks/results"; mkdir -p "$OUTDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="$OUTDIR/e010_bakeoff_${STAMP}.txt"
ARCTIC_PREFIX="Represent this sentence for searching relevant passages: "

# Ensure huggingface_hub is available in the venv (isolated, no system touch).
"$PY" -c "import huggingface_hub" 2>/dev/null || "$PY" -m pip install -q huggingface_hub 2>/dev/null

fetch_model() {  # $1=repo  $2=dir
  "$PY" - "$1" "$2" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
repo, dir = sys.argv[1], sys.argv[2]
snapshot_download(repo, local_dir=dir, allow_patterns=[
    "onnx/model.onnx", "model.onnx", "*.json", "tokenizer*", "vocab*", "1_Pooling/*"])
print("fetched", repo)
PYEOF
}

# name | hf_repo | pooling | query_prefix | doc_prefix
CANDIDATES=(
  "arctic-s|Snowflake/snowflake-arctic-embed-s|cls|${ARCTIC_PREFIX}|"
  "bge-small|BAAI/bge-small-en-v1.5|cls|${ARCTIC_PREFIX}|"
  "gte-small|thenlper/gte-small|mean||"
  "e5-small-v2|intfloat/e5-small-v2|mean|query: |passage: "
  "granite-small-r2|onnx-community/granite-embedding-small-english-r2-ONNX|cls||"
)

echo "E-010 embedding bake-off  $(date -u)  conv=$CONV dataset=$DATASET" | tee "$SUMMARY"
echo "model                R@K     n_evid" | tee -a "$SUMMARY"

for entry in "${CANDIDATES[@]}"; do
  IFS='|' read -r name repo pooling qpref dpref <<< "$entry"
  dir="models/e010-$name"
  if [ ! -f "$dir/onnx/model.onnx" ] && [ ! -f "$dir/model.onnx" ]; then
    echo "[fetch] $name <- $repo" | tee -a "$SUMMARY"
    fetch_model "$repo" "$dir" >>"$OUTDIR/e010_${name}_fetch.err" 2>&1
  fi
  if [ ! -f "$dir/onnx/model.onnx" ] && [ ! -f "$dir/model.onnx" ]; then
    echo "$(printf '%-20s' "$name") FETCH_FAILED (no model.onnx)" | tee -a "$SUMMARY"; continue
  fi
  out="$OUTDIR/e010_${name}_${STAMP}.json"
  # The loader env overrides take precedence over name detection.
  R=$(TAOSMD_ONNX_POOLING="$pooling" \
      TAOSMD_ONNX_QUERY_PREFIX="$qpref" \
      TAOSMD_ONNX_DOC_PREFIX="$dpref" \
      "$PY" benchmarks/retrieval_latency_probe.py \
        --dataset "$DATASET" --conversations "$CONV" \
        --embed-mode onnx --onnx-path "$dir" --reranker off \
        --out "$out" 2>>"$OUTDIR/e010_${name}_${STAMP}.err" \
      | grep -E "^Overall|R@K" | head -1)
  rk=$("$PY" -c "import json,sys; d=json.load(open('$out')); print(d.get('r_at_k', d.get('overall',{}).get('r_at_k','?')))" 2>/dev/null || echo "?")
  echo "$(printf '%-20s' "$name") $(printf '%-7s' "$rk")" | tee -a "$SUMMARY"
done

echo "" | tee -a "$SUMMARY"
echo "done. summary: $SUMMARY" | tee -a "$SUMMARY"
