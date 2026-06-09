#!/usr/bin/env bash
# ColBERT model comparison probe — subset-200, gemma4:e2b rescore.
#
# Tests proper ColBERT models (trained with late-interaction objectives) against
# the MiniLM MaxSim baseline (which is NOT a ColBERT-trained model). Gate logic:
#   MiniLM MaxSim: gemma +0.06 confirmed x3, q34b +0.105. Gate passed.
#   These runs test whether a PROPER ColBERT model beats MiniLM MaxSim.
#
# Models probed:
#   answerai_small  answerdotai/answerai-colbert-small-v1  (~33M, 128-dim, ST-native)
#   colbertv2       colbert-ir/colbertv2.0                 (~110M, 128-dim, BERT-based)
#
# Uses --colbert-model which implies --late-interaction + --embed-mode local.
# GPU queue: generation (qwen3.5:9b) is GPU-bound; one at a time.
set -u
cd /home/jay/taosmd

TS="$(date +%Y%m%d_%H%M%S)"
OUT="benchmarks/results"
LOG="${OUT}/colbert_models_${TS}.log"
SUMMARY="${OUT}/colbert_models_${TS}_summary.tsv"
mkdir -p "$OUT"
export TQDM_DISABLE=1

BASE="--strategy vector-only --fusion mem0_additive --top-k 10 --adjacent-turns 2 --limit 200 --no-inline-judge --embed-mode local"

summary () {
  python3 - "$1" "$2" "$SUMMARY" <<'PY'
import json, sys
p, label, tsv = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    d = json.load(open(p)); rows = d.get("results", d)
except Exception as e:
    print(f"  VERDICT {label}: (no rescore file: {e})"); sys.exit(0)
names = {1: "Multi", 2: "Temp", 3: "Open", 4: "Single"}
bycat = {}; tot = []
for r in rows:
    v = r.get("judge_rejudged")
    if v is None: continue
    tot.append(v); bycat.setdefault(int(r.get("category", 0)), []).append(v)
ov = sum(tot) / len(tot) if tot else 0.0
cats = " ".join(f"{names.get(c,c)}={sum(v)/len(v):.2f}" for c, v in sorted(bycat.items()))
line = f"  VERDICT {label}: n={len(tot)} overall={ov:.3f} | {cats}"
print(line)
open(tsv, "a").write(f"{label}\t{len(tot)}\t{ov:.4f}\t{cats}\n")
PY
}

run () {
  local name="$1"; shift
  local json="${OUT}/${TS}_${name}.json"
  echo "===================================================================" | tee -a "$LOG"
  echo "[$(date +%H:%M:%S)] PROBE ${name} START :: $*" | tee -a "$LOG"
  python3 benchmarks/locomo_runner.py $BASE "$@" --out "$json" >>"$LOG" 2>&1
  echo "[$(date +%H:%M:%S)] PROBE ${name} run done -> rescoring (gemma4:e2b)" | tee -a "$LOG"
  python3 benchmarks/locomo_rescore.py "$json" --rejudge-model gemma4:e2b >>"$LOG" 2>&1
  mv "${json%.json}.rescored.json" "${json%.json}.rescored_gemma.json" 2>/dev/null
  summary "${json%.json}.rescored_gemma.json" "$name" | tee -a "$LOG"
  echo "[$(date +%H:%M:%S)] PROBE ${name} DONE" | tee -a "$LOG"
}

echo "COLBERT MODELS PROBE start ${TS}" | tee -a "$LOG"
echo "Baselines from prior runs: minilm_maxsim gemma=0.740 (+0.060 vs 0.680 baseline)" | tee -a "$LOG"

# Download models before GPU work starts (CPU-only, won't block Ollama)
echo "[$(date +%H:%M:%S)] Pre-fetching ColBERT models via huggingface_hub..." | tee -a "$LOG"
python3 - <<'PY' 2>&1 | tee -a "$LOG"
from huggingface_hub import snapshot_download
for repo in ["answerdotai/answerai-colbert-small-v1", "colbert-ir/colbertv2.0"]:
    try:
        path = snapshot_download(repo, ignore_patterns=["*.msgpack", "*.h5", "*.ot"])
        print(f"  [OK] {repo} -> {path}")
    except Exception as e:
        print(f"  [WARN] {repo}: {e}")
PY

# Probe 1: answerai-colbert-small-v1 (proper ST-native ColBERT, 33M, 128-dim)
run answerai_small \
  --model qwen3.5:9b --retrieval-top-k 20 \
  --colbert-model answerdotai/answerai-colbert-small-v1

# Probe 2: colbertv2.0 (Stanford ColBERT v2, 110M, 128-dim)
run colbertv2 \
  --model qwen3.5:9b --retrieval-top-k 20 \
  --colbert-model colbert-ir/colbertv2.0

echo "COLBERT MODELS PROBE DONE ${TS}" | tee -a "$LOG"
echo "SUMMARY FILE: ${SUMMARY}" | tee -a "$LOG"
echo "Compare: answerai_small + colbertv2 vs minilm_maxsim baseline 0.740" | tee -a "$LOG"
