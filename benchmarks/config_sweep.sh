#!/usr/bin/env bash
# Config-diversity sweep: temperature x reasoning-effort x a few off-beat combos,
# layered on the confirmed rerank winner (rtk50 -> bge -> 20). The thesis (Jay):
# one slight off-the-wall change can flip results, so vary settings widely, not
# one lever at a time. Subset-200, gemma read per config; q34b on winners.
set -u
cd "$(dirname "$0")/.."
TS="$(date +%Y%m%d_%H%M%S)"
OUT="benchmarks/results"
LOG="${OUT}/config_sweep_${TS}.log"
mkdir -p "$OUT"
export TQDM_DISABLE=1

# Rerank-winner base (the +0.045 config) minus gen-temp / reasoning, which vary per run.
BASE="--strategy vector-only --fusion mem0_additive --top-k 20 --retrieval-top-k 50 --reranker bge-v2-m3 --adjacent-turns 2 --embed-mode onnx --limit 200 --no-inline-judge --model qwen3.5:9b"

summary () {
  python3 - "$1" "$2" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1])); rows=d.get("results",d)
except Exception as e:
    print(f"  GEMMA VERDICT {sys.argv[2]}: (no file: {e})"); sys.exit(0)
names={1:"Multi",2:"Temp",3:"Open",4:"Single"}
bycat={}; tot=[]
for r in rows:
    v=r.get("judge_rejudged")
    if v is None: continue
    tot.append(v); bycat.setdefault(int(r.get("category",0)),[]).append(v)
ov=sum(tot)/len(tot) if tot else 0.0
cats=" ".join(f"{names.get(c,c)}={sum(v)/len(v):.2f}" for c,v in sorted(bycat.items()))
print(f"  GEMMA VERDICT {sys.argv[2]}: n={len(tot)} overall={ov:.3f} | {cats}")
PY
}

run () {
  local name="$1"; shift
  local json="${OUT}/${TS}_${name}.json"
  echo "[$(date +%H:%M:%S)] CFG ${name} START :: $*" | tee -a "$LOG"
  python3 benchmarks/locomo_runner.py $BASE "$@" --out "$json" >>"$LOG" 2>&1
  python3 benchmarks/locomo_rescore.py "$json" --rejudge-model gemma4:e2b >>"$LOG" 2>&1
  mv "${json%.json}.rescored.json" "${json%.json}.rescored_gemma.json" 2>/dev/null
  summary "${json%.json}.rescored_gemma.json" "$name" | tee -a "$LOG"
  echo "[$(date +%H:%M:%S)] CFG ${name} DONE" | tee -a "$LOG"
}

echo "CONFIG SWEEP start ${TS}" | tee -a "$LOG"
# temperature axis (default rerank uses --no-think-prefix + gen-temp 0.2)
run c_t00_nothink  --gen-temp 0.0 --no-think-prefix
run c_t02_nothink  --gen-temp 0.2 --no-think-prefix
run c_t05_nothink  --gen-temp 0.5 --no-think-prefix
# reasoning axis: thinking ON (qwen3 chain-of-thought) layered on rerank — off-beat retry
run c_t02_thinking --gen-temp 0.2 --thinking-mode
# combo: rerank + binq + cold temp (cheap pool + precise rerank + deterministic gen)
run c_binq_t00     --gen-temp 0.0 --no-think-prefix --binary-quant
echo "CONFIG SWEEP DONE ${TS}  (q34b on winners)" | tee -a "$LOG"
