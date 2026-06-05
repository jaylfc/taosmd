#!/usr/bin/env bash
# MiniLM MaxSim late-interaction probe (ColBERT feasibility) vs cosine baseline.
# Subset-200, gemma read per config. If MaxSim shows a lift, integrating a proper
# ColBERT model is justified; if flat/negative, late-interaction doesn't help at
# our tier with MiniLM and we stop before the big model integration.
# NOTE: MaxSim stores a token matrix per memory + a per-doc python scoring loop,
# so ingest + search are heavier/slower than cosine. Watch throughput.
set -u
cd "$(dirname "$0")/.."
TS="$(date +%Y%m%d_%H%M%S)"
OUT="benchmarks/results"
LOG="${OUT}/lateint_probe_${TS}.log"
mkdir -p "$OUT"
export TQDM_DISABLE=1

BASE="--strategy vector-only --fusion mem0_additive --top-k 10 --adjacent-turns 2 --embed-mode onnx --limit 200 --no-inline-judge --model qwen3.5:9b"

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
  echo "[$(date +%H:%M:%S)] LI ${name} START :: $*" | tee -a "$LOG"
  python3 benchmarks/locomo_runner.py $BASE "$@" --out "$json" >>"$LOG" 2>&1
  python3 benchmarks/locomo_rescore.py "$json" --rejudge-model gemma4:e2b >>"$LOG" 2>&1
  mv "${json%.json}.rescored.json" "${json%.json}.rescored_gemma.json" 2>/dev/null
  summary "${json%.json}.rescored_gemma.json" "$name" | tee -a "$LOG"
  echo "[$(date +%H:%M:%S)] LI ${name} DONE" | tee -a "$LOG"
}

echo "LATE-INTERACTION PROBE start ${TS}" | tee -a "$LOG"
run li_baseline  --retrieval-top-k 20
run li_maxsim    --retrieval-top-k 20 --late-interaction
echo "LATE-INTERACTION PROBE DONE ${TS}  (q34b on li_maxsim if it beats baseline)" | tee -a "$LOG"
