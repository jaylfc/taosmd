#!/usr/bin/env bash
# Full-1540 confirmation of the cross-encoder rerank win (subset-200: q34b +0.045).
# Runs baseline + rerank on the FULL set with a fast gemma rescore per config for
# a quick read. Strict q34b rescore is a separate ~5h/config step (q34b_rescore.sh)
# run on the survivors afterward.
set -u
cd "$(dirname "$0")/.."
TS="$(date +%Y%m%d_%H%M%S)"
OUT="benchmarks/results"
LOG="${OUT}/full1540_rerank_${TS}.log"
mkdir -p "$OUT"
export TQDM_DISABLE=1

BASE="--strategy vector-only --fusion mem0_additive --top-k 10 --adjacent-turns 2 --embed-mode onnx --limit 0 --no-inline-judge"

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
  echo "[$(date +%H:%M:%S)] FULL PROBE ${name} START :: $*" | tee -a "$LOG"
  python3 benchmarks/locomo_runner.py $BASE "$@" --out "$json" >>"$LOG" 2>&1
  echo "[$(date +%H:%M:%S)] FULL PROBE ${name} run done -> gemma rescore" | tee -a "$LOG"
  python3 benchmarks/locomo_rescore.py "$json" --rejudge-model gemma4:e2b >>"$LOG" 2>&1
  mv "${json%.json}.rescored.json" "${json%.json}.rescored_gemma.json" 2>/dev/null
  summary "${json%.json}.rescored_gemma.json" "$name" | tee -a "$LOG"
  echo "[$(date +%H:%M:%S)] FULL PROBE ${name} DONE (q34b rescore separately)" | tee -a "$LOG"
}

echo "FULL-1540 RERANK CONFIRM start ${TS}" | tee -a "$LOG"
run full_baseline --model qwen3.5:9b --retrieval-top-k 20
run full_rerank   --model qwen3.5:9b --retrieval-top-k 50 --top-k 20 --reranker bge-v2-m3
echo "FULL-1540 RERANK CONFIRM DONE ${TS}  (next: q34b_rescore.sh ${TS} full_baseline full_rerank)" | tee -a "$LOG"
