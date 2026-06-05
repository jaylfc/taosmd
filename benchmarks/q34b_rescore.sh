#!/usr/bin/env bash
# Strict-judge (qwen3:4b) rescore of already-run probe JSONs.
# Usage: q34b_rescore.sh <TS> <name1> [name2 ...]
# Reads benchmarks/results/<TS>_<name>.json, writes <...>.rescored_qwen34b.json,
# prints a Q34B VERDICT per config. ~50 min/config (3060 compute-bound).
set -u
cd "$(dirname "$0")/.."
TS="$1"; shift
OUT="benchmarks/results"
LOG="${OUT}/q34b_rescore_${TS}.log"
export TQDM_DISABLE=1

summary () {  # rescored.json  label
  python3 - "$1" "$2" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1])); rows=d.get("results",d)
except Exception as e:
    print(f"Q34B VERDICT {sys.argv[2]}: (no file: {e})"); sys.exit(0)
names={1:"Multi",2:"Temp",3:"Open",4:"Single"}
bycat={}; tot=[]
for r in rows:
    v=r.get("judge_rejudged")
    if v is None: continue
    tot.append(v); bycat.setdefault(int(r.get("category",0)),[]).append(v)
ov=sum(tot)/len(tot) if tot else 0.0
cats=" ".join(f"{names.get(c,c)}={sum(v)/len(v):.2f}" for c,v in sorted(bycat.items()))
print(f"Q34B VERDICT {sys.argv[2]}: n={len(tot)} overall={ov:.3f} | {cats}")
PY
}

echo "Q34B RESCORE start ${TS} :: $*" | tee -a "$LOG"
for name in "$@"; do
  j="${OUT}/${TS}_${name}.json"
  echo "[$(date +%H:%M:%S)] q34b rescoring ${name}" | tee -a "$LOG"
  python3 benchmarks/locomo_rescore.py "$j" --rejudge-model qwen3:4b >>"$LOG" 2>&1
  mv "${j%.json}.rescored.json" "${j%.json}.rescored_qwen34b.json" 2>/dev/null
  summary "${j%.json}.rescored_qwen34b.json" "$name" | tee -a "$LOG"
done
echo "Q34B RESCORE DONE ${TS}" | tee -a "$LOG"
