#!/usr/bin/env bash
# Retrieval-lever probe chain (subset-200, per-probe verdict, early-abortable).
#
# Methodology = the validated clag/binq one: run subset-200 on the vector-only
# leader recipe, evaluate by the EXTERNAL judge rescore (NOT the inline self-
# judge, which is a lenient extra generation we skip via --no-inline-judge).
# Recipe matches the binq baseline EXACTLY (no --no-think-prefix) so deltas are
# comparable to 0.520 q34b / 0.694 gemma.
#
# For speed of feedback we rescore each probe with the FAST lenient judge
# (gemma4:e2b, ~10 min) right after it runs and print an overall+per-category
# verdict. The slow strict judge (qwen3:4b, ~50 min) is run separately on
# survivors only, then survivors escalate to full-1540.
#
# Probes run serially (GPU-contention rule). Watch the "VERDICT" lines; kill the
# chain if a lever is a clear dud relative to baseline.
set -u
cd "$(dirname "$0")/.."

TS="$(date +%Y%m%d_%H%M%S)"
OUT="benchmarks/results"
LOG="${OUT}/retrieval_probes_${TS}.log"
mkdir -p "$OUT"
export TQDM_DISABLE=1   # silence bm25s/spaCy progress bars in the log

# Vector-only leader recipe (the binq baseline), shared across probes.
BASE="--strategy vector-only --fusion mem0_additive --top-k 10 --adjacent-turns 2 --embed-mode onnx --limit 200 --no-inline-judge"

summary () {  # rescored.json  label
  python3 - "$1" "$2" <<'PY'
import json,sys
p,label=sys.argv[1],sys.argv[2]
try:
    d=json.load(open(p)); rows=d.get("results",d)
except Exception as e:
    print(f"  VERDICT {label}: (no rescore file: {e})"); sys.exit(0)
names={1:"Multi",2:"Temp",3:"Open",4:"Single"}
bycat={}; tot=[]
for r in rows:
    v=r.get("judge_rejudged")
    if v is None: continue
    tot.append(v); bycat.setdefault(int(r.get("category",0)),[]).append(v)
ov=sum(tot)/len(tot) if tot else 0.0
cats=" ".join(f"{names.get(c,c)}={sum(v)/len(v):.2f}" for c,v in sorted(bycat.items()))
print(f"  VERDICT {label}: n={len(tot)} overall={ov:.3f} | {cats}")
PY
}

run () {  # name  extra-args...
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

echo "RETRIEVAL PROBE CHAIN start ${TS}" | tee -a "$LOG"

run baseline    --model qwen3.5:9b --retrieval-top-k 20
run rerank_only --model qwen3.5:9b --retrieval-top-k 50 --reranker bge-v2-m3
run binq_rerank --model qwen3.5:9b --retrieval-top-k 50 --reranker bge-v2-m3 --binary-quant
run bm25_lemma  --model qwen3.5:9b --retrieval-top-k 20 --fusion bm25_lemma_rrf
run gen_35b     --model qwen3.6:35b-a3b-q4_K_M --retrieval-top-k 20

echo "RETRIEVAL PROBE CHAIN DONE ${TS}  (gemma reads above; run q34b rescore on survivors, then full-1540)" | tee -a "$LOG"
