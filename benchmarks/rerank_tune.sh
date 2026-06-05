#!/usr/bin/env bash
# Tune the confirmed cross-encoder rerank win (subset-200, gemma read per config).
# Builds on rerank_only (rtk50->bge->20, q34b +0.045). Varies pool width, final
# top-k, and reranker model. baseline + rerank_only are the anchors (already run
# in chain 20260605_124404, but re-run here so all configs share one TS/commit).
set -u
cd "$(dirname "$0")/.."
TS="$(date +%Y%m%d_%H%M%S)"
OUT="benchmarks/results"
LOG="${OUT}/rerank_tune_${TS}.log"
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
  echo "[$(date +%H:%M:%S)] TUNE ${name} START :: $*" | tee -a "$LOG"
  python3 benchmarks/locomo_runner.py $BASE "$@" --out "$json" >>"$LOG" 2>&1
  python3 benchmarks/locomo_rescore.py "$json" --rejudge-model gemma4:e2b >>"$LOG" 2>&1
  mv "${json%.json}.rescored.json" "${json%.json}.rescored_gemma.json" 2>/dev/null
  summary "${json%.json}.rescored_gemma.json" "$name" | tee -a "$LOG"
  echo "[$(date +%H:%M:%S)] TUNE ${name} DONE" | tee -a "$LOG"
}

echo "RERANK TUNE start ${TS}" | tee -a "$LOG"
run t_baseline      --retrieval-top-k 20
run t_rerank50_20   --retrieval-top-k 50  --top-k 20 --reranker bge-v2-m3   # the confirmed winner
run t_rerank100_20  --retrieval-top-k 100 --top-k 20 --reranker bge-v2-m3   # wider pool
run t_rerank50_10   --retrieval-top-k 50  --top-k 10 --reranker bge-v2-m3   # narrower final ctx
run t_rerank50_30   --retrieval-top-k 50  --top-k 30 --reranker bge-v2-m3   # wider final ctx
run t_rerank_msmarco --retrieval-top-k 50 --top-k 20 --reranker ms-marco    # different reranker model
echo "RERANK TUNE DONE ${TS}  (q34b on winners next)" | tee -a "$LOG"
