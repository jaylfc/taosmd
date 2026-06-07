#!/usr/bin/env bash
# Creative "tweak every link" matrix on the confirmed rerank winner (rtk50->bge->top20).
# Subset-200, gemma read per config, NO --no-think-prefix (catastrophic on qwen3.5:9b).
# Tweaks: first-stage scorer (cosine/binq/MaxSim), generator, adjacency, temperature,
# and rerank-on-the-full-stack-leader. Winners escalate to q34b + full-1540.
set -u
cd "$(dirname "$0")/.."
TS="$(date +%Y%m%d_%H%M%S)"; OUT="benchmarks/results"; LOG="${OUT}/chain_tweak_${TS}.log"
mkdir -p "$OUT"; export TQDM_DISABLE=1
BASE="--strategy vector-only --fusion mem0_additive --top-k 20 --retrieval-top-k 50 --reranker bge-v2-m3 --adjacent-turns 2 --embed-mode onnx --limit 200 --no-inline-judge"
summary(){ python3 - "$1" "$2" <<'PY'
import json,sys
try: d=json.load(open(sys.argv[1])); rows=d.get("results",d)
except Exception as e: print(f"  VERDICT {sys.argv[2]}: (no file: {e})"); sys.exit(0)
names={1:"Multi",2:"Temp",3:"Open",4:"Single"}; bycat={}; tot=[]
for r in rows:
    v=r.get("judge_rejudged")
    if v is None: continue
    tot.append(v); bycat.setdefault(int(r.get("category",0)),[]).append(v)
ov=sum(tot)/len(tot) if tot else 0
cats=" ".join(f"{names.get(c,c)}={sum(v)/len(v):.2f}" for c,v in sorted(bycat.items()))
print(f"  VERDICT {sys.argv[2]}: n={len(tot)} overall={ov:.3f} | {cats}")
PY
}
run(){ local n="$1"; shift; local j="${OUT}/${TS}_${n}.json"
  echo "[$(date +%H:%M:%S)] CT ${n} START :: $*" | tee -a "$LOG"
  python3 benchmarks/locomo_runner.py $BASE "$@" --out "$j" >>"$LOG" 2>&1
  python3 benchmarks/locomo_rescore.py "$j" --rejudge-model gemma4:e2b >>"$LOG" 2>&1
  mv "${j%.json}.rescored.json" "${j%.json}.rescored_gemma.json" 2>/dev/null
  summary "${j%.json}.rescored_gemma.json" "$n" | tee -a "$LOG"
  echo "[$(date +%H:%M:%S)] CT ${n} DONE" | tee -a "$LOG"
}
echo "CHAIN TWEAK start ${TS}" | tee -a "$LOG"
run x_winner       --model qwen3.5:9b                                   # rerank winner anchor
run x_maxsim_rrk   --model qwen3.5:9b --late-interaction                # MaxSim first-stage -> cross-encoder (STACK two precision levers)
run x_binq_rrk     --model qwen3.5:9b --binary-quant                    # 32x-cheaper pool + rerank
run x_llama_rrk    --model llama3.1:8b                                  # cheaper/faster generator + rerank (does rerank lift the fast tier?)
run x_adj1         --model qwen3.5:9b --adjacent-turns 1                # less neighbour context post-rerank
run x_adj3         --model qwen3.5:9b --adjacent-turns 3                # more neighbour context post-rerank
run x_temp00       --model qwen3.5:9b --gen-temp 0.0                    # clean temperature sweep (no /no_think confound)
run x_temp07       --model qwen3.5:9b --gen-temp 0.7
run x_fullstack    --model qwen3.5:9b --multi-level-retrieval --llm-query-expansion --fusion rrf  # rerank ON TOP of the full-stack leader recipe
run x_fts5_rerank  --model qwen3.5:9b --fusion fts5_rrf                  # FTS5 lexical arm (zero-dep, no per-query rebuild) + rerank
run x_fts5_norrk   --model qwen3.5:9b --fusion fts5_rrf --reranker off --retrieval-top-k 20  # FTS5 fusion alone vs baseline (isolate the lexical swap)
echo "CHAIN TWEAK DONE ${TS}  (escalate winners to q34b + full-1540)" >> "$LOG"
echo "CHAIN_TWEAK_DONE" >> "$OUT/chain_tweak_done.marker"
