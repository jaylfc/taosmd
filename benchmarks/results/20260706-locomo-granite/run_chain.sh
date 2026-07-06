#!/usr/bin/env bash
# LoCoMo generator-comparison probe: granite4.1:8b vs qwen3.5:9b baseline
# Shipped 12GB config (maxsim-rerank-9b equivalents), external qwen3:4b judge.
set -u
cd /home/jay/taosmd || exit 1
DIR="/home/jay/taosmd/bench-logs/20260706-locomo-granite"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${DIR}/chain_${TS}.log"
PY="python3.14"
OLLAMA="http://localhost:11434"
JUDGE="qwen3:4b"
export TQDM_DISABLE=1
log(){ echo "[$(date "+%F %T")] $*" | tee -a "$LOG"; }
unload(){ curl -s "${OLLAMA}/api/generate" -d "{\"model\":\"$1\",\"keep_alive\":0}" >/dev/null 2>&1 || true; sleep 5; }

run_arm(){
  local MODEL="$1" TAG="$2"
  local OUTF="${DIR}/locomo_${TS}_${TAG}_s200.json"
  log "=== ARM ${TAG}: generation with ${MODEL} START ==="
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee -a "$LOG"
  $PY -u benchmarks/locomo_runner.py \
      --model "$MODEL" \
      --strategy vector-only \
      --fusion mem0_additive \
      --top-k 10 \
      --retrieval-top-k 50 \
      --adjacent-turns 2 \
      --reranker bge-v2-m3 \
      --embed-mode onnx \
      --limit 200 \
      --no-inline-judge \
      --concurrency 1 \
      --timeout 600 \
      --run-id "granite_probe_${TAG}_s200" \
      --out "$OUTF" >>"$LOG" 2>&1 \
    || { log "ERROR: runner exited non-zero for ${TAG}"; return 1; }
  unload "$MODEL"
  $PY - "$OUTF" <<"PYEOF" >>"$LOG" 2>&1 || { log "ERROR: ${TAG} zero/low real predictions -- not judging"; return 1; }
import json, sys
d = json.load(open(sys.argv[1])); rows = d.get("results", [])
ok = [r for r in rows if r.get("predicted") and not r["predicted"].startswith("[generation_error")]
print(f"rows={len(rows)} real_predictions={len(ok)}")
sys.exit(0 if len(ok) >= max(1, int(len(rows) * 0.9)) else 1)
PYEOF
  local RSF="${OUTF%.json}.rescored_q34b.json"
  log "ARM ${TAG}: rescore with external judge ${JUDGE}"
  $PY benchmarks/locomo_rescore_streaming.py "$OUTF" --judge-model "$JUDGE" --concurrency 2 --timeout 300 --out "$RSF" >>"$LOG" 2>&1 \
    || { log "ERROR: rescore failed for ${TAG}"; unload "$JUDGE"; return 1; }
  unload "$JUDGE"
  $PY - "$RSF" "$TAG" <<"PYEOF" | tee -a "$LOG"
import json, sys
d = json.load(open(sys.argv[1])); rows = d.get("results", d)
names = {1: "MultiHop", 2: "Temporal", 3: "OpenDomain", 4: "SingleHop"}
tot = []; bycat = {}
for r in rows:
    v = r.get("judge_rejudged")
    if v is None: continue
    tot.append(v); bycat.setdefault(int(r.get("category", 0)), []).append(v)
if not tot:
    print(f"VERDICT {sys.argv[2]}: NO JUDGED ROWS")
else:
    cats = " ".join(f"{names.get(c, c)}={sum(v)/len(v):.3f}(n={len(v)})" for c, v in sorted(bycat.items()))
    print(f"VERDICT {sys.argv[2]}: n={len(tot)} overall={sum(tot)/len(tot):.4f} | {cats}")
PYEOF
  log "=== ARM ${TAG} DONE ==="
}

log "repo at $(git log --oneline -1)"
run_arm "granite4.1:8b" "granite41_8b" || log "ARM granite41_8b FAILED"
run_arm "qwen3.5:9b" "qwen35_9b" || log "ARM qwen35_9b FAILED"

log "final cleanup: stopping all ollama models"
for M in $(ollama ps 2>/dev/null | awk "NR>1{print \$1}"); do ollama stop "$M" >/dev/null 2>&1 || true; done
sleep 5
log "final GPU: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
log "=== CHAIN COMPLETE ==="
