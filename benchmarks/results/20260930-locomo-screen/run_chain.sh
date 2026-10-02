#!/usr/bin/env bash
# E-031 LoCoMo subset-200 generator screen (pre-registered on PR #508).
# Runner args VERBATIM from benchmarks/results/20260706-locomo-granite/run_chain.sh.
# Changes vs July: interpreter (.venv on the rebuilt host), arms, output dir,
# and the pre-registered validity checks (<90% real preds OR any <think => VOID).
set -u
cd $HOME/taosmd || exit 1
DIR="$HOME/taosmd/bench-logs/20260930-locomo-screen"
mkdir -p "$DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${DIR}/chain_${TS}.log"
PY="$HOME/taosmd/.venv/bin/python"
OLLAMA="http://localhost:11434"
JUDGE="qwen3:4b"
export TQDM_DISABLE=1
log(){ echo "[$(date "+%F %T")] $*" | tee -a "$LOG"; }
unload(){ curl -s "${OLLAMA}/api/generate" -d "{\"model\":\"$1\",\"keep_alive\":0}" >/dev/null 2>&1 || true; sleep 5; }

run_arm(){
  local MODEL="$1" TAG="$2"
  local OUTF="${DIR}/locomo_${TS}_${TAG}_s200.json"
  log "=== ARM ${TAG}: generation with ${MODEL} START ==="
  log "digest ${MODEL}: $(ollama list | awk -v m="$MODEL" '$1==m{print $2}')"
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
      --run-id "e031_${TAG}_s200" \
      --out "$OUTF" >>"$LOG" 2>&1 \
    || { log "ERROR: runner exited non-zero for ${TAG}"; echo "ARMRESULT ${TAG} VOID runner-nonzero" | tee -a "$LOG"; unload "$MODEL"; return 1; }
  unload "$MODEL"
  $PY - "$OUTF" "$TAG" <<"PYEOF" 2>&1 | tee -a "$LOG"
import json, sys
d = json.load(open(sys.argv[1])); rows = d.get("results", [])
ok = [r for r in rows if r.get("predicted") and not r["predicted"].startswith("[generation_error")]
think = [r for r in rows if "<think" in (r.get("predicted") or "").lower()]
print(f"VALIDITY {sys.argv[2]}: rows={len(rows)} real_predictions={len(ok)} think_tags={len(think)}")
sys.exit(0 if (len(ok) >= max(1, int(len(rows) * 0.9)) and not think) else 1)
PYEOF
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    log "ERROR: ${TAG} failed validity (pre-registered VOID) -- not judging"
    echo "ARMRESULT ${TAG} VOID validity" | tee -a "$LOG"; return 1
  fi
  local RSF="${OUTF%.json}.rescored_q34b.json"
  log "ARM ${TAG}: rescore with external judge ${JUDGE}"
  $PY benchmarks/locomo_rescore_streaming.py "$OUTF" --judge-model "$JUDGE" --concurrency 2 --timeout 300 --out "$RSF" >>"$LOG" 2>&1 \
    || { log "ERROR: rescore failed for ${TAG}"; unload "$JUDGE"; echo "ARMRESULT ${TAG} VOID rescore-failed" | tee -a "$LOG"; return 1; }
  unload "$JUDGE"
  $PY - "$RSF" "$TAG" "$OUTF" <<"PYEOF" | tee -a "$LOG"
import json, sys
d = json.load(open(sys.argv[1])); rows = d.get("results", d)
g = json.load(open(sys.argv[3])); ov = g.get("overall", {})
allrows = g.get("results", [])
empty = sum(1 for r in allrows if not (r.get("predicted") or "").strip())
err = sum(1 for r in allrows if (r.get("predicted") or "").startswith("[generation_error"))
names = {1: "MultiHop", 2: "Temporal", 3: "OpenDomain", 4: "SingleHop"}
tot = []; bycat = {}
for r in rows:
    v = r.get("judge_rejudged")
    if v is None: continue
    tot.append(v); bycat.setdefault(int(r.get("category", 0)), []).append(v)
if not tot:
    print(f"ARMRESULT {sys.argv[2]} VOID no-judged-rows")
else:
    cats = " ".join(f"{names.get(c, c)}={sum(v)/len(v):.3f}(n={len(v)})" for c, v in sorted(bycat.items()))
    print(f"ARMRESULT {sys.argv[2]} n={len(tot)} overall={sum(tot)/len(tot):.4f} | {cats} | "
          f"lat_mean_ms={ov.get('mean_latency_ms')} ctx_tok={ov.get('mean_context_tokens')} empty={empty} err={err}")
PYEOF
  log "=== ARM ${TAG} DONE ==="
}

log "repo at $(git log --oneline -1) | ollama $(ollama --version 2>&1 | tail -1)"
log "judge digest ${JUDGE}: $(ollama list | awk -v m="$JUDGE" '$1==m{print $2}')"
log "reranker onnx sha256: $(sha256sum models/bge-reranker-v2-m3-onnx/model.onnx models/bge-reranker-v2-m3-onnx/model.onnx_data | awk '{print $1}' | tr '\n' ' ')"
log "dataset sha256: $(sha256sum data/locomo/data/locomo10.json | awk '{print $1}')"
run_arm "qwen3.5:9b" "A0_qwen35_9b" || log "ARM A0 FAILED"
run_arm "granite4.2:8b" "A1_granite42_8b" || log "ARM A1 FAILED"
run_arm "gemma4:12b-it-qat" "A2_gemma4_12b_qat" || log "ARM A2 FAILED"
run_arm "granite4.2:3b" "A3_granite42_3b" || log "ARM A3 FAILED"

log "final cleanup: stopping all ollama models"
for M in $(ollama ps 2>/dev/null | awk "NR>1{print \$1}"); do ollama stop "$M" >/dev/null 2>&1 || true; done
sleep 5
log "final GPU: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
log "=== CHAIN COMPLETE ==="
