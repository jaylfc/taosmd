#!/usr/bin/env bash
# E-033 LoCoMo subset-200: small-model precision ladder (ternary PQ2_0 vs 1-bit Q1_0 vs Qwen3 base Q4_K_M), all via the prism llama-server.
# Cloned from the E-032 chain; only DIR, run-id prefix, the wait target and the arms change.
# Runner args + validity + scoring IDENTICAL to E-031 (run_chain.sh in 20260930-locomo-screen); only the
# generator backend changes (--llm-backend llama-server). Waits for the E-031 chain to exit first (one GPU job).
# Runner args VERBATIM from benchmarks/results/20260706-locomo-granite/run_chain.sh.
# Changes vs July: interpreter (.venv on the rebuilt host), arms, output dir,
# and the pre-registered validity checks (<90% real preds OR any <think => VOID).
set -u
cd /home/jay/taosmd || exit 1
DIR="/home/jay/taosmd/bench-logs/20261001-precision-ladder"
mkdir -p "$DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${DIR}/chain_${TS}.log"
PY="/home/jay/taosmd/.venv/bin/python"
OLLAMA="http://localhost:11434"
JUDGE="qwen3:4b"
export TQDM_DISABLE=1
log(){ echo "[$(date "+%F %T")] $*" | tee -a "$LOG"; }
unload(){ curl -s "${OLLAMA}/api/generate" -d "{\"model\":\"$1\",\"keep_alive\":0}" >/dev/null 2>&1 || true; sleep 5; }

B=/home/jay/llama-prism/b10743/llama-prism-b10743-adfffbe
export LD_LIBRARY_PATH="$(cat /home/jay/llama-prism/ldpath.txt):$B"
PORT=8091
SRV_PID=""
start_server(){  # $1 gguf path, rest = extra server flags
  local M="$1"; shift
  "$B/llama-server" -m "$M" -ngl 99 -c 16384 -t 6 --host 127.0.0.1 --port $PORT "$@" > "${DIR}/server_${CUR_TAG}.log" 2>&1 &
  SRV_PID=$!
  for i in $(seq 1 180); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && return 0; kill -0 $SRV_PID 2>/dev/null || break; sleep 2; done
  return 1
}
stop_server(){ [ -n "$SRV_PID" ] && { kill $SRV_PID 2>/dev/null; wait $SRV_PID 2>/dev/null; }; SRV_PID=""; sleep 5; }

run_arm(){
  local MODEL="$1" TAG="$2" SRVFLAGS="$3" RUNFLAGS="$4"
  CUR_TAG="$TAG"
  local OUTF="${DIR}/locomo_${TS}_${TAG}_s200.json"
  log "=== ARM ${TAG}: generation with ${MODEL} START ==="
  log "gguf sha256 ${MODEL}: $(sha256sum "$MODEL" | awk '{print $1}')"
  log "server flags: ${SRVFLAGS} | runner extra: ${RUNFLAGS}"
  # shellcheck disable=SC2086
  start_server "$MODEL" $SRVFLAGS || { log "ERROR: server failed to start for ${TAG}"; tail -20 "${DIR}/server_${TAG}.log" | tee -a "$LOG"; stop_server; echo "ARMRESULT ${TAG} VOID server-start" | tee -a "$LOG"; return 1; }
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee -a "$LOG"
  grep -iE "offload|CUDA0 model buffer|fit" "${DIR}/server_${TAG}.log" | head -5 | tee -a "$LOG"
  $PY -u benchmarks/locomo_runner.py \
      --model "$TAG" --llm-backend llama-server --llm-server-url "http://127.0.0.1:$PORT" $RUNFLAGS \
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
      --run-id "e033_${TAG}_s200" \
      --out "$OUTF" >>"$LOG" 2>&1 \
    || { log "ERROR: runner exited non-zero for ${TAG}"; echo "ARMRESULT ${TAG} VOID runner-nonzero" | tee -a "$LOG"; stop_server; return 1; }
  stop_server
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

log "waiting for E-032 chain (pid 1804020) to exit before touching the GPU"
while kill -0 1804020 2>/dev/null; do sleep 60; done
while pgrep -f "[l]ocomo_runner.py|[l]ocomo_rescore" >/dev/null; do sleep 60; done
log "E-032 gone; GPU now: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
log "repo at $(git log --oneline -1) | ollama $(ollama --version 2>&1 | tail -1)"
log "prism llama-server: $("$B/llama-server" --version 2>&1 | head -1)"
log "judge digest ${JUDGE}: $(ollama list | awk -v m="$JUDGE" '$1==m{print $2}')"
log "reranker onnx sha256: $(sha256sum models/bge-reranker-v2-m3-onnx/model.onnx models/bge-reranker-v2-m3-onnx/model.onnx_data | awk '{print $1}' | tr '\n' ' ')"
log "dataset sha256: $(sha256sum data/locomo/data/locomo10.json | awk '{print $1}')"
run_arm "/home/jay/models/Ternary-Bonsai-8B-gguf/Ternary-Bonsai-8B-PQ2_0.gguf" "C0_tbonsai_8b" "--reasoning off" "" || log "ARM C0 FAILED"
run_arm "/var/lib/ollama/blobs/sha256-a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f" "C1_qwen3_8b_q4km" "--reasoning off" "" || log "ARM C1 FAILED"
run_arm "/home/jay/models/Bonsai-8B-gguf/Bonsai-8B-Q1_0.gguf" "C2_bonsai_8b_1bit" "--reasoning off" "" || log "ARM C2 FAILED"
run_arm "/home/jay/models/Ternary-Bonsai-4B-gguf/Ternary-Bonsai-4B-PQ2_0.gguf" "C3_tbonsai_4b" "--reasoning off" "" || log "ARM C3 FAILED"
run_arm "/home/jay/models/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf" "C4_qwen3_4b_q4km" "--reasoning off" "" || log "ARM C4 FAILED"
run_arm "/home/jay/models/Bonsai-4B-gguf/Bonsai-4B-Q1_0.gguf" "C5_bonsai_4b_1bit" "--reasoning off" "" || log "ARM C5 FAILED"
run_arm "/home/jay/models/Ternary-Bonsai-1.7B-gguf/Ternary-Bonsai-1.7B-PQ2_0.gguf" "C6_tbonsai_1p7b" "--reasoning off" "" || log "ARM C6 FAILED"
run_arm "/var/lib/ollama/blobs/sha256-3d0b790534fe4b79525fc3692950408dca41171676ed7e21db57af5c65ef6ab6" "C7_qwen3_1p7b_q4km" "--reasoning off" "" || log "ARM C7 FAILED"
run_arm "/home/jay/models/Bonsai-1.7B-gguf/Bonsai-1.7B-Q1_0.gguf" "C8_bonsai_1p7b_1bit" "--reasoning off" "" || log "ARM C8 FAILED"
log "final cleanup: stopping all ollama models"
for M in $(ollama ps 2>/dev/null | awk "NR>1{print \$1}"); do ollama stop "$M" >/dev/null 2>&1 || true; done
sleep 5
log "final GPU: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
log "=== CHAIN COMPLETE ==="
