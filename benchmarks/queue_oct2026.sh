#!/usr/bin/env bash
# =============================================================================
# benchmarks/queue_oct2026.sh: the Tier A chain of the October 2026 experiment
# queue (docs/research-report.md, section 6.x, E-038 and E-039).
#
# E-038  granite4.2:8b settings sweep on LoCoMo subset-200, external qwen3:4b
#        judge. Phase "gen": --gen-temp {0.0, 0.2, 0.7} x --repeat-penalty
#        {1.0, 1.1}, six arms. Phase "retrieval": --adjacent-turns {1, 2, 3} x
#        --top-k {5, 10, 15}, nine arms, run AT ONE generation setting that the
#        operator passes explicitly after reading the gen phase (default: the
#        shipped 0.2 / 1.0). The script never auto-picks "the best cell": the
#        pre-registered kill rule says that when no cell clears +0.02 the result
#        is "within noise" and the shipped config stays, so the retrieval phase
#        defaults to the shipped setting unless the operator names another.
#        The (0.2, 1.0, adjacent 2, top-k 10) cell is the fresh same-chain
#        anchor; it appears once in each phase, which gives a within-chain
#        replicate for the noise band.
#
# E-039  Generator screen: G0 qwen3.5:9b at the pinned control digest
#        6488c96fa5fa, G1 qwen3.5:9b at the re-pushed upstream digest, G2
#        Mellum2.1-12B-A2.5B-Thinking Q4 (thinking off), G3 granite4.2:8b as
#        the second control. G3 is skipped when the E-038 anchor cell has
#        already been judged in the same OUTDIR (same host, same chain), and
#        the log says so.
#
# Instrument: the E-031 arguments verbatim
# (benchmarks/results/20260930-locomo-screen/run_chain.sh), one arm at a time
# on a shared 12 GB GPU: generate, unload the generator, rescore with the
# external judge, unload the judge. Every arm records `ollama --version`,
# /api/version and the /api/tags digest of every model it used, in the chain
# log and in a per-arm .provenance.json sidecar (card tsk-xl5chj puts the same
# facts inside the results JSON once it lands; until then the sidecar is the
# record).
#
# Validity, pre-registered: an arm is VOID, not a score, if fewer than 90
# percent of its rows carry a real prediction or if any prediction contains a
# `<think` tag (E-031 rule). Pause flag: the runner itself checks
# /tmp/taosmd-bench-pause between conversations (--ckpt); this chain also
# checks it BEFORE every arm and exits 3 so a 3060 lease can end cleanly.
# Re-running the same phase skips every arm whose rescored JSON already holds
# judged rows, so a paused chain resumes where it stopped.
#
# Usage:
#   benchmarks/queue_oct2026.sh <gen|retrieval|screen|all> [--dry-run]
#       [--gen-temp T] [--repeat-penalty P]      (retrieval phase setting)
#
# Environment (the operator fills these on the bench host):
#   REPO, DATASET, OUTDIR            required (dry-run tolerates missing paths)
#   QUEUE_GPU_CLAIM_MSG              required: the bus id or text of the
#                                    [GPU CLAIM] you posted (GPU lease protocol)
#   OLLAMA                           default http://localhost:11434
#   OLLAMA_CONTEXT_LENGTH            default 8192 (the runner has no --num-ctx)
#   PAUSE_FLAG                       default /tmp/taosmd-bench-pause
#   BACKED_UP_TSV                    default ~/.taos-team/backed-up.tsv
#   MELLUM_MODEL                     default mellum2.1-12b-a2.5b-thinking:q4
#   QUEUE_ALLOW_PULL=1               lets G1 run `ollama pull qwen3.5:9b`; it
#                                    is refused unless the control blob is
#                                    already copied to CONTROL_TAG AND that tag
#                                    has a row in BACKED_UP_TSV (model-backup
#                                    rule: the pull overwrites the control blob)
# =============================================================================
set -u

PHASE="${1:-}"
shift || true
DRY_RUN=0
RET_GEN_TEMP="0.2"
RET_REPEAT_PENALTY="1.0"
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --gen-temp) RET_GEN_TEMP="$2"; shift 2 ;;
    --repeat-penalty) RET_REPEAT_PENALTY="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$PHASE" in
  gen|retrieval|screen|all) ;;
  *) echo "usage: $0 <gen|retrieval|screen|all> [--dry-run] [--gen-temp T] [--repeat-penalty P]" >&2; exit 2 ;;
esac

REPO="${REPO:-}"
DATASET="${DATASET:-}"
OUTDIR="${OUTDIR:-}"
OLLAMA="${OLLAMA:-http://localhost:11434}"
PAUSE_FLAG="${PAUSE_FLAG:-/tmp/taosmd-bench-pause}"
BACKED_UP_TSV="${BACKED_UP_TSV:-$HOME/.taos-team/backed-up.tsv}"
MELLUM_MODEL="${MELLUM_MODEL:-mellum2.1-12b-a2.5b-thinking:q4}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-8192}"
export TQDM_DISABLE=1

JUDGE="qwen3:4b"
GRANITE="granite4.2:8b"
QWEN_TAG="qwen3.5:9b"
CONTROL_DIGEST="6488c96fa5fa"
CONTROL_TAG="qwen3.5:9b-20260930-6488c96f"
NEW_DIGEST_EXPECTED="fdf5fd4cd409"
LIMIT="200"
GPU_FREE_LIMIT_MIB=1500

# E-031 retrieval arguments verbatim; the retrieval phase overrides adjacent
# turns and top-k per cell, the other phases keep these exactly.
BASE_RETRIEVAL="--strategy vector-only --fusion mem0_additive --retrieval-top-k 50 --reranker bge-v2-m3 --embed-mode onnx"
BASE_RUN="--limit $LIMIT --no-inline-judge --concurrency 1 --timeout 600 --ckpt --pause-flag $PAUSE_FLAG"

TS="$(date +%Y%m%d_%H%M%S)"
PY="${PY:-python3}"
LOG=""
log(){ echo "[$(date "+%F %T")] $*" | { if [ -n "$LOG" ]; then tee -a "$LOG"; else cat; fi; }; }

# --- prechecks ---------------------------------------------------------------
precheck_fail=0
pc(){ echo "PRECHECK $*"; }
if command -v ollama >/dev/null 2>&1; then pc "ollama binary: $(ollama --version 2>&1 | tail -1)"; else pc "REFUSE: no ollama binary on PATH"; precheck_fail=1; fi
if command -v nvidia-smi >/dev/null 2>&1; then
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  if [ -n "$used" ] && [ "$used" -lt "$GPU_FREE_LIMIT_MIB" ]; then pc "gpu memory.used=${used} MiB < ${GPU_FREE_LIMIT_MIB}: ok"; else pc "REFUSE: gpu memory.used=${used:-unknown} MiB, need < ${GPU_FREE_LIMIT_MIB} (someone else holds the GPU)"; precheck_fail=1; fi
else pc "REFUSE: no nvidia-smi"; precheck_fail=1; fi
if [ -n "${QUEUE_GPU_CLAIM_MSG:-}" ]; then pc "gpu claim recorded: ${QUEUE_GPU_CLAIM_MSG}"; else pc "REFUSE: QUEUE_GPU_CLAIM_MSG unset (post the [GPU CLAIM] first, then export its id)"; precheck_fail=1; fi
for v in REPO DATASET OUTDIR; do
  if [ -z "${!v}" ]; then pc "REFUSE: $v unset"; precheck_fail=1; fi
done
if [ -n "$REPO" ] && [ ! -f "$REPO/benchmarks/locomo_runner.py" ]; then pc "REFUSE: $REPO/benchmarks/locomo_runner.py missing"; precheck_fail=1; fi
if [ -n "$DATASET" ] && [ ! -f "$DATASET" ]; then pc "REFUSE: DATASET $DATASET missing"; precheck_fail=1; fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY-RUN: prechecks would $( [ $precheck_fail -eq 0 ] && echo PASS || echo REFUSE ); listing arms only, nothing runs"
elif [ $precheck_fail -ne 0 ]; then
  echo "REFUSING TO START: fix the PRECHECK lines above"; exit 1
fi

# --- helpers -----------------------------------------------------------------
digest_of(){
  # Prints the /api/tags digest (12 hex) of a model name, or "absent".
  curl -s "${OLLAMA}/api/tags" | $PY -c '
import json, sys
want = sys.argv[1]
try:
    models = json.load(sys.stdin).get("models", [])
except Exception:
    print("absent"); sys.exit(0)
for m in models:
    n = m.get("name", "")
    if n == want or (":" not in want and n == want + ":latest"):
        print((m.get("digest") or "")[:12]); sys.exit(0)
print("absent")' "$1" 2>/dev/null || echo "absent"
}
model_present(){ [ "$(digest_of "$1")" != "absent" ]; }
unload(){ curl -s "${OLLAMA}/api/generate" -d "{\"model\":\"$1\",\"keep_alive\":0}" >/dev/null 2>&1 || true; ollama stop "$1" >/dev/null 2>&1 || true; sleep 5; }
check_pause(){
  if [ -e "$PAUSE_FLAG" ]; then
    log "PAUSE flag $PAUSE_FLAG present before arm $1: exiting 3. Remove the flag and re-run the same phase; judged arms are skipped."
    exit 3
  fi
}
already_judged(){
  # 0 when the rescored file exists and carries at least one judged row.
  [ -f "$1" ] || return 1
  $PY - "$1" <<'PYEOF' >/dev/null 2>&1
import json, sys
d = json.load(open(sys.argv[1])); rows = d.get("results", d)
sys.exit(0 if any(r.get("judge_rejudged") is not None for r in rows) else 1)
PYEOF
}
write_provenance(){
  # $1 sidecar path, $2.. model names
  local out="$1"; shift
  local ver; ver="$(curl -s "${OLLAMA}/api/version" 2>/dev/null | tr -d '\n' | cut -c1-200)"
  {
    echo "{"
    echo "  \"ollama_cli_version\": \"$(ollama --version 2>&1 | tail -1 | sed 's/"/\\"/g')\","
    echo "  \"ollama_api_version\": \"$(echo "$ver" | sed 's/"/\\"/g')\","
    echo "  \"ollama_context_length\": \"${OLLAMA_CONTEXT_LENGTH}\","
    echo "  \"repo_head\": \"$(git -C "$REPO" rev-parse HEAD 2>/dev/null)\","
    echo "  \"dataset_sha256\": \"$(sha256sum "$DATASET" 2>/dev/null | awk '{print $1}')\","
    echo "  \"model_digests\": {"
    local first=1
    for m in "$@"; do
      [ $first -eq 1 ] || echo ","
      first=0
      printf '    "%s": "%s"' "$m" "$(digest_of "$m")"
    done
    echo ""
    echo "  }"
    echo "}"
  } > "$out"
  log "provenance -> $out: $(tr -d '\n' < "$out" | tr -s ' ')"
}

ARMS_LISTED=0
# run_arm TAG MODEL "<runner flags for this cell>"
run_arm(){
  local TAG="$1" MODEL="$2" CELL_FLAGS="$3"
  local OUTF="${OUTDIR}/locomo_${TAG}_s${LIMIT}.json"
  local RSF="${OUTF%.json}.rescored_q34b.json"
  ARMS_LISTED=$((ARMS_LISTED + 1))
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "ARM ${TAG}: model=${MODEL} flags: ${BASE_RETRIEVAL} ${CELL_FLAGS} ${BASE_RUN} -> ${OUTF}"
    return 0
  fi
  if already_judged "$RSF"; then
    log "ARM ${TAG}: SKIP, already judged at ${RSF}"
    return 0
  fi
  check_pause "$TAG"
  if ! model_present "$MODEL"; then
    log "ARM ${TAG}: REFUSED, model ${MODEL} absent from ${OLLAMA}/api/tags"
    echo "ARMRESULT ${TAG} VOID model-absent" | tee -a "$LOG"; return 1
  fi
  log "=== ARM ${TAG}: generation with ${MODEL} START (digest $(digest_of "$MODEL")) ==="
  write_provenance "${OUTF%.json}.provenance.json" "$MODEL" "$JUDGE"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee -a "$LOG"
  # shellcheck disable=SC2086
  $PY -u "$REPO/benchmarks/locomo_runner.py" \
      --dataset "$DATASET" \
      --model "$MODEL" \
      --ollama-url "$OLLAMA" \
      $BASE_RETRIEVAL \
      $CELL_FLAGS \
      $BASE_RUN \
      --run-id "${TAG}_s${LIMIT}" \
      --out "$OUTF" >>"$LOG" 2>&1
  local rc=$?
  if [ $rc -eq 3 ]; then
    log "ARM ${TAG}: runner paused on ${PAUSE_FLAG} (rc=3). Remove the flag and re-run this phase; the runner resumes from its sidecar."
    unload "$MODEL"; exit 3
  fi
  if [ $rc -ne 0 ]; then
    log "ERROR: runner exited rc=${rc} for ${TAG}"
    echo "ARMRESULT ${TAG} VOID runner-nonzero" | tee -a "$LOG"; unload "$MODEL"; return 1
  fi
  unload "$MODEL"
  $PY - "$OUTF" "$TAG" <<'PYEOF' 2>&1 | tee -a "$LOG"
import json, sys
d = json.load(open(sys.argv[1])); rows = d.get("results", [])
ok = [r for r in rows if r.get("predicted") and not r["predicted"].startswith("[generation_error")]
think = [r for r in rows if "<think" in (r.get("predicted") or "").lower()]
print(f"VALIDITY {sys.argv[2]}: rows={len(rows)} real_predictions={len(ok)} think_tags={len(think)}")
sys.exit(0 if (len(ok) >= max(1, int(len(rows) * 0.9)) and not think) else 1)
PYEOF
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    log "ERROR: ${TAG} failed validity (pre-registered VOID), not judging"
    echo "ARMRESULT ${TAG} VOID validity" | tee -a "$LOG"; return 1
  fi
  check_pause "${TAG}-rescore"
  log "ARM ${TAG}: rescore with external judge ${JUDGE} (digest $(digest_of "$JUDGE"))"
  $PY "$REPO/benchmarks/locomo_rescore_streaming.py" "$OUTF" --judge-model "$JUDGE" --ollama-url "$OLLAMA" \
      --concurrency 2 --timeout 300 --out "$RSF" >>"$LOG" 2>&1 \
    || { log "ERROR: rescore failed for ${TAG}"; unload "$JUDGE"; echo "ARMRESULT ${TAG} VOID rescore-failed" | tee -a "$LOG"; return 1; }
  unload "$JUDGE"
  $PY - "$RSF" "$TAG" "$OUTF" <<'PYEOF' | tee -a "$LOG"
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

# --- phases ------------------------------------------------------------------
phase_gen(){
  log "E-038 phase gen: ${GRANITE}, 3 temps x 2 repeat penalties, adjacent 2, top-k 10"
  for t in 0.0 0.2 0.7; do
    for rp in 1.0 1.1; do
      run_arm "e038_gen_t${t}_rp${rp}" "$GRANITE" "--gen-temp $t --repeat-penalty $rp --adjacent-turns 2 --top-k 10" \
        || log "ARM e038_gen_t${t}_rp${rp} FAILED"
    done
  done
}
phase_retrieval(){
  log "E-038 phase retrieval: ${GRANITE} at gen-temp ${RET_GEN_TEMP} repeat-penalty ${RET_REPEAT_PENALTY} (operator-chosen; default is the shipped setting, the script never auto-picks a best cell)"
  for adj in 1 2 3; do
    for k in 5 10 15; do
      run_arm "e038_ret_adj${adj}_k${k}" "$GRANITE" "--gen-temp $RET_GEN_TEMP --repeat-penalty $RET_REPEAT_PENALTY --adjacent-turns $adj --top-k $k" \
        || log "ARM e038_ret_adj${adj}_k${k} FAILED"
    done
  done
}
g1_guard(){
  # The pull overwrites the control blob, so refuse unless the control is
  # already copied to a dated tag AND backed up to Drive (backed-up.tsv row).
  local ok=1
  if [ "${QUEUE_ALLOW_PULL:-0}" != "1" ]; then log "G1 guard: QUEUE_ALLOW_PULL is not 1"; ok=0; fi
  if ! model_present "$CONTROL_TAG"; then log "G1 guard: control tag ${CONTROL_TAG} absent (run: ollama cp ${QWEN_TAG} ${CONTROL_TAG})"; ok=0; fi
  if ! grep -F -q "$CONTROL_TAG" "$BACKED_UP_TSV" 2>/dev/null; then log "G1 guard: no ${CONTROL_TAG} row in ${BACKED_UP_TSV} (Drive backup first, model-backup rule)"; ok=0; fi
  [ $ok -eq 1 ]
}
phase_screen(){
  local E038_ANCHOR_RSF="${OUTDIR}/locomo_e038_gen_t0.2_rp1.0_s${LIMIT}.rescored_q34b.json"
  local E031_FLAGS="--gen-temp 0.2 --adjacent-turns 2 --top-k 10"
  log "E-039 screen: G0 control digest ${CONTROL_DIGEST}, G1 new digest (expected ${NEW_DIGEST_EXPECTED}), G2 ${MELLUM_MODEL}, G3 ${GRANITE}"
  # G0: the pinned control. Prefer the dated tag; fall back to qwen3.5:9b only
  # while its digest is still the control digest.
  local g0_model="$CONTROL_TAG"
  if [ "$DRY_RUN" -eq 0 ] && ! model_present "$CONTROL_TAG"; then
    local cur; cur="$(digest_of "$QWEN_TAG")"
    if [ "$cur" = "$CONTROL_DIGEST" ]; then g0_model="$QWEN_TAG"; log "G0: ${CONTROL_TAG} absent, ${QWEN_TAG} still carries the control digest ${cur}, using it";
    else log "G0: REFUSED, ${CONTROL_TAG} absent and ${QWEN_TAG} digest is ${cur}, not the control ${CONTROL_DIGEST}"; echo "ARMRESULT e039_G0_control VOID control-digest-missing" | tee -a "$LOG"; g0_model=""; fi
  fi
  [ -n "$g0_model" ] && { run_arm "e039_G0_control" "$g0_model" "$E031_FLAGS" || log "ARM e039_G0_control FAILED"; }
  # G1: the re-pushed upstream tag, pulled only behind the guard.
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "G1 guard (dry-run): requires QUEUE_ALLOW_PULL=1, ${CONTROL_TAG} present, and a ${CONTROL_TAG} row in ${BACKED_UP_TSV}; then ollama pull ${QWEN_TAG}"
    run_arm "e039_G1_newdigest" "$QWEN_TAG" "$E031_FLAGS"
  else
    local before; before="$(digest_of "$QWEN_TAG")"
    if [ "$before" = "$CONTROL_DIGEST" ] || [ "$before" = "absent" ]; then
      if g1_guard; then
        log "G1: pulling ${QWEN_TAG} (digest before: ${before})"
        ollama pull "$QWEN_TAG" >>"$LOG" 2>&1 || log "G1: ollama pull exited non-zero"
      else
        log "G1: REFUSED, guard not satisfied, ${QWEN_TAG} left at ${before}"; echo "ARMRESULT e039_G1_newdigest VOID pull-guard" | tee -a "$LOG"
      fi
    fi
    local after; after="$(digest_of "$QWEN_TAG")"
    log "G1: ${QWEN_TAG} digest now ${after} (control ${CONTROL_DIGEST}, expected new ${NEW_DIGEST_EXPECTED})"
    if [ "$after" = "$CONTROL_DIGEST" ] || [ "$after" = "absent" ]; then
      log "G1: SKIPPED, ${QWEN_TAG} is not at a new digest"
    else
      [ "$after" = "$NEW_DIGEST_EXPECTED" ] || log "G1: NOTE digest ${after} differs from the digest recorded at pre-registration (${NEW_DIGEST_EXPECTED}); recorded, arm still runs"
      run_arm "e039_G1_newdigest" "$QWEN_TAG" "$E031_FLAGS" || log "ARM e039_G1_newdigest FAILED"
    fi
  fi
  # G2: Mellum2.1 MoE, thinking off (runner default: think false), validity check catches any <think tag.
  run_arm "e039_G2_mellum" "$MELLUM_MODEL" "$E031_FLAGS" || log "ARM e039_G2_mellum FAILED"
  # G3: second control, reused from the E-038 anchor cell when it was judged in this OUTDIR.
  if [ "$DRY_RUN" -eq 0 ] && already_judged "$E038_ANCHOR_RSF"; then
    log "G3: REUSED, the E-038 anchor cell ${E038_ANCHOR_RSF} is judged on this host and Ollama version; no separate granite arm"
    echo "ARMRESULT e039_G3_granite REUSED ${E038_ANCHOR_RSF}" | tee -a "$LOG"
  else
    [ "$DRY_RUN" -eq 1 ] && echo "G3 (dry-run): skipped when ${E038_ANCHOR_RSF} is already judged, else:"
    run_arm "e039_G3_granite" "$GRANITE" "$E031_FLAGS" || log "ARM e039_G3_granite FAILED"
  fi
}

# --- main --------------------------------------------------------------------
if [ "$DRY_RUN" -eq 0 ]; then
  mkdir -p "$OUTDIR"
  LOG="${OUTDIR}/queue_oct2026_${PHASE}_${TS}.log"
  cd "$REPO" || { echo "ERROR: cannot cd to REPO=$REPO"; exit 1; }
  log "repo at $(git log --oneline -1 2>/dev/null || echo unknown) | ollama cli $(ollama --version 2>&1 | tail -1) | api $(curl -s "${OLLAMA}/api/version" | tr -d '\n')"
  log "OLLAMA_CONTEXT_LENGTH=${OLLAMA_CONTEXT_LENGTH} judge=${JUDGE} digest $(digest_of "$JUDGE") | gpu claim: ${QUEUE_GPU_CLAIM_MSG}"
  log "dataset sha256: $(sha256sum "$DATASET" | awk '{print $1}')"
fi

case "$PHASE" in
  gen) phase_gen ;;
  retrieval) phase_retrieval ;;
  screen) phase_screen ;;
  all) phase_gen; phase_retrieval; phase_screen ;;
esac

if [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY-RUN: ${ARMS_LISTED} arms listed for phase ${PHASE}"
  exit 0
fi
log "final cleanup: stopping all ollama models"
for M in $(ollama ps 2>/dev/null | awk 'NR>1{print $1}'); do ollama stop "$M" >/dev/null 2>&1 || true; done
sleep 5
log "final GPU: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
log "=== CHAIN ${PHASE} COMPLETE: $(grep -c '^ARMRESULT' "$LOG") ARMRESULT lines in ${LOG} ==="
