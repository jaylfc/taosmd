#!/usr/bin/env bash
# =============================================================================
# LoCoMo generation-side prompt-engineering sweep (promptx).
#
# Anchor: qwen3.5:9b + --no-think-prefix lifts LoCoMo from 0.535 -> 0.635
# (external qwen3:4b judge, subset-200) by stopping the model hedging on
# temporal questions, at the cost of ~5.5% fully-empty answers. This sweep
# tries to KEEP that gain while KILLING the empties, plus tests a decoding
# lever. Every arm keeps --no-think-prefix as the base (0.635 is the anchor we
# are improving on); only the added generation-side lever changes per arm.
#
# Arms run SEQUENTIALLY (shared 12 GB GPU: never two Ollama-backed jobs at once;
# GPU contention starves both into timeouts). Each arm: generate -> unload the
# generator -> rescore with the external qwen3:4b judge -> unload the judge.
#
#   G0_anchor      --no-think-prefix
#                  Hypothesis: reproduces the 0.635 anchor. Sanity gate — if
#                  this does not land near 0.635, RETRIEVAL_FLAGS below does not
#                  match the anchor's retrieval config; fix that before reading
#                  the other arms.
#
#   G1_reppen      --no-think-prefix --repeat-penalty 1.0
#                  Hypothesis: disabling Ollama's default repeat penalty stops
#                  the model collapsing into repetitive hedges/refusals, so it
#                  commits to an answer more often without hurting correct ones.
#
#   G2_antihedge   --no-think-prefix --anti-hedge-prompt
#                  Hypothesis: a commit-focused prompt ("commit to a specific
#                  answer, absolute dates only, say unknown only after checking
#                  every memory") converts hedged/empty temporal answers into
#                  committed ones -> higher Judge, fewer empties.
#
#   G3_emptyretry  --no-think-prefix --empty-retry
#                  Hypothesis: the asymmetric fallback recovers exactly the
#                  ~5.5% empties: no-think for the commit gain, then a one-shot
#                  thinking-ON rescue only on the answers that came back empty
#                  or abstaining. Kills empties without touching the winners.
#
#   G4_combo       --no-think-prefix --repeat-penalty 1.0 --anti-hedge-prompt \
#                  --num-predict 120 --stop <newline-Question / paragraph> \
#                  --empty-retry
#                  Hypothesis: stack the levers — commit prompt + no repeat
#                  penalty + a tight output cap/stop to keep answers to one
#                  clean sentence + the empty rescue. Best-case ceiling of the
#                  sweep. (num-predict also bounds the thinking-ON rescue's
#                  tokens, so watch G4 for any rescue truncation vs G3.)
#
# NOTE on the G4 --stop value: the design brief wrote it as
#   --stop "$'\n'Question:|||$'\n\n'"
# but $'...' does NOT expand inside double quotes in bash, so that literal
# would pass the characters $'\n'... unchanged. We build the intended value
# with ANSI-C quoting as a standalone token (STOP4 below) so the '\n's become
# real newlines while '|||' stays the literal Python-side split delimiter. The
# runner splits --stop on '|||' into ["\nQuestion:", "\n\n"] — stop at the next
# echoed "Question:" line or the next paragraph break.
# =============================================================================
set -u

# --- operator fills these in on the host -------------------------------------
REPO="__FILL_REPO_ROOT__"            # e.g. /home/jay/taosmd
DATASET="__FILL_DATASET_PATH__"      # LoCoMo JSON, e.g. $REPO/data/locomo/data/locomo10.json
OUTDIR="__FILL_OUTPUT_DIR__"         # where per-arm result + rescore JSONs land
# -----------------------------------------------------------------------------

OLLAMA="http://localhost:11434"
JUDGE="qwen3:4b"                     # external judge (different family than generator)
MODEL="qwen3.5:9b"                   # generator under test
GEN_TEMP="0.2"                       # every prior LoCoMo cell ran at 0.2
LIMIT="200"                          # subset-200
CONCURRENCY="1"                      # 1 = strictly sequential QAs (GPU-safe)
TIMEOUT="600"                        # per-call HTTP timeout (s); generous for think-ON rescue

# Retrieval config held CONSTANT across all arms (only generation changes).
# Defaults to the shipped 12 GB MaxSim+rerank recipe. If G0_anchor does not
# reproduce ~0.635, adjust this to match the anchor run's retrieval flags.
RETRIEVAL_FLAGS="--strategy vector-only --fusion mem0_additive --top-k 10 --retrieval-top-k 50 --adjacent-turns 2 --reranker bge-v2-m3 --embed-mode onnx"

# num_ctx 8192 and seed 7 are NOT locomo_runner.py flags (the runner exposes
# neither --num-ctx nor --seed). Apply them at the Ollama layer:
#   * num_ctx: bake into the served model (Modelfile: PARAMETER num_ctx 8192,
#     then `ollama create`), or export OLLAMA_CONTEXT_LENGTH=8192 before
#     starting the Ollama server. Confirm with `ollama show ${MODEL}`.
#   * seed: the runner does not thread a sampling seed, so at gen-temp 0.2
#     expect minor run-to-run variance. If exact reproducibility is required,
#     add a --seed flag to the runner (Ollama options.seed) first.
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-8192}"

PY="${PY:-python3}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${OUTDIR}/promptx_chain_${TS}.log"
export TQDM_DISABLE=1

mkdir -p "$OUTDIR"
cd "$REPO" || { echo "ERROR: cannot cd to REPO=$REPO"; exit 1; }

log(){ echo "[$(date "+%F %T")] $*" | tee -a "$LOG"; }
unload(){ curl -s "${OLLAMA}/api/generate" -d "{\"model\":\"$1\",\"keep_alive\":0}" >/dev/null 2>&1 || true; ollama stop "$1" >/dev/null 2>&1 || true; sleep 5; }

# G4 stop value — real newlines, literal '|||' split delimiter (see NOTE above).
STOP4=$'\nQuestion:|||\n\n'

run_arm(){
  local TAG="$1"; shift
  local ARM_FLAGS="$*"
  local OUTF="${OUTDIR}/promptx_${TS}_${TAG}_s${LIMIT}.json"
  log "=== ARM ${TAG}: generate with ${MODEL} START ==="
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee -a "$LOG"
  log "ARM ${TAG} flags: ${ARM_FLAGS}"
  # shellcheck disable=SC2086
  $PY -u benchmarks/locomo_runner.py \
      --dataset "$DATASET" \
      --model "$MODEL" \
      $RETRIEVAL_FLAGS \
      --gen-temp "$GEN_TEMP" \
      --limit "$LIMIT" \
      --no-inline-judge \
      --concurrency "$CONCURRENCY" \
      --timeout "$TIMEOUT" \
      --run-id "promptx_${TAG}_s${LIMIT}" \
      --out "$OUTF" \
      $ARM_FLAGS >>"$LOG" 2>&1 \
    || { log "ERROR: runner exited non-zero for ${TAG}"; unload "$MODEL"; return 1; }
  unload "$MODEL"

  # Guard: refuse to judge a run that produced mostly generation errors.
  $PY - "$OUTF" <<'PYEOF' >>"$LOG" 2>&1 || { log "ERROR: ${TAG} too few real predictions -- not judging"; return 1; }
import json, sys
d = json.load(open(sys.argv[1])); rows = d.get("results", [])
ok = [r for r in rows if r.get("predicted") and not r["predicted"].startswith("[generation_error")]
print(f"rows={len(rows)} real_predictions={len(ok)}")
sys.exit(0 if len(ok) >= max(1, int(len(rows) * 0.9)) else 1)
PYEOF

  local RSF="${OUTF%.json}.rescored_q34b.json"
  log "ARM ${TAG}: rescore with external judge ${JUDGE}"
  $PY benchmarks/locomo_rescore_streaming.py "$OUTF" \
      --judge-model "$JUDGE" --ollama-url "$OLLAMA" \
      --concurrency 2 --timeout 300 --out "$RSF" >>"$LOG" 2>&1 \
    || { log "ERROR: rescore failed for ${TAG}"; unload "$JUDGE"; return 1; }
  unload "$JUDGE"

  # Verdict: external-judge overall + per-category, mirrors the granite chain.
  $PY - "$RSF" "$TAG" <<'PYEOF' | tee -a "$LOG"
import json, sys
d = json.load(open(sys.argv[1])); rows = d.get("results", d)
names = {1: "MultiHop", 2: "Temporal", 3: "OpenDomain", 4: "SingleHop"}
tot = []; bycat = {}
fired = still = 0
for r in rows:
    fired += int(r.get("empty_retry_fired", 0) or 0)
    still += int(r.get("empty_retry_still_empty", 0) or 0)
    v = r.get("judge_rejudged")
    if v is None: continue
    tot.append(v); bycat.setdefault(int(r.get("category", 0)), []).append(v)
if not tot:
    print(f"VERDICT {sys.argv[2]}: NO JUDGED ROWS")
else:
    cats = " ".join(f"{names.get(c, c)}={sum(v)/len(v):.3f}(n={len(v)})" for c, v in sorted(bycat.items()))
    print(f"VERDICT {sys.argv[2]}: n={len(tot)} overall={sum(tot)/len(tot):.4f} | {cats}")
print(f"  empty-retry: fired={fired} still_empty_after={still}")
PYEOF
  log "=== ARM ${TAG} DONE ==="
}

log "repo at $(git log --oneline -1 2>/dev/null || echo unknown)"
log "OLLAMA_CONTEXT_LENGTH=${OLLAMA_CONTEXT_LENGTH} (num_ctx target 8192); generator=${MODEL} judge=${JUDGE}"

run_arm "G0_anchor"     --no-think-prefix                                                                       || log "ARM G0_anchor FAILED"
run_arm "G1_reppen"     --no-think-prefix --repeat-penalty 1.0                                                  || log "ARM G1_reppen FAILED"
run_arm "G2_antihedge"  --no-think-prefix --anti-hedge-prompt                                                   || log "ARM G2_antihedge FAILED"
run_arm "G3_emptyretry" --no-think-prefix --empty-retry                                                         || log "ARM G3_emptyretry FAILED"
run_arm "G4_combo"      --no-think-prefix --repeat-penalty 1.0 --anti-hedge-prompt --num-predict 120 --stop "$STOP4" --empty-retry || log "ARM G4_combo FAILED"

log "final cleanup: stopping all ollama models"
for M in $(ollama ps 2>/dev/null | awk 'NR>1{print $1}'); do ollama stop "$M" >/dev/null 2>&1 || true; done
sleep 5
command -v nvidia-smi >/dev/null 2>&1 && log "final GPU: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
log "=== CHAIN COMPLETE ==="
