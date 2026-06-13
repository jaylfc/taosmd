# Bench Pause and Resume

## What this is

LoCoMo benchmark runs can take several hours. The runner supports conversation-granularity checkpointing so a run can be paused cleanly, the machine rebooted (or handed off for another purpose), and the run continued later with no repeated work. After each conversation the runner writes a record to a sidecar file (`<out>.ckpt.jsonl`, fsync'd to disk). When a pause flag file is present the runner finishes its current in-flight conversation, writes the checkpoint, prints a resume hint, and exits with code 3. The most you can lose to a clean pause is the conversation currently running when the flag is created; every earlier conversation is safe on disk.

---

## Starting a pausable run

Pass `--ckpt` together with an explicit `--out` path. The sidecar is placed next to the output file and named `<out>.ckpt.jsonl`.

```bash
python3 benchmarks/locomo_runner.py \
  --model gemma4:e2b \
  --retrieval-top-k 50 --adjacent-turns 2 --reranker bge-v2-m3 --fusion mem0_additive \
  --out benchmarks/results/locomo-rerank-gemma4e2b.json \
  --ckpt
```

The sidecar for this run will be `benchmarks/results/locomo-rerank-gemma4e2b.json.ckpt.jsonl`.

You can supply a custom pause-flag path with `--pause-flag /path/to/flag`. The default is `/tmp/taosmd-bench-pause`.

---

## Pausing a run (including before a Windows reboot)

1. Create the pause-flag file:

   ```bash
   touch /tmp/taosmd-bench-pause
   ```

2. Wait. The runner checks the flag between conversations, not mid-conversation. Depending on how far through the current conversation it is, this can take a few minutes. That wait is correct behavior; the runner will not cut a conversation short.

3. When the runner exits, verify it exited with code 3:

   ```bash
   echo $?   # expect: 3
   ```

4. Confirm the sidecar exists and has completed conversations:

   ```bash
   ls -lh benchmarks/results/locomo-rerank-gemma4e2b.json.ckpt.jsonl
   grep -c '"kind": "conv"' benchmarks/results/locomo-rerank-gemma4e2b.json.ckpt.jsonl
   ```

5. It is now safe to reboot or shut down.

The `rebootwin.sh` helper (see `benchmarks/rebootwin.sh`) automates steps 1 and 2: `bash benchmarks/rebootwin.sh pause`.

---

## Resuming after the box is back

1. Remove the pause flag **before** re-running:

   ```bash
   rm -f /tmp/taosmd-bench-pause
   ```

2. Re-run the **identical** command with `--resume` appended:

   ```bash
   python3 benchmarks/locomo_runner.py \
     --model gemma4:e2b \
     --retrieval-top-k 50 --adjacent-turns 2 --reranker bge-v2-m3 --fusion mem0_additive \
     --out benchmarks/results/locomo-rerank-gemma4e2b.json \
     --resume
   ```

   `--resume` implies `--ckpt`. The runner loads the sidecar, verifies the config hash, skips every already-completed conversation (including their ingest), and picks up where it stopped.

**Important:** every flag that affects the run config must match the original command. The runner computes a hash over the config at startup and compares it to the hash stored in the sidecar header. A mismatch exits with code 2 and refuses to proceed. If you need to change flags, start a fresh run with a different `--out` path.

---

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Normal completion. All conversations finished. |
| 3 | Clean pause. The pause-flag was found between conversations. Remove the flag and re-run with `--resume` to continue. |
| 2 | Config mismatch or startup error. The flags on the resume command do not match the original run, or the sidecar is corrupt. Do not continue this run; investigate or start fresh with a new `--out`. |
| 1 | All QAs failed. The run completed but every question failed to score. |

---

## Failure and recovery

**Hard power loss or OOM kill mid-conversation:** the sidecar keeps every conversation that was already completed and fsync'd before the kill. The in-flight conversation (the one running when the process died) has no record in the sidecar and will be re-run from scratch on resume. No completed conversation is lost.

**Truncated final line in the sidecar:** the runner detects a malformed final line and skips it with a warning on load, then re-runs that conversation. This is the expected crash-mid-write recovery path.

**Sidecar missing on resume:** if you pass `--resume` and no sidecar exists, the runner starts fresh and logs that it is doing so. No error; the run proceeds from conversation 1.

**Corrupt sidecar body (non-final line malformed):** the runner raises an error and refuses to load. This indicates data corruption beyond normal crash tolerance. Investigate or start a new run.

---

## Quick reference

| Action | Command |
|--------|---------|
| Start a pausable run | `python3 benchmarks/locomo_runner.py ... --out benchmarks/results/<name>.json --ckpt` |
| Pause (manual) | `touch /tmp/taosmd-bench-pause` |
| Pause (helper, waits for clean exit) | `bash benchmarks/rebootwin.sh pause` |
| Check pause/runner status | `bash benchmarks/rebootwin.sh status` |
| Resume after reboot | Remove flag: `rm -f /tmp/taosmd-bench-pause`, then re-run original command with `--resume` |
| Resume (helper prompt) | `bash benchmarks/rebootwin.sh resume` |
| Check sidecar conversation count | `grep -c '"kind": "conv"' benchmarks/results/<name>.json.ckpt.jsonl` |
| Check last exit code | `echo $?` immediately after the runner exits |
