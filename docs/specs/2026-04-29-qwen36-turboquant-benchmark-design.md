# Qwen3.6-35B-A3B + TurboQuant — benchmark design spec

**Workstream:** new architectural lever for the LoCoMo same-tier leaderboard. Tests whether a 35B MoE running under llama.cpp + TurboQuant breaks the 9B+adj=2 ceiling (currently `rrf_full_stack_qwen9b_adj2 = 0.5571` external).
**Depends on:** current Fedora chain (`kitchen_sink_qwen9b_adj2` + LMEKU Phase A) finishing — these must clear the GPU before we install a second inference runtime alongside Ollama.
**Source:** community config posted on r/LocalLLM (`drepublic`, +25), running Qwen3.6-35B-A3B-Q4 at 36-38 tps on a 2070 8GB / 32GB DDR4 system using `TheTom/llama-cpp-turboquant`. We have stronger hardware (3060 12GB / 32GB DDR4) so should hit similar or better throughput, with more GPU layers.

## What we're testing

**Question:** Does a 35B-A3B MoE running under TurboQuant + CPU-MoE offload beat the qwen3.5:9b same-tier ceiling on LoCoMo?

**Hypothesis:** Yes. 35B-A3B has roughly 4× the in-weight knowledge of 9B and the LoCoMo cross-tier finding showed that smaller generators benefit *more* from retrieval improvements, with the 9B model already saturating most of our retrieval levers. A larger generator on the same retrieval stack should finally break out of the 0.55-0.56 plateau.

**Failure mode we want to catch:** the community report is a chat-quality / coding observation, not a memory-recall benchmark. The model could be smarter at conversation but no better — or worse — at retrieving a specific fact from a 400-turn LoCoMo session. We need to measure, not assume.

## Why now

- The 9B + adj=2 same-tier leaderboard has visibly plateaued: `rrf_full_stack 0.5571`, `multilevel_full_stack 0.5519`, with diminishing returns from every new lever (multi-level + RRF regressed; HyDE regressed at both 9B and Pi NPU; inverse-temporal regressed slightly).
- Every recent regression has been on the *retrieval* side. The honest interpretation is that 9B is the bottleneck — adding more retrieval signal hands the model context it can't make use of.
- Bumping the generator is the next obvious lever. Going to 35B-dense on a 12GB card is a non-starter; 35B-A3B with CPU-MoE offload is the only way to fit, and TurboQuant is the only way to fit 128k context KV cache on the same card.

## What needs to exist

1. **`TheTom/llama-cpp-turboquant` built from source.** This is a beta fork of llama.cpp with TurboQuant KV-cache quantisation (`turbo3`). Mainline llama.cpp does not carry it; Ollama does not carry it. Compile path: clone the fork, `cmake -DGGML_CUDA=on`, `make -j`. Verify the resulting `llama-server` runs `--help` and lists `--cache-type-k turbo3` as a valid value.
2. **The unsloth GGUF.** `unsloth/Qwen3.6-35B-A3B-GGUF` — file `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, ~20 GB. Download via `huggingface-cli download` to `/home/jay/models/`. We do not need the chat template separately; unsloth GGUFs embed it in the metadata.
3. **`llama-server` running on a free port** (the post uses 8085; we already have Ollama on 11434, qmd on others — pick something unused). Started as a systemd-style background service with the recommended flags.
4. **Adapter shim in `benchmarks/locomo_runner.py`.** The runner currently hardcodes Ollama's `/api/generate` JSON shape via `_ollama_generate`. llama-server speaks an OpenAI-compatible `/v1/chat/completions`. Add a `--llm-backend ollama|llama-server` flag, dispatching to two small adapter functions. Keep Ollama as default so existing chains keep working.

## The recommended `llama-server` config (from drepublic, with one tweak for our hardware)

```bash
llama-server \
    --model /home/jay/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
    --host 127.0.0.1 \
    --port 8085 \
    --ctx-size 131072 \
    --n-gpu-layers 999 \
    --n-cpu-moe 24 \                # was 32 on 8GB; 12GB lets us put more experts on GPU
    --cache-type-k turbo3 \
    --cache-type-v turbo3 \
    --flash-attn on \
    --batch-size 1024 \
    --parallel 1 \
    --ubatch-size 512 \
    --threads 6 \
    --cont-batching \
    --timeout 300 \
    --temp 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --metrics
```

`--n-cpu-moe 24` is a starting estimate for 12GB. Tune empirically: lower it (more experts on GPU) until we hit OOM, then back off by 2. Goal is maximum tps without spilling.

## Implementation plan

**Phase 0 — runtime install (~1h, blocked on chain end):**

- `git clone https://github.com/TheTom/llama-cpp-turboquant /home/jay/llama-cpp-turboquant`
- Build: `cmake -B build -DGGML_CUDA=on && cmake --build build -j`
- Verify `--cache-type-k turbo3` appears in `--help`. If the flag is missing, the fork has been rebased away from TurboQuant — fall back to vanilla llama.cpp without turbo3 KV (still useful as a baseline 35B-A3B test, just slower).
- `huggingface-cli download unsloth/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --local-dir /home/jay/models/qwen36-a3b-q4`

**Phase 1 — adapter shim (~half day):**

- Add `--llm-backend ollama|llama-server` (default `ollama`) to `locomo_runner.py`.
- Extract `_ollama_generate` body into a backend dispatcher; add `_llama_server_generate(client, url, model, prompt, ...)` that POSTs to `${url}/v1/chat/completions` with `messages=[{"role":"user","content":prompt}]`.
- Honor the same temperature / top_p / top_k options.
- Smoke test: run `--limit 5` on LoCoMo cat 4 (open-domain) with `--llm-backend llama-server --model "Qwen3.6-35B-A3B-UD-Q4_K_M"` to confirm end-to-end answer generation.

**Phase 2 — first measured run (~6h):**

- Start `llama-server` in a screen session as a daemon for the duration of the bench.
- One smoke cell first: `qwen36_turboquant_baseline` = `--adjacent-turns 2` only, full 1540 QAs. This tells us the model's accuracy without any retrieval scaffolding, comparable to other `qwen9b_adj2` baseline cells. ~4-6h depending on measured tps.
- Ship the JSON through the existing `locomo_rescore_streaming.py` pipeline with `--judge-model qwen3:4b`. Same external judge as every other cell — apples-to-apples.

**Phase 3 — full-stack run (~6h):**

If Phase 2 hits ≥0.516 (the qwen9b_adj2 baseline), it's worth burning another bench:

- `qwen36_turboquant_full_stack` = adj=2 + `--retrieval-top-k 20 --llm-query-expansion --multi-level-retrieval --fusion rrf --reranker bge-v2-m3` (the same recipe as the current 0.5571 leader).
- Same external rescore.

**Phase 4 — scorecard + docs (~1h):**

- Add a row to `docs/benchmarks.md` same-tier leaderboard.
- Update `docs/specs/2026-04-19-locomo-scorecards.md`.
- If we beat 0.557: README headline gets bumped. Don't pre-commit to that — measure first.

## Configs to bench (minimal viable matrix)

| Cell | Backend | Model | Recipe | ETA | Purpose |
|------|---------|-------|--------|-----|---------|
| `qwen36_turboquant_baseline` | llama-server | Qwen3.6-35B-A3B-UD-Q4_K_M | adj=2 | ~5h | does the model help at all? |
| `qwen36_turboquant_full_stack` | llama-server | Qwen3.6-35B-A3B-UD-Q4_K_M | adj=2 + k=20 + llm-exp + RRF + multi-level + BGE | ~6h | does it break the same-tier ceiling? |

Run sequentially. Skip Phase 3 if Phase 2 regresses below `qwen9b_adj2` baseline (saves 6h on a model that isn't going to ship).

## Success criteria

**Honest bar:** `qwen36_turboquant_baseline` ≥ 0.516 (matches `qwen9b_adj2` baseline). Below this, 35B-A3B is *worse* than 9B for memory recall — surprising but reportable.

**Acceptable:** `qwen36_turboquant_full_stack` ≥ 0.5571 (matches current leader). At parity, the larger generator isn't worth its own scorecard row but tells us 35B-A3B is a viable 9B replacement for users who already have 12GB+ VRAM and 32GB+ RAM.

**Headline:** `qwen36_turboquant_full_stack` ≥ 0.60 external. Breaks the 9B plateau. Goes in the README. Worth a methodology note about the TurboQuant + CPU-MoE setup.

**Dream bar:** ≥ 0.65. Approaches Mem0's hosted gpt-4o-mini number (0.66) on local hardware. Publishable as a standalone finding.

## Failure cases

1. **TurboQuant fork unmaintained / won't build.** Fallback: vanilla llama.cpp same flags minus `--cache-type-k turbo3`. KV cache uses default Q4 instead of TurboQuant's turbo3 — likely OOMs the 12GB card at 128k context. Drop `--ctx-size` to 32k for a baseline test of *just* the model, separate from the TurboQuant claim.
2. **Throughput is awful (e.g. <10 tps).** A 1540-QA run becomes >40h, infeasible. Subsample to 300 QAs (cat 1+2 weighted) for a coarse signal. Note in the JSON.
3. **Accuracy regresses vs 9B.** Honest write-up: "35B-A3B does not generalise to memory recall tasks despite higher token quality." Don't ship the model. Add a memory note so we don't re-test it. Consider that A3B's per-token activation budget (3B effective) may be the actual limit, not the 35B nameplate.
4. **Model loads but the OpenAI-compat endpoint mangles Qwen's chat template.** Chat-template-bake-in via unsloth is supposed to handle this, but if responses are garbage, pass `--chat-template-file` explicitly with the Qwen3.6 jinja template.

## Open questions

1. **Is `--n-cpu-moe 24` right for 12GB?** Empirically tune. Run `nvidia-smi` while llama-server is loaded; if VRAM has >2GB headroom, lower to 20 (more experts on GPU = faster).
2. **Does the runner need streaming?** Current `_ollama_generate` uses `stream: false`. llama-server's `/v1/chat/completions` defaults to non-streaming if `stream` is omitted. Keep it simple — non-streaming.
3. **Same external judge?** Yes, `qwen3:4b` on Ollama. We need to keep Ollama running on :11434 *alongside* llama-server on :8085 during the rescore step. Both fit on the 3060 (judge is 2.5 GB, 35B-A3B fits via offload).
4. **Should we also bench `qwen3.6:35b-a3b` via Ollama for comparison?** Optional Phase 5. If TurboQuant + CPU-MoE genuinely helps, the Ollama version will be slower and either match or underperform. Worth one cell to confirm — but only after the TurboQuant cells land, otherwise we're spending compute on a known weaker config.

## Rough timeline

- 1h — install turboquant + download model
- 4h — adapter shim + smoke test (5 QAs)
- 5-6h — Phase 2 baseline bench
- 1h — rescore + analyse
- 5-6h — Phase 3 full-stack bench (if baseline doesn't regress)
- 1h — rescore + analyse
- 1h — docs + scorecard

Total: ~18-24h of clock time, mostly machine-bound on the 3060. Human time: ~half day (mostly the adapter and the failure-case branches).

## Out of scope for this spec

- **Pi NPU port.** rkllama doesn't load llama.cpp GGUFs and the NPU has only 16 GB total. 35B-A3B is not feasible on Pi. Stays a 12GB-tier-only experiment.
- **Multi-GPU.** The 3060 12GB is single-card. Multi-GPU configs (the Reddit thread mentions various 1060/2060/2070 builds) are out of scope.
- **Production wire-in.** This is a benchmark-only experiment. If it wins, the production wire-in (taOSmd's actual generator selection) is a separate piece of work — most users running taOSmd aren't going to install a llama.cpp fork.
