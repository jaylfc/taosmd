# E-012 Score-Up Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two independent, default-off reasoning levers (query decomposition with iterative retrieval; CoVe-style answer self-verification) to the LongMemEval Judge harness, plus an ablation runner, so we can attribute any Judge-score gain over the 47.2% baseline to a specific lever.

**Architecture:** Both levers are env-gated toggles on `benchmarks/longmemeval_runner.py`. Decomposition replaces the single vector search with a fan-out over sub-queries (deduplicated union). Self-verification adds one extra generator pass after the first answer. Default off means the baseline arm is byte-for-byte the current behavior. A bash matrix runner sweeps the arms sequentially (one Ollama job at a time per the GPU-contention rule).

**Tech Stack:** Python 3, httpx async, Ollama HTTP API, pytest. No GPU needed to build or unit-test; the matrix run itself needs the bench-host GPU.

**Branch:** `bench/e012-judge-harness` (already checked out; the harness and the depth/rerank levers already live here).

---

## File Structure

- Modify `benchmarks/longmemeval_runner.py`: add `MULTIHOP_PROMPT`, `VERIFY_PROMPT`, the `TAOSMD_DECOMPOSE`/`TAOSMD_DECOMPOSE_MODEL`/`TAOSMD_SELF_VERIFY` env flags, `decompose_query()`, `self_verify_answer()`, and wire both into `run_benchmark()`.
- Create `tests/test_e012_levers.py`: unit tests for the two helpers using a fake Ollama client (no live LLM).
- Create `benchmarks/e012_ablation.sh`: the matrix runner that sweeps arms by setting env and invoking the harness.

The runner is a single benchmark script and stays one file; the new helpers are small and belong beside the existing `llm_answer`/`score_answer_llm`.

---

## Task 1: Query decomposition helper (TAOSMD_DECOMPOSE)

**Files:**
- Modify: `benchmarks/longmemeval_runner.py` (config block after line 54; new helper after `llm_answer`, ~line 143)
- Test: `tests/test_e012_levers.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_e012_levers.py`:

```python
import importlib.util
import pathlib
import asyncio

_RUNNER = pathlib.Path(__file__).resolve().parent.parent / "benchmarks" / "longmemeval_runner.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("lme_runner", _RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeResp:
    def __init__(self, content, status=200):
        self.status_code = status
        self._content = content

    def json(self):
        return {"message": {"content": self._content}}


class _FakeClient:
    """Returns a fixed Ollama chat response for any .post()."""
    def __init__(self, content, status=200):
        self._content = content
        self._status = status

    async def post(self, *args, **kwargs):
        return _FakeResp(self._content, self._status)


class _BoomClient:
    async def post(self, *args, **kwargs):
        raise RuntimeError("boom")


def test_decompose_returns_lines_when_two_or_more():
    mod = _load_runner()
    out = asyncio.run(mod.decompose_query(_FakeClient("who did she marry\nwhen did they meet"), "Q"))
    assert out == ["who did she marry", "when did they meet"]


def test_decompose_falls_back_when_fewer_than_two_lines():
    mod = _load_runner()
    out = asyncio.run(mod.decompose_query(_FakeClient("only one line"), "original question"))
    assert out == ["original question"]


def test_decompose_falls_back_on_error():
    mod = _load_runner()
    out = asyncio.run(mod.decompose_query(_BoomClient(), "original question"))
    assert out == ["original question"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_e012_levers.py -v`
Expected: FAIL with `AttributeError: module 'lme_runner' has no attribute 'decompose_query'`.

- [ ] **Step 3: Add the config flags**

In `benchmarks/longmemeval_runner.py`, after the rerank config block (after line 54, the `RERANK_TOP_K = ...` line), add:

```python
# E-012 lever 2: query decomposition with iterative retrieval. Split the
# question into 2-3 sub-queries, retrieve for each, union deduplicated. Ported
# from locomo_runner._decompose_query. Default off.
DECOMPOSE = os.environ.get("TAOSMD_DECOMPOSE", "0") == "1"
DECOMPOSE_MODEL = os.environ.get("TAOSMD_DECOMPOSE_MODEL", "gemma4:e2b")
```

- [ ] **Step 4: Add the MULTIHOP_PROMPT and decompose_query helper**

After the `llm_answer` function (after line 143), add:

```python
MULTIHOP_PROMPT = """Split this question into 2 or 3 shorter, focused sub-queries that together cover everything needed to answer it. Respond with one sub-query per line. No numbering, no explanation.

Question: {question}"""


async def decompose_query(client, question: str) -> list:
    """Split a question into 2-3 sub-queries via a small utility model.

    Returns a list of >=2 stripped lines, or [question] on failure or if fewer
    than two lines come back (so callers can iterate uniformly).
    """
    try:
        resp = await client.post(
            f"{REMOTE_LLM_URL}/api/chat",
            json={
                "model": DECOMPOSE_MODEL,
                "messages": [{"role": "user", "content": MULTIHOP_PROMPT.format(question=question) + " /no_think"}],
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_predict": 80},
            },
            timeout=30,
        )
        if resp.status_code == 200:
            raw = resp.json().get("message", {}).get("content", "")
            lines = [l.strip() for l in raw.splitlines() if l.strip()]
            if len(lines) >= 2:
                return lines
    except Exception:
        pass
    return [question]
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_e012_levers.py -v`
Expected: the three decompose tests PASS.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/longmemeval_runner.py tests/test_e012_levers.py
git commit -m "bench(e012): add query-decomposition helper (lever 2)"
```

---

## Task 2: Wire decomposition into retrieval

**Files:**
- Modify: `benchmarks/longmemeval_runner.py:261` (the `vector_results = await vmem.search(...)` line)

- [ ] **Step 1: Replace the single vector search with the decomposition fan-out**

In `run_benchmark`, replace this exact line (currently line 261):

```python
        vector_results = await vmem.search(question, limit=RETRIEVE_LIMIT)
```

with:

```python
        if DECOMPOSE and llm_client is not None:
            sub_queries = await decompose_query(llm_client, question)
            seen_texts = set()
            vector_results = []
            for sq in sub_queries:
                for r in await vmem.search(sq, limit=RETRIEVE_LIMIT):
                    t = r.get("text", "")
                    if t and t not in seen_texts:
                        seen_texts.add(t)
                        vector_results.append(r)
            vector_results = vector_results[:RETRIEVE_LIMIT]
        else:
            vector_results = await vmem.search(question, limit=RETRIEVE_LIMIT)
```

The following `rr = _get_reranker()` block is unchanged: rerank still runs over the (now possibly unioned) `vector_results`.

- [ ] **Step 2: Verify the harness still imports and runs offline (no LLM, decompose off = baseline unchanged)**

Run: `.venv/bin/python benchmarks/longmemeval_runner.py --limit 1`
Expected: it runs one question in substring mode and prints a RESULTS block (no crash). Note: this requires `benchmarks/data/longmemeval_oracle.json` present; on a machine without it, skip this step and rely on the bench-host run in Task 5.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/longmemeval_runner.py
git commit -m "bench(e012): wire decomposition fan-out into retrieval (lever 2)"
```

---

## Task 3: Answer self-verification helper (TAOSMD_SELF_VERIFY)

**Files:**
- Modify: `benchmarks/longmemeval_runner.py` (config block; new helper after `decompose_query`)
- Test: `tests/test_e012_levers.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_e012_levers.py`:

```python
def test_self_verify_returns_revised_when_nonempty():
    mod = _load_runner()
    out = asyncio.run(mod.self_verify_answer(_FakeClient("corrected answer"), "ctx", "Q", "draft answer"))
    assert out == "corrected answer"


def test_self_verify_keeps_draft_on_error():
    mod = _load_runner()
    out = asyncio.run(mod.self_verify_answer(_BoomClient(), "ctx", "Q", "draft answer"))
    assert out == "draft answer"


def test_self_verify_keeps_draft_when_revision_empty():
    mod = _load_runner()
    out = asyncio.run(mod.self_verify_answer(_FakeClient("   "), "ctx", "Q", "draft answer"))
    assert out == "draft answer"


def test_self_verify_noops_on_empty_draft():
    mod = _load_runner()
    out = asyncio.run(mod.self_verify_answer(_FakeClient("anything"), "ctx", "Q", ""))
    assert out == ""
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_e012_levers.py -k self_verify -v`
Expected: FAIL with `AttributeError: module 'lme_runner' has no attribute 'self_verify_answer'`.

- [ ] **Step 3: Add the config flag**

After the `DECOMPOSE_MODEL = ...` line added in Task 1, add:

```python
# E-012 lever 3: CoVe-style answer self-verification. After the first answer,
# one extra generator pass keeps the draft if it is fully supported by the
# context, else rewrites it from the context. Default off.
SELF_VERIFY = os.environ.get("TAOSMD_SELF_VERIFY", "0") == "1"
```

- [ ] **Step 4: Add the VERIFY_PROMPT and self_verify_answer helper**

After the `decompose_query` function, add:

```python
VERIFY_PROMPT = """You are checking a draft answer against the context.
If every part of the draft answer is supported by the context, repeat the draft answer exactly.
If any part is unsupported or contradicted by the context, write a corrected answer using ONLY the context.
Answer concisely in 1-2 sentences, no explanation. /no_think

Context:
{context}

Question: {question}

Draft answer: {answer}

Final answer:"""


async def self_verify_answer(client, context: str, question: str, answer: str) -> str:
    """One CoVe-style verification pass.

    Keeps the draft if it is supported by the context, otherwise returns a
    corrected answer. Falls back to the original draft on empty draft, empty
    revision, or any failure, so it can never make an answer worse by erroring.
    """
    if not answer:
        return answer
    try:
        resp = await client.post(
            f"{REMOTE_LLM_URL}/api/chat",
            json={
                "model": REMOTE_LLM_MODEL,
                "messages": [{"role": "user", "content": VERIFY_PROMPT.format(context=context[:CONTEXT_CHARS], question=question, answer=answer)}],
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_predict": 100},
            },
            timeout=30,
        )
        if resp.status_code == 200:
            revised = resp.json().get("message", {}).get("content", "").strip()
            if revised:
                return revised
    except Exception:
        pass
    return answer
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_e012_levers.py -v`
Expected: all seven tests PASS.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/longmemeval_runner.py tests/test_e012_levers.py
git commit -m "bench(e012): add CoVe-style answer self-verification (lever 3)"
```

---

## Task 4: Wire self-verification after generation

**Files:**
- Modify: `benchmarks/longmemeval_runner.py` (the answer-generation block, currently ~line 275)

- [ ] **Step 1: Add the verification pass after the first answer**

In `run_benchmark`, find this exact line (currently line 275):

```python
            answer = await llm_answer(llm_client, full_context, question)
```

and add immediately after it:

```python
            if SELF_VERIFY:
                answer = await self_verify_answer(llm_client, full_context, question, answer)
```

- [ ] **Step 2: Verify the full test suite for the new helpers still passes**

Run: `.venv/bin/python -m pytest tests/test_e012_levers.py -v`
Expected: all seven tests PASS (the wiring change does not affect the unit tests, which call the helpers directly; this confirms no syntax error was introduced).

- [ ] **Step 3: Commit**

```bash
git add benchmarks/longmemeval_runner.py
git commit -m "bench(e012): run self-verification pass when enabled (lever 3)"
```

---

## Task 5: Ablation matrix runner

**Files:**
- Create: `benchmarks/e012_ablation.sh`

- [ ] **Step 1: Write the matrix runner**

Create `benchmarks/e012_ablation.sh`:

```bash
#!/usr/bin/env bash
# E-012 score-up ablation. Sweeps the lever arms SEQUENTIALLY (one Ollama job
# at a time per the GPU-contention rule) and records each arm's overall and
# per-type Judge accuracy.
#
# Usage: benchmarks/e012_ablation.sh [LIMIT]
#   LIMIT defaults to 100 (screening pass). Use 500 for the full oracle set.
#
# Required env (export before running):
#   TAOSMD_OLLAMA_MODEL   generator (e.g. qwen3.5:9b)
#   TAOSMD_JUDGE_MODEL    external judge (e.g. qwen3:4b-instruct-2507)
#   TAOSMD_OLLAMA_URL     Ollama base URL (default http://localhost:11434)
# Fixed substrate (depth + rerank context-release = lever 1) is set per arm.
set -u
LIMIT="${1:-100}"
PY=".venv/bin/python"
OUT="benchmarks/results/e012_ablation_$(date -u +%Y%m%d_%H%M%S).log"
RUNNER="benchmarks/longmemeval_runner.py"

run_arm () {
  local name="$1"; shift
  echo "==================== ARM: $name (limit=$LIMIT) ====================" | tee -a "$OUT"
  # Reset all levers off, then apply this arm's env (passed as VAR=VAL args).
  env -u TAOSMD_DECOMPOSE -u TAOSMD_SELF_VERIFY -u TAOSMD_RERANK \
      TAOSMD_ASSEMBLE_TOKENS="${TAOSMD_ASSEMBLE_TOKENS:-4000}" \
      TAOSMD_RETRIEVE_LIMIT="${TAOSMD_RETRIEVE_LIMIT:-12}" \
      TAOSMD_FTS_LIMIT="${TAOSMD_FTS_LIMIT:-5}" \
      TAOSMD_CONTEXT_CHARS="${TAOSMD_CONTEXT_CHARS:-16000}" \
      "$@" \
      "$PY" "$RUNNER" --llm --limit "$LIMIT" 2>&1 | tee -a "$OUT"
  echo "" | tee -a "$OUT"
}

# Arms. Lever 1 = TAOSMD_RERANK=1, lever 2 = TAOSMD_DECOMPOSE=1, lever 3 = TAOSMD_SELF_VERIFY=1.
run_arm "baseline"
run_arm "L1_context_release"  TAOSMD_RERANK=1
run_arm "L2_decompose"        TAOSMD_DECOMPOSE=1
run_arm "L3_self_verify"      TAOSMD_SELF_VERIFY=1
run_arm "L1+L2"               TAOSMD_RERANK=1 TAOSMD_DECOMPOSE=1
run_arm "L1+L3"               TAOSMD_RERANK=1 TAOSMD_SELF_VERIFY=1
run_arm "L2+L3"               TAOSMD_DECOMPOSE=1 TAOSMD_SELF_VERIFY=1
run_arm "L1+L2+L3"            TAOSMD_RERANK=1 TAOSMD_DECOMPOSE=1 TAOSMD_SELF_VERIFY=1

echo "Ablation complete. Full log: $OUT" | tee -a "$OUT"
echo "Per-arm overall accuracy:" | tee -a "$OUT"
grep -E "ARM:|Overall:" "$OUT" | tee -a "$OUT"
```

- [ ] **Step 2: Make it executable and syntax-check it**

```bash
chmod +x benchmarks/e012_ablation.sh
bash -n benchmarks/e012_ablation.sh && echo "syntax OK"
```
Expected: `syntax OK`.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/e012_ablation.sh
git commit -m "bench(e012): ablation matrix runner (8 lever arms, sequential)"
```

---

## Task 6: Final review and run handoff

**Files:** none (verification only)

- [ ] **Step 1: Run the full new-helper test suite once more**

Run: `.venv/bin/python -m pytest tests/test_e012_levers.py -v`
Expected: 7 passed.

- [ ] **Step 2: Confirm the baseline arm is unchanged behavior**

Verify by inspection that with all three flags unset, `run_benchmark` takes the exact original path: `decompose_query` is skipped (the `else` branch), `self_verify_answer` is skipped, rerank is gated by `RERANK`. The baseline arm therefore reproduces the 47.2% control.

- [ ] **Step 3: Hand off to the run**

The build is done and GPU-free. The matrix run is a separate, GPU-gated step (coordinate with @taOS via the lease protocol): screen all arms at `--limit` 100 first to find which levers move the number, then full-500 on the baseline plus the winning arm(s), then the bigger-generator arm (`TAOSMD_OLLAMA_MODEL=qwen3.6:35b-a3b-q4_K_M`) on the winner, then the tri-judge firm-up. Record the result as an F or N finding in the research report with provenance, either way.

---

## Self-Review

**Spec coverage:** Lever 1 (context release) is the existing rerank/depth, exercised as arms in Task 5 (no build needed, per spec). Lever 2 (decomposition) = Tasks 1, 2. Lever 3 (self-verify) = Tasks 3, 4. The bigger-generator arm = Task 6 Step 3 handoff (env only, no code). Ablation matrix = Task 5. Judge protocol and tri-judge firm-up = Task 6 Step 3. All spec sections map to a task.

**Placeholder scan:** No TBD/TODO; every code step shows full code; the only "skip" is the optionally-absent dataset in Task 2 Step 2, which has an explicit fallback.

**Type consistency:** `decompose_query(client, question) -> list` and `self_verify_answer(client, context, question, answer) -> str` are used with matching signatures in the wiring (Tasks 2, 4) and the tests. Env flag names (`TAOSMD_DECOMPOSE`, `TAOSMD_DECOMPOSE_MODEL`, `TAOSMD_SELF_VERIFY`) match between the runner, the helpers, and the ablation script.
