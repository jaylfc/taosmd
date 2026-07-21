# JOB-004: Make the EventQA runner exit non-zero when it refuses to run

Read docs/agent-jobs/README.md first and follow its absolute rules.

- Tracks issue #205.
- Branch: `fix/eventqa-runner-exit-code` (from `origin/master`)
- Commit message: `fix(bench): exit non-zero when the EventQA runner refuses to run`
- PR title: `fix(bench): exit non-zero when the EventQA runner refuses to run`
- Allowed files: `benchmarks/eventqa_runner.py`,
  `tests/test_eventqa_retrieve_wiring.py` ONLY.

## The bug

`benchmarks/eventqa_runner.py`, in `async def run(args)` (around line 715),
refuses to run when the checkout does not support the requested lever. The
refusal itself is correct: without it the two E-025 arms would be
byte-identical and the benchmark would report a null result that is really an
instrument failure.

The problem is how it refuses. It prints an error and then does a bare
`return`, so the process exits **0**. An overnight chain that runs arms in
sequence and checks `$?` reads "I refused to run" as "this arm succeeded",
carries on, and the missing arm only shows up much later as an absent result
file rather than as a failure at the point it happened.

The sibling LongMemEval runner already has the correct shape. There is a
worked reference implementation on branch `feat/longmemeval-retrieve-wiring`
(PR #204): see `benchmarks/longmemeval_runner.py` around line 534, and its
tests in `tests/test_longmemeval_retrieve_wiring.py` around line 251. Read
both before you start. You are copying that pattern, not inventing one.

Fetch the reference without checking it out:

```
git fetch origin feat/longmemeval-retrieve-wiring
git show origin/feat/longmemeval-retrieve-wiring:benchmarks/longmemeval_runner.py | sed -n '525,545p'
git show origin/feat/longmemeval-retrieve-wiring:tests/test_longmemeval_retrieve_wiring.py | sed -n '236,285p'
```

Do NOT branch from `feat/longmemeval-retrieve-wiring`, and do not merge it
into your branch. Branch from `origin/master` as usual. The reference is for
reading only.

## Two refusals to fix, and one return you must NOT touch

Inside `async def run(args)` there are exactly two error paths that print an
`ERROR:` line and then `return`. Both must exit non-zero:

1. around line 715: the `graph_expansion` support probe
2. around line 726: `if not rows:` after `load_eventqa_rows(...)`

There is a THIRD bare `return` further down (around line 765), at the end of
the `if not args.llm:` dry-run summary block. That one is a SUCCESS path: a
dry wiring check that completed normally. It must keep exiting 0. Do not
touch it. If you find yourself editing anything near the words
`DRY WIRING CHECK`, you have the wrong return.

## Steps

1. `git fetch origin && git checkout -b fix/eventqa-runner-exit-code origin/master`
2. Read `benchmarks/eventqa_runner.py` from line 690 to line 790 before
   editing, so you can see all three returns and tell them apart.
3. Confirm `sys` is already imported (it is, around line 58). Do not add the
   import again.
4. Write the failing tests FIRST. Append them to the END of the existing file
   `tests/test_eventqa_retrieve_wiring.py`. That file already has a `runner`
   fixture and loads the runner module by path; reuse it, do not write a new
   loader.

   The file does NOT currently import `argparse`. Add `import argparse` to the
   import block at the top of the file, in alphabetical order (it goes before
   `import asyncio`).

   Then append this block at the end of the file:

```python
# ---------------------------------------------------------------------------
# 7. a refusal must abort with a non-zero exit
# ---------------------------------------------------------------------------

def _args(**overrides):
    base = dict(
        tier="eventqa_65536",
        contexts=1,
        limit=1,
        llm=False,
        graph_expansion=512,
        retrieval_path="retrieve",
        report_retrieval_delta=False,
        out="",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_unsupported_graph_expansion_aborts_the_run(runner, monkeypatch):
    """The safety-critical case: refuse rather than run two identical arms."""
    monkeypatch.setattr(runner, "retrieve_supports_graph_expansion", lambda: False)

    opened = []
    monkeypatch.setattr(
        runner, "load_eventqa_rows",
        lambda tier, contexts: opened.append("loaded") or [],
    )

    with pytest.raises(SystemExit) as exc:
        asyncio.run(runner.run(_args()))

    # Non-zero exit: a chain that checks $? must read a refusal as a failure.
    assert exc.value.code != 0
    assert opened == [], "the dataset must not even be loaded after a refusal"


def test_empty_dataset_aborts_the_run(runner, monkeypatch):
    """No rows loaded is an instrument failure, not an empty success."""
    monkeypatch.setattr(runner, "retrieve_supports_graph_expansion", lambda: True)
    monkeypatch.setattr(runner, "load_eventqa_rows", lambda tier, contexts: [])

    with pytest.raises(SystemExit) as exc:
        asyncio.run(runner.run(_args()))

    assert exc.value.code != 0


def test_zero_graph_expansion_never_probes(runner, monkeypatch):
    """graph_expansion=0 must run on any checkout, probe or no probe."""
    def _boom():
        raise AssertionError("probe must not gate the default arm")

    monkeypatch.setattr(runner, "retrieve_supports_graph_expansion", _boom)
    monkeypatch.setattr(runner, "load_eventqa_rows", lambda tier, contexts: [])

    # Still aborts, but on the empty dataset, not on the probe.
    with pytest.raises(SystemExit):
        asyncio.run(runner.run(_args(graph_expansion=0)))
```

5. Run just the new tests and watch them FAIL:
   `python3 -m pytest tests/test_eventqa_retrieve_wiring.py -q`
   The two abort tests must fail with "DID NOT RAISE SystemExit". If they
   pass before you have changed the runner, something is different from what
   this job describes: STOP per rule 9.
6. Now fix `benchmarks/eventqa_runner.py`. Change ONLY the two `return`
   statements named above to `sys.exit(1)`.

   The probe refusal becomes:

```python
    # Refuse rather than silently run two identical arms. Exits non-zero on
    # purpose so an automated chain checking $? reads a refusal as a failure.
    if args.graph_expansion and not retrieve_supports_graph_expansion():
        print(
            "  ERROR: the installed taosmd.retrieval.retrieve() has no "
            "graph_expansion parameter, so both E-025 arms would be identical. "
            "Run this on a checkout that includes the bi-temporal fact-readback "
            "work (PR #191).",
            file=sys.stderr,
        )
        sys.exit(1)
```

   The empty-rows refusal becomes:

```python
    rows = load_eventqa_rows(args.tier, args.contexts)
    if not rows:
        print("  ERROR: no rows loaded for tier", args.tier, file=sys.stderr)
        sys.exit(1)
```

   Do not change the wording of either message. Do not add any other
   `sys.exit` call anywhere in the file.
7. Re-run the file: `python3 -m pytest tests/test_eventqa_retrieve_wiring.py -q`
   (all green), then the FULL suite
   `python3 -m pytest -q -m "not slow"`, then
   `python3 -m ruff check benchmarks/eventqa_runner.py tests/test_eventqa_retrieve_wiring.py`.
8. One commit, push, open the PR. PR body: the bug (a refusal exited 0 so a
   chain read it as success), the fix (`sys.exit(1)` on both refusal paths,
   the dry-run success return untouched), the reference it copies
   (PR #204 for the LongMemEval runner), and the test count. Reference
   issue #205. Do not merge.

## STOP conditions

- If `run()` is not an `async def` taking `args`, or the two `ERROR:` returns
  are not where this job says, STOP and describe what you found in the PR
  body.
- If the full suite has failures you did not cause, STOP, open the PR with
  what you have, and say which tests failed.
- Issue #205 also asks whether OTHER runners under `benchmarks/` refuse
  without a non-zero exit. That is a survey, not this job. Do NOT go looking,
  and do NOT edit any other runner. If you happen to notice one, write the
  filename in the PR body as a note and leave it alone.
