"""Tests for the LongMemEval runner's retrieval wiring (unblocks E-030).

The runner used to assemble its context by hand (ContextAssembler +
archive.search_fts + vmem.search + reranker) and never called
taosmd.retrieval.retrieve(), so setting the graph_expansion control had no
effect and both E-030 arms would have been byte-identical. These tests pin
the wiring:

  1. The vector stage goes through retrieve().
  2. graph_expansion is forwarded only when the flag is set (default 0 stays
     off, and the kwarg is omitted entirely so the runner still imports on a
     checkout predating PR #191).
  3. The derived fact block reaches the assembled context.
  4. The legacy path refuses graph_expansion instead of silently ignoring it.
  5. At graph_expansion=0 the wired path is parameterised to match the legacy
     stage (the anchor condition), including the candidate pool width.
  6. The signature probe REFUSES the run, with a non-zero exit, on a checkout
     whose retrieve() has no graph_expansion parameter. The EventQA sibling
     test only asserted the probe returns a bool and never asserted the abort,
     which left the safety-critical behaviour untested.
  7. The CLI exposes --graph-expansion / --retrieval-path /
     --report-retrieval-delta with the documented defaults.

The LongMemEval-S / oracle dataset is gitignored and generally absent, so
every test here uses synthetic fixtures and never touches benchmarks/data.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_PATH = REPO_ROOT / "benchmarks" / "longmemeval_runner.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("longmemeval_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeVectorMemory:
    """Minimal vector source: returns fixed hits in retrieve()'s expected shape."""

    def __init__(self, texts):
        self.texts = texts
        self.calls = []

    async def search(self, query, limit=5, hybrid=True, fusion="boost",
                     project=None, search_agents=None):
        self.calls.append({"query": query, "limit": limit, "fusion": fusion})
        return [
            {"id": i, "text": t, "similarity": 1.0 - i * 0.01, "metadata": {}}
            for i, t in enumerate(self.texts[:limit])
        ]


class _FakeArchive:
    async def search_fts(self, term, limit=5):
        return []


class _FakeKG:
    pass


class _RecordingAssembler:
    """Stands in for ContextAssembler; contributes nothing to the context."""

    def __init__(self, kg=None, archive=None):
        pass

    async def assemble(self, query, depth="auto", max_total_tokens=4000):
        return {"context": ""}


@pytest.fixture()
def runner(monkeypatch):
    mod = _load_runner()
    monkeypatch.setattr(mod, "ContextAssembler", _RecordingAssembler)
    return mod


def _context(mod, **kwargs):
    vmem = kwargs.pop("vmem", None) or _FakeVectorMemory(["alpha chunk", "beta chunk"])
    return asyncio.run(
        mod.retrieve_context(
            "who met the captain", _FakeKG(), _FakeArchive(), vmem, **kwargs
        )
    ), vmem


# ---------------------------------------------------------------------------
# 1-2. the vector stage goes through retrieve(), carrying graph_expansion
# ---------------------------------------------------------------------------

def test_default_path_calls_retrieve_without_graph_expansion(runner, monkeypatch):
    seen = {}

    async def fake_retrieve(query, **kwargs):
        seen.update(kwargs)
        seen["query"] = query
        return [{"text": "alpha chunk"}]

    monkeypatch.setattr(runner, "_retrieve", fake_retrieve)
    ctx, _ = _context(runner)

    assert seen["query"] == "who met the captain"
    assert seen["memory_layers"] == ["vector"]
    # The KG must be in sources even though only the vector layer is searched:
    # _append_graph_expansion reads sources["kg"] regardless.
    assert set(seen["sources"]) == {"vector", "kg"}
    # Default arm must not pass the kwarg at all, so the runner still imports
    # and runs on a checkout whose retrieve() predates the control.
    assert "graph_expansion" not in seen
    # Claims-gating must stay off: the legacy stage never verified, and turning
    # it on here would silently change what the benchmark measures.
    assert seen.get("verify", False) is False
    assert "alpha chunk" in ctx


def test_flag_forwards_graph_expansion_to_retrieve(runner, monkeypatch):
    seen = {}

    async def fake_retrieve(query, **kwargs):
        seen.update(kwargs)
        return [{"text": "alpha chunk"}]

    monkeypatch.setattr(runner, "_retrieve", fake_retrieve)
    _context(runner, graph_expansion=512)

    assert seen["graph_expansion"] == 512


# ---------------------------------------------------------------------------
# 3. the derived fact block reaches the generator context
# ---------------------------------------------------------------------------

def test_graph_expansion_block_reaches_assembled_context(runner, monkeypatch):
    async def fake_retrieve(query, **kwargs):
        hits = [{"text": "alpha chunk", "source": "vector"}]
        if kwargs.get("graph_expansion"):
            hits.append({
                "text": "captain commands ship",
                "source": "kg_expansion",
                "derived": True,
            })
        return hits

    monkeypatch.setattr(runner, "_retrieve", fake_retrieve)

    off, _ = _context(runner, graph_expansion=0)
    on, _ = _context(runner, graph_expansion=512)

    assert "captain commands ship" not in off
    assert "captain commands ship" in on
    assert off != on, "the two E-030 arms must not produce identical context"


# ---------------------------------------------------------------------------
# 4. the legacy path refuses the control rather than ignoring it
# ---------------------------------------------------------------------------

def test_legacy_path_rejects_graph_expansion(runner):
    with pytest.raises(ValueError, match="graph_expansion"):
        _context(runner, retrieval_path="legacy", graph_expansion=512)


def test_decompose_rejects_graph_expansion(runner, monkeypatch):
    """The sub-query union truncates, which can drop the derived block."""
    monkeypatch.setattr(runner, "DECOMPOSE", True)
    with pytest.raises(ValueError, match="graph_expansion"):
        _context(runner, graph_expansion=512, llm_client=object())


# ---------------------------------------------------------------------------
# 5. anchor condition: the wired default is parameterised like the legacy stage
# ---------------------------------------------------------------------------

def test_wired_default_matches_legacy_context(runner):
    """With no reranker and distinct chunks, wired and legacy contexts agree."""
    texts = ["alpha chunk one", "beta chunk two", "gamma chunk three"]
    wired, vmem_wired = _context(runner, vmem=_FakeVectorMemory(texts))
    legacy, vmem_legacy = _context(
        runner, vmem=_FakeVectorMemory(texts), retrieval_path="legacy"
    )

    assert wired == legacy
    # Both stages must fetch the same candidate pool: retrieve() would default
    # to limit * 3, so candidate_top_k has to pin it to RETRIEVE_LIMIT.
    assert vmem_wired.calls[0]["limit"] == vmem_legacy.calls[0]["limit"]
    assert vmem_wired.calls[0]["limit"] == runner.RETRIEVE_LIMIT


def test_wired_decompose_matches_legacy_decompose(runner, monkeypatch):
    """Decomposed retrieval must union/dedup/truncate identically either way."""
    texts = ["alpha chunk one", "beta chunk two", "gamma chunk three"]

    async def fake_decompose(client, question):
        return ["sub one", "sub two"]

    monkeypatch.setattr(runner, "DECOMPOSE", True)
    monkeypatch.setattr(runner, "decompose_query", fake_decompose)

    wired, vmem_wired = _context(
        runner, vmem=_FakeVectorMemory(texts), llm_client=object()
    )
    legacy, vmem_legacy = _context(
        runner, vmem=_FakeVectorMemory(texts), llm_client=object(),
        retrieval_path="legacy",
    )

    assert wired == legacy
    assert [c["query"] for c in vmem_wired.calls] == [
        c["query"] for c in vmem_legacy.calls
    ]
    assert all(c["limit"] == runner.RETRIEVE_LIMIT for c in vmem_wired.calls)


def test_retrieve_supports_graph_expansion_reports_installed_signature(runner):
    # Never raises; the answer depends on whether PR #191 is in the checkout.
    assert isinstance(runner.retrieve_supports_graph_expansion(), bool)


# ---------------------------------------------------------------------------
# 6. the probe must ABORT the run, with a non-zero exit
# ---------------------------------------------------------------------------

def _args(**overrides):
    base = dict(
        limit=1,
        type=None,
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
        runner, "load_dataset",
        lambda: opened.append("loaded") or [],
    )

    with pytest.raises(SystemExit) as exc:
        asyncio.run(runner.run_benchmark(args=_args()))

    # Non-zero exit: a chain that checks $? must read a refusal as a failure.
    # (The EventQA sibling uses a bare `return`, so it exits 0 and a refusal
    # reads as success; see the PR body.)
    assert exc.value.code != 0
    assert opened == [], "the dataset must not even be loaded after a refusal"


def test_supported_graph_expansion_does_not_abort(runner, monkeypatch):
    monkeypatch.setattr(runner, "retrieve_supports_graph_expansion", lambda: True)
    monkeypatch.setattr(runner, "load_dataset", lambda: [])

    # Empty dataset short-circuits before any store is created; the point is
    # only that the probe did not raise SystemExit.
    asyncio.run(runner.run_benchmark(args=_args()))


def test_zero_graph_expansion_never_probes(runner, monkeypatch):
    """graph_expansion=0 must run on any checkout, probe or no probe."""
    def _boom():
        raise AssertionError("probe must not gate the default arm")

    monkeypatch.setattr(runner, "retrieve_supports_graph_expansion", _boom)
    monkeypatch.setattr(runner, "load_dataset", lambda: [])

    asyncio.run(runner.run_benchmark(args=_args(graph_expansion=0)))


# ---------------------------------------------------------------------------
# 7. CLI surface
# ---------------------------------------------------------------------------

def test_cli_exposes_the_e030_arms(runner, monkeypatch):
    captured = {}

    monkeypatch.setattr(runner.asyncio, "run", lambda coro: None)
    monkeypatch.setattr(
        runner, "run_benchmark", lambda args: captured.update(vars(args))
    )
    monkeypatch.setattr(
        sys, "argv",
        ["longmemeval_runner.py", "--graph-expansion", "512", "--limit", "2"],
    )
    runner.main()

    assert captured["graph_expansion"] == 512
    assert captured["retrieval_path"] == "retrieve"
    assert captured["limit"] == 2


def test_cli_defaults(runner, monkeypatch):
    captured = {}
    monkeypatch.setattr(runner.asyncio, "run", lambda coro: None)
    monkeypatch.setattr(
        runner, "run_benchmark", lambda args: captured.update(vars(args))
    )
    monkeypatch.setattr(sys, "argv", ["longmemeval_runner.py"])
    runner.main()

    assert captured["graph_expansion"] == 0
    assert captured["retrieval_path"] == "retrieve"
    assert captured["report_retrieval_delta"] is False


def test_cli_rejects_unknown_retrieval_path(runner, monkeypatch):
    monkeypatch.setattr(runner.asyncio, "run", lambda coro: None)
    monkeypatch.setattr(sys, "argv", ["longmemeval_runner.py", "--retrieval-path", "hand"])
    with pytest.raises(SystemExit):
        runner.main()


def test_summarize_retrieval_delta_reports_identity(runner):
    results = [
        {"retrieval_delta": {"legacy_chars": 10, "wired_chars": 10, "identical": True}},
        {"retrieval_delta": {"legacy_chars": 10, "wired_chars": 12, "identical": False}},
        {"retrieval_delta": None},
    ]
    summary = runner.summarize_retrieval_delta(results)
    assert summary["n"] == 2
    assert summary["identical"] == 1
    assert summary["identical_pct"] == 50.0
    assert runner.summarize_retrieval_delta([{"correct": True}]) is None
