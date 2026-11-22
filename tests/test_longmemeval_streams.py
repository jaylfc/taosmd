"""Tests for TAOSMD_LME_STREAMS environment switch in longmemeval_runner.py.

The runner builds answer context from three streams: assembler, FTS, vector.
TAOSMD_LME_STREAMS selects a subset (comma-separated, order-insensitive,
whitespace-tolerant). This test suite validates the parsing, the call gating,
and the context composition.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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
    """Minimal vector source that records calls."""

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
    """Minimal archive that records FTS calls."""

    def __init__(self, fts_rows=None):
        self.fts_rows = fts_rows or []
        self.fts_calls = []

    async def search_fts(self, term, limit=5):
        self.fts_calls.append({"term": term, "limit": limit})
        return list(self.fts_rows)


class _FakeKG:
    pass


class _RecordingAssembler:
    """Stands in for ContextAssembler; records assemble calls."""

    def __init__(self, kg=None, archive=None):
        self.assemble_calls = []

    async def assemble(self, query, depth="auto", max_total_tokens=4000):
        self.assemble_calls.append({
            "query": query,
            "depth": depth,
            "max_total_tokens": max_total_tokens,
        })
        return {"context": "ASSEMBLER_TEXT"}


@pytest.fixture()
def runner(monkeypatch):
    mod = _load_runner()
    monkeypatch.setattr(mod, "ContextAssembler", _RecordingAssembler)
    return mod


def _make_env_runner(monkeypatch, env_value=None):
    """Load runner with TAOSMD_LME_STREAMS set to env_value (or unset)."""
    if env_value is not None:
        monkeypatch.setenv("TAOSMD_LME_STREAMS", env_value)
    else:
        monkeypatch.delenv("TAOSMD_LME_STREAMS", raising=False)
    return _load_runner()


async def _call_retrieve_context(mod, streams=None, fts_rows=None, **kwargs):
    """Helper to call retrieve_context with fakes."""
    kg = _FakeKG()
    archive = _FakeArchive(fts_rows)
    vmem = _FakeVectorMemory(["VECTOR_TEXT"])
    assembler = _RecordingAssembler(kg, archive)
    mod.ContextAssembler = lambda kg, archive: assembler

    if streams is not None:
        return await mod.retrieve_context(
            "test question", kg, archive, vmem, streams=streams, **kwargs
        ), assembler, archive, vmem
    else:
        return await mod.retrieve_context(
            "test question", kg, archive, vmem, **kwargs
        ), assembler, archive, vmem


# ---------------------------------------------------------------------------
# 1. Parsing: valid values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("env_value,expected", [
    ("assembler", ["assembler"]),
    ("fts", ["fts"]),
    ("vector", ["vector"]),
    ("assembler,fts", ["assembler", "fts"]),
    ("assembler,vector", ["assembler", "vector"]),
    ("fts,vector", ["fts", "vector"]),
    ("assembler,fts,vector", ["assembler", "fts", "vector"]),
    ("vector,fts,assembler", ["assembler", "fts", "vector"]),  # order-insensitive
    (" assembler , fts , vector ", ["assembler", "fts", "vector"]),  # whitespace-tolerant
    ("fts, assembler", ["assembler", "fts"]),
])
def test_valid_streams_env_values(monkeypatch, env_value, expected):
    """Valid TAOSMD_LME_STREAMS values are parsed into sorted lists."""
    mod = _make_env_runner(monkeypatch, env_value)
    # The parsing happens in a function called from run_benchmark; we test
    # that the parsed value is available and correct by checking the function
    # that run_benchmark will call.
    parsed = mod._parse_streams_env()
    assert parsed == expected


def test_unset_streams_defaults_to_all_three(monkeypatch):
    """Unset TAOSMD_LME_STREAMS defaults to all three streams."""
    mod = _make_env_runner(monkeypatch, None)
    parsed = mod._parse_streams_env()
    assert parsed == ["assembler", "fts", "vector"]


# ---------------------------------------------------------------------------
# 2. Parsing: invalid values -> exit 2 with error message
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_value", [
    "vectr",           # unknown token
    "fts,fts",         # duplicate
    "",                # empty string
    "assembler,unknown",  # mixed valid/invalid
    "vector,",         # trailing comma -> empty token
    ",assembler",      # leading comma -> empty token
])
def test_invalid_streams_env_exits_2(monkeypatch, capsys, bad_value):
    """Invalid TAOSMD_LME_STREAMS prints error to stderr and exits 2."""
    mod = _make_env_runner(monkeypatch, bad_value)
    with pytest.raises(SystemExit) as exc:
        mod._parse_streams_env()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "TAOSMD_LME_STREAMS" in err
    assert bad_value in err
    assert "assembler" in err and "fts" in err and "vector" in err


# ---------------------------------------------------------------------------
# 3. Call gating: disabled streams are NOT called
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("streams,expect_assembler,expect_fts,expect_vector", [
    (["vector"], False, False, True),
    (["fts"], False, True, False),
    (["assembler"], True, False, False),
    (["assembler", "fts"], True, True, False),
    (["assembler", "vector"], True, False, True),
    (["fts", "vector"], False, True, True),
    (["assembler", "fts", "vector"], True, True, True),
])
async def test_stream_call_gating(runner, monkeypatch, streams, expect_assembler, expect_fts, expect_vector):
    """Only enabled streams are called; disabled streams are never invoked."""
    # Monkeypatch the retrieve path to avoid network/ollama
    retrieve_vector_results_calls = []

    async def fake_retrieve_vector_results(*args, **kwargs):
        retrieve_vector_results_calls.append(args)
        return [{"text": "VECTOR_TEXT"}]

    monkeypatch.setattr(runner, "retrieve_vector_results", fake_retrieve_vector_results)

    ctx, assembler, archive, vmem = await _call_retrieve_context(runner, streams=streams)

    # Check assembler
    if expect_assembler:
        assert len(assembler.assemble_calls) == 1
        assert "ASSEMBLER_TEXT" in ctx
    else:
        assert len(assembler.assemble_calls) == 0
        assert "ASSEMBLER_TEXT" not in ctx

    # Check FTS
    if expect_fts:
        assert len(archive.fts_calls) > 0
    else:
        assert len(archive.fts_calls) == 0

    # Check vector (via retrieve_vector_results)
    if expect_vector:
        assert len(retrieve_vector_results_calls) == 1
        assert "VECTOR_TEXT" in ctx
    else:
        assert len(retrieve_vector_results_calls) == 0
        assert "VECTOR_TEXT" not in ctx


# ---------------------------------------------------------------------------
# 4. Context composition: join order is assembler, fts, vector
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_join_order_is_assembler_fts_vector(runner, monkeypatch):
    """Enabled streams are joined in fixed order: assembler, fts, vector."""
    async def fake_retrieve_vector_results(*args, **kwargs):
        return [{"text": "VECTOR_TEXT"}]

    monkeypatch.setattr(runner, "retrieve_vector_results", fake_retrieve_vector_results)

    # Enable fts and vector only (assembler disabled)
    ctx, assembler, archive, vmem = await _call_retrieve_context(
        runner, streams=["fts", "vector"]
    )
    # FTS returns empty in our fake, so context should be just vector
    assert ctx.strip() == "VECTOR_TEXT"

    # Enable assembler and vector (fts disabled)
    ctx, assembler, archive, vmem = await _call_retrieve_context(
        runner, streams=["assembler", "vector"]
    )
    # Should be assembler then vector
    assert ctx == "ASSEMBLER_TEXT VECTOR_TEXT"

    # Enable all three
    ctx, assembler, archive, vmem = await _call_retrieve_context(
        runner, streams=["assembler", "fts", "vector"]
    )
    # FTS returns empty in our fake; original logic produces two spaces
    # between assembler and vector when archive_text is empty:
    # assembler_ctx + " " + "" + " " + vector_text
    assert ctx == "ASSEMBLER_TEXT  VECTOR_TEXT"


# ---------------------------------------------------------------------------
# 5. Unset env == explicit all-three (byte-identical context)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unset_env_matches_explicit_all_three(runner, monkeypatch):
    """Unset TAOSMD_LME_STREAMS produces byte-identical context to explicit all-three."""
    async def fake_retrieve_vector_results(*args, **kwargs):
        return [{"text": "VECTOR_TEXT"}]

    monkeypatch.setattr(runner, "retrieve_vector_results", fake_retrieve_vector_results)

    # Explicit all three
    ctx_explicit, _, _, _ = await _call_retrieve_context(
        runner, streams=["assembler", "fts", "vector"]
    )

    # Unset env (will use default in retrieve_context)
    ctx_default, _, _, _ = await _call_retrieve_context(runner, streams=None)

    assert ctx_explicit == ctx_default


# ---------------------------------------------------------------------------
# 6. Integration: run_benchmark records streams and context_chars
# ---------------------------------------------------------------------------

def _args(**overrides):
    base = dict(
        limit=1,
        type=None,
        llm=False,
        graph_expansion=0,
        retrieval_path="retrieve",
        report_retrieval_delta=False,
        out="",
        writer_url=None,
        writer_model="default",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.mark.asyncio
async def test_run_benchmark_records_streams_and_context_chars(runner, monkeypatch, tmp_path):
    """run_benchmark writes streams list and context_chars to results JSON."""
    # Minimal dataset
    monkeypatch.setattr(runner, "load_dataset", lambda: [{
        "question_type": "temporal",
        "question": "When did Alice move to Paris?",
        "answer": "Alice moved to Paris in 2020",
        "haystack_sessions": [[{"role": "user", "content": "Alice moved to Paris in 2020"}]],
    }])

    # Fast fakes
    async def fake_retrieve_vector_results(*args, **kwargs):
        return [{"text": "VECTOR_TEXT"}]

    monkeypatch.setattr(runner, "retrieve_vector_results", fake_retrieve_vector_results)
    monkeypatch.setattr(runner, "ContextAssembler", _RecordingAssembler)

    out_file = str(tmp_path / "results.json")

    # Run with explicit streams
    await runner.run_benchmark(args=_args(
        out=out_file,
        # We need to pass streams through somehow - the task says to pass
        # stream set into retrieve_context as a parameter. Let's check if
        # run_benchmark picks up the env var.
    ))

    # Read results
    with open(out_file) as f:
        result_doc = json.load(f)

    # Check config block has streams
    assert "streams" in result_doc
    assert result_doc["streams"] == ["assembler", "fts", "vector"]

    # Check each result has context_chars
    for r in result_doc["results"]:
        assert "context_chars" in r
        assert r["context_chars"] == r["retrieved_chars"]


# ---------------------------------------------------------------------------
# 7. The _parse_streams_env function exists and is called from run_benchmark
# ---------------------------------------------------------------------------

def test_parse_streams_env_function_exists(runner):
    """The runner exposes _parse_streams_env function."""
    assert hasattr(runner, "_parse_streams_env")
    assert callable(runner._parse_streams_env)


@pytest.mark.asyncio
async def test_run_benchmark_passes_streams_to_retrieve_context(runner, monkeypatch, tmp_path):
    """run_benchmark reads TAOSMD_LME_STREAMS and passes it to retrieve_context."""
    captured_streams = {}

    async def fake_retrieve_context(question, kg, archive, vmem, llm_client=None,
                                     graph_expansion=0, retrieval_path="retrieve",
                                     streams=None, date_frame="off", haystack_dates=None):
        captured_streams["streams"] = streams
        return "CONTEXT"

    monkeypatch.setattr(runner, "retrieve_context", fake_retrieve_context)
    monkeypatch.setattr(runner, "load_dataset", lambda: [{
        "question_type": "temporal",
        "question": "test?",
        "answer": "ans",
        "haystack_sessions": [[{"role": "user", "content": "content"}]],
    }])

    # Set env var
    monkeypatch.setenv("TAOSMD_LME_STREAMS", "assembler,vector")

    await runner.run_benchmark(args=_args(
        out=str(tmp_path / "results.json"),
    ))

    assert captured_streams["streams"] == ["assembler", "vector"]


# ---------------------------------------------------------------------------
# 8. FTS text flows through the enabled stream
# ---------------------------------------------------------------------------

_FTS_ROWS = [
    {"data_json": "FTS-ALPHA", "summary": ""},
    {"data_json": "FTS-BETA", "summary": ""},
]


@pytest.mark.asyncio
async def test_fts_only_context_contains_fts_text(runner, monkeypatch):
    """streams=['fts'] puts canned FTS text in the context and nothing else."""
    ctx, assembler, archive, vmem = await _call_retrieve_context(
        runner, streams=["fts"], fts_rows=_FTS_ROWS
    )
    assert "FTS-ALPHA" in ctx
    assert "FTS-BETA" in ctx
    assert "ASSEMBLER_TEXT" not in ctx
    assert "VECTOR_TEXT" not in ctx


@pytest.mark.asyncio
async def test_all_streams_context_contains_fts_text(runner, monkeypatch):
    """All three streams join in fixed order: assembler, fts, vector."""

    async def fake_retrieve_vector_results(*args, **kwargs):
        return [{"text": "VECTOR_TEXT"}]

    monkeypatch.setattr(runner, "retrieve_vector_results", fake_retrieve_vector_results)

    ctx, assembler, archive, vmem = await _call_retrieve_context(
        runner,
        streams=["assembler", "fts", "vector"],
        fts_rows=_FTS_ROWS,
    )
    assert "ASSEMBLER_TEXT" in ctx
    assert "FTS-ALPHA" in ctx
    assert "VECTOR_TEXT" in ctx
    assert (
        ctx.index("ASSEMBLER_TEXT")
        < ctx.index("FTS-ALPHA")
        < ctx.index("VECTOR_TEXT")
    )


@pytest.mark.asyncio
async def test_fts_disabled_omits_fts_text(runner, monkeypatch):
    """streams without 'fts' never calls search_fts and omits FTS text."""

    async def fake_retrieve_vector_results(*args, **kwargs):
        return [{"text": "VECTOR_TEXT"}]

    monkeypatch.setattr(runner, "retrieve_vector_results", fake_retrieve_vector_results)

    ctx, assembler, archive, vmem = await _call_retrieve_context(
        runner, streams=["assembler", "vector"], fts_rows=_FTS_ROWS
    )
    assert "FTS-ALPHA" not in ctx
    assert "FTS-BETA" not in ctx
    assert archive.fts_calls == []
    assert "ASSEMBLER_TEXT" in ctx
    assert "VECTOR_TEXT" in ctx


@pytest.mark.asyncio
async def test_retrieval_delta_call_gets_streams(monkeypatch, tmp_path):
    """--report-retrieval-delta with TAOSMD_LME_STREAMS='vector' keeps the
    legacy/delta retrieve_context call on the vector stream only: no assembler
    call and no search_fts call."""
    mod = _make_env_runner(monkeypatch, "vector")
    monkeypatch.setattr(mod, "ContextAssembler", _RecordingAssembler)

    archive_instances = []

    class _FakeKG:
        def __init__(self, *a, **k):
            pass

        async def init(self):
            pass

        async def close(self):
            pass

    class _FakeArchiveStore:
        def __init__(self, *a, **k):
            archive_instances.append(self)
            self.fts_calls = []
            self.records = []

        async def init(self):
            pass

        async def search_fts(self, term, limit=5):
            self.fts_calls.append({"term": term, "limit": limit})
            return []

        async def record(self, *a, **k):
            self.records.append((a, k))

        async def close(self):
            pass

    class _FakeVmem:
        def __init__(self, *a, **k):
            pass

        async def init(self, http_client=None):
            pass

        async def add(self, *a, **k):
            pass

        async def close(self):
            pass

    assembler = _RecordingAssembler()
    mod.ContextAssembler = lambda kg, archive: assembler

    async def fake_retrieve_vector_results(*args, **kwargs):
        return [{"text": "VECTOR_TEXT"}]

    monkeypatch.setattr(mod, "retrieve_vector_results", fake_retrieve_vector_results)
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", _FakeKG)
    monkeypatch.setattr(mod, "ArchiveStore", _FakeArchiveStore)
    monkeypatch.setattr(mod, "VectorMemory", _FakeVmem)
    monkeypatch.setattr(
        mod,
        "load_dataset",
        lambda: [
            {
                "question_type": "temporal",
                "question": "test question",
                "answer": "ans",
                "haystack_sessions": [],
            }
        ],
    )

    out_file = str(tmp_path / "results.json")
    await mod.run_benchmark(
        args=_args(
            out=out_file,
            report_retrieval_delta=True,
            limit=1,
        )
    )

    assert assembler.assemble_calls == []
    for inst in archive_instances:
        assert inst.fts_calls == []