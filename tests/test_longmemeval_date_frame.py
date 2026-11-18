"""Tests for the LongMemEval runner's date framing (E-036).

The TAOSMD_LME_DATE_FRAME environment variable controls how dates are presented
to the generator and how session dates are handled at ingest and read time.

Five arms:
- off (or unset): current behaviour, no date information added
- question: adds "Today is <question_date>." line to the answer prompt
- sessions: prefixes archive/vmem content with session dates, orders vector results chronologically
- both: question + sessions
- labelled: both + at read time, rewrites latest session prefix to "most recent" and others to "historical"

All invalid values exit 2 with an error message (no silent fallback).
"""

from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_PATH = REPO_ROOT / "benchmarks" / "longmemeval_runner.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("longmemeval_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Fixtures: minimal fakes for the stores
# ---------------------------------------------------------------------------

class _FakeVectorMemory:
    """Records add() calls and returns controlled search results."""

    def __init__(self, search_results=None, **kwargs):
        self.kwargs = kwargs
        self.adds = []  # list of (text, metadata)
        self._search_results = search_results or []
        self._search_call_count = 0

    async def init(self, http_client=None):
        return None

    async def add(self, text, metadata=None):
        self.adds.append((text, dict(metadata or {})))
        return 1

    async def embed(self, text, task="search_document"):
        return [1.0, 0.0, 0.0]

    async def search(self, query, limit=5, hybrid=True, fusion="boost", project=None, search_agents=None):
        self._search_call_count += 1
        if self._search_results:
            return [dict(r) for r in self._search_results]
        return []

    async def close(self):
        return None

    def set_search_results(self, results):
        self._search_results = results


class _FakeArchive:
    def __init__(self):
        self.records = []

    async def init(self):
        return None

    async def record(self, event_type, data, agent_name=None, app_id=None, summary="", project=None):
        self.records.append((event_type, dict(data), summary))
        return 1

    async def search_fts(self, term, limit=5):
        return []

    async def close(self):
        return None


class _FakeKG:
    def __init__(self):
        pass

    async def init(self):
        return None

    async def close(self):
        return None


class _FakeAssembler:
    def __init__(self, kg=None, archive=None):
        pass

    async def assemble(self, query, depth="auto", max_total_tokens=4000):
        return {"context": "ASSEMBLER_CONTEXT"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import httpx


def _sample_item(question_date="2023/05/20 (Sat) 02:21", haystack_dates=None):
    """Return a minimal LongMemEval item with the required date fields."""
    if haystack_dates is None:
        haystack_dates = [
            "2023/05/10 (Wed) 10:00",
            "2023/05/15 (Mon) 14:30",
            "2023/05/20 (Sat) 02:21",
        ]
    return {
        "question": "When did Alice move to Paris?",
        "question_type": "temporal",
        "answer": "Alice moved to Paris in 2020",
        "question_date": question_date,
        "haystack_dates": haystack_dates,
        "haystack_sessions": [
            [{"role": "user", "content": "Session A content"}],
            [{"role": "user", "content": "Session B content"}],
            [{"role": "user", "content": "Session C content"}],
        ],
    }


def _runner_with_fakes(monkeypatch, date_frame_env=None):
    """Load runner with fakes and optional date frame env."""
    if date_frame_env is not None:
        monkeypatch.setenv("TAOSMD_LME_DATE_FRAME", date_frame_env)
    else:
        monkeypatch.delenv("TAOSMD_LME_DATE_FRAME", raising=False)

    mod = _load_runner()
    monkeypatch.setattr(mod, "VectorMemory", _FakeVectorMemory)
    monkeypatch.setattr(mod, "ArchiveStore", _FakeArchive)
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", _FakeKG)
    monkeypatch.setattr(mod, "ContextAssembler", _FakeAssembler)
    # Avoid network calls
    monkeypatch.setattr(mod, "process_conversation_turn", AsyncMock(return_value={"method": "regex"}))
    # Mock dataset loading
    monkeypatch.setattr(mod, "load_dataset", lambda: [_sample_item()])
    return mod


class _CapturingHTTPClient:
    """Captures the JSON payload posted to /api/chat."""

    def __init__(self):
        self.last_payload = None
        self.responses = []

    def add_response(self, content="Test answer"):
        self.responses.append({
            "status_code": 200,
            "json": lambda: {"message": {"content": content}}
        })

    async def post(self, url, json=None, timeout=30):
        self.last_payload = json
        return self.responses.pop(0) if self.responses else MagicMock(
            status_code=200,
            json=MagicMock(return_value={"message": {"content": "Test answer"}})
        )

    async def aclose(self):
        pass


def _make_args(tmp_path=None, **overrides):
    base = dict(
        limit=1,
        type="temporal",
        llm=False,
        graph_expansion=0,
        retrieval_path="retrieve",
        report_retrieval_delta=False,
        out=str(tmp_path / "results.json") if tmp_path else "",
        writer_url=None,
        writer_model="default",
    )
    base.update(overrides)
    return type("Args", (), base)


# ---------------------------------------------------------------------------
# 1. _date_frame_from_env parsing and validation
# ---------------------------------------------------------------------------

def test_date_frame_from_env_unset_is_off(monkeypatch):
    """Unset env -> 'off'."""
    monkeypatch.delenv("TAOSMD_LME_DATE_FRAME", raising=False)
    mod = _load_runner()
    assert mod._date_frame_from_env() == "off"


def test_date_frame_from_env_empty_is_off(monkeypatch):
    """Empty-after-strip -> 'off' (treated as unset)."""
    monkeypatch.setenv("TAOSMD_LME_DATE_FRAME", "  ")
    mod = _load_runner()
    assert mod._date_frame_from_env() == "off"


def test_date_frame_from_env_valid_values(monkeypatch):
    """Each of the five valid values is returned as-is."""
    for val in ["off", "question", "sessions", "both", "labelled"]:
        monkeypatch.setenv("TAOSMD_LME_DATE_FRAME", val)
        mod = _load_runner()
        assert mod._date_frame_from_env() == val


def test_date_frame_from_env_invalid_exits_2(monkeypatch, capsys):
    """Invalid value prints error naming the bad value and the five valid ones, exits 2."""
    for bad in ["sesions", "ON", "Question", "bothh", "labeld", "invalid"]:
        monkeypatch.setenv("TAOSMD_LME_DATE_FRAME", bad)
        mod = _load_runner()
        with pytest.raises(SystemExit) as exc:
            mod._date_frame_from_env()
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert bad in err
        for valid in ["off", "question", "sessions", "both", "labelled"]:
            assert valid in err


# ---------------------------------------------------------------------------
# 2. Ingest-time prefixing (sessions, both, labelled arms)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ingest_prefixes_archive_and_vmem_for_sessions_arm(monkeypatch, tmp_path):
    """sessions/both/labelled: every archive.record and vmem.add gets [Session date: <d>] prefix."""
    mod = _runner_with_fakes(monkeypatch, "sessions")
    item = _sample_item()
    vmem = _FakeVectorMemory()
    archive = _FakeArchive()
    kg = _FakeKG()

    monkeypatch.setattr(mod, "VectorMemory", lambda **kw: vmem)
    monkeypatch.setattr(mod, "ArchiveStore", lambda **kw: archive)
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", lambda **kw: kg)

    await mod.run_benchmark(args=_make_args(tmp_path=tmp_path))

    # Check archive records
    for _, data, _ in archive.records:
        content = data.get("content", "")
        assert content.startswith("[Session date: "), f"Archive record missing prefix: {content[:80]}"
        assert any(d in content for d in item["haystack_dates"])

    # Check vmem adds
    for text, meta in vmem.adds:
        assert text.startswith("[Session date: "), f"VMEM add missing prefix: {text[:80]}"
        assert any(d in text for d in item["haystack_dates"])
        assert "session_date" in meta
        assert meta["session_date"] in item["haystack_dates"]


@pytest.mark.asyncio
async def test_ingest_no_prefix_for_off_arm(monkeypatch, tmp_path):
    """off: no date prefixes on archive records or vmem adds."""
    mod = _runner_with_fakes(monkeypatch, "off")
    vmem = _FakeVectorMemory()
    archive = _FakeArchive()
    kg = _FakeKG()

    monkeypatch.setattr(mod, "VectorMemory", lambda **kw: vmem)
    monkeypatch.setattr(mod, "ArchiveStore", lambda **kw: archive)
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", lambda **kw: kg)

    await mod.run_benchmark(args=_make_args(tmp_path=tmp_path))

    for _, data, _ in archive.records:
        content = data.get("content", "")
        assert not content.startswith("[Session date: "), f"Archive record should not have prefix: {content[:80]}"

    for text, meta in vmem.adds:
        assert not text.startswith("[Session date: "), f"VMEM add should not have prefix: {text[:80]}"
        assert "session_date" not in meta


@pytest.mark.asyncio
async def test_ingest_question_arm_no_prefix(monkeypatch, tmp_path):
    """question: no ingest-time prefixes (only affects answer prompt)."""
    mod = _runner_with_fakes(monkeypatch, "question")
    vmem = _FakeVectorMemory()
    archive = _FakeArchive()
    kg = _FakeKG()

    monkeypatch.setattr(mod, "VectorMemory", lambda **kw: vmem)
    monkeypatch.setattr(mod, "ArchiveStore", lambda **kw: archive)
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", lambda **kw: kg)

    await mod.run_benchmark(args=_make_args(tmp_path=tmp_path))

    for _, data, _ in archive.records:
        content = data.get("content", "")
        assert not content.startswith("[Session date: "), f"Archive record should not have prefix: {content[:80]}"

    for text, meta in vmem.adds:
        assert not text.startswith("[Session date: "), f"VMEM add should not have prefix: {text[:80]}"
        assert "session_date" not in meta


# ---------------------------------------------------------------------------
# 3. Answer prompt: "Today is <question_date>." line for question/both/labelled
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_question_arm_adds_today_line_to_prompt(monkeypatch, tmp_path):
    """question/both/labelled: answer prompt contains 'Today is <question_date>.' before Context:"""
    mod = _runner_with_fakes(monkeypatch, "question")

    client = _CapturingHTTPClient()
    client.add_response("Alice moved to Paris in 2020")

    monkeypatch.setattr(mod, "VectorMemory", lambda **kw: _FakeVectorMemory())
    monkeypatch.setattr(mod, "ArchiveStore", lambda **kw: _FakeArchive())
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", lambda **kw: _FakeKG())
    monkeypatch.setattr(mod, "ContextAssembler", _FakeAssembler)
    monkeypatch.setattr(mod, "process_conversation_turn", AsyncMock(return_value={"method": "regex"}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client)

    await mod.run_benchmark(args=_make_args(tmp_path=tmp_path, llm=True))

    payload = client.last_payload
    assert payload is not None
    content = payload["messages"][0]["content"]
    assert "Today is 2023/05/20 (Sat) 02:21." in content
    today_idx = content.index("Today is 2023/05/20 (Sat) 02:21.")
    context_idx = content.index("Context:")
    assert today_idx < context_idx
    assert content[today_idx:context_idx].strip() == "Today is 2023/05/20 (Sat) 02:21."


@pytest.mark.asyncio
async def test_both_arm_adds_today_line_to_prompt(monkeypatch, tmp_path):
    """both: answer prompt contains 'Today is <question_date>.' before Context:"""
    mod = _runner_with_fakes(monkeypatch, "both")

    client = _CapturingHTTPClient()
    client.add_response("Alice moved to Paris in 2020")

    monkeypatch.setattr(mod, "VectorMemory", lambda **kw: _FakeVectorMemory())
    monkeypatch.setattr(mod, "ArchiveStore", lambda **kw: _FakeArchive())
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", lambda **kw: _FakeKG())
    monkeypatch.setattr(mod, "ContextAssembler", _FakeAssembler)
    monkeypatch.setattr(mod, "process_conversation_turn", AsyncMock(return_value={"method": "regex"}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client)

    await mod.run_benchmark(args=_make_args(tmp_path=tmp_path, llm=True))

    payload = client.last_payload
    assert payload is not None
    content = payload["messages"][0]["content"]
    assert "Today is 2023/05/20 (Sat) 02:21." in content
    today_idx = content.index("Today is 2023/05/20 (Sat) 02:21.")
    context_idx = content.index("Context:")
    assert today_idx < context_idx


@pytest.mark.asyncio
async def test_labelled_arm_adds_today_line_to_prompt(monkeypatch, tmp_path):
    """labelled: answer prompt contains 'Today is <question_date>.' before Context:"""
    mod = _runner_with_fakes(monkeypatch, "labelled")

    client = _CapturingHTTPClient()
    client.add_response("Alice moved to Paris in 2020")

    monkeypatch.setattr(mod, "VectorMemory", lambda **kw: _FakeVectorMemory())
    monkeypatch.setattr(mod, "ArchiveStore", lambda **kw: _FakeArchive())
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", lambda **kw: _FakeKG())
    monkeypatch.setattr(mod, "ContextAssembler", _FakeAssembler)
    monkeypatch.setattr(mod, "process_conversation_turn", AsyncMock(return_value={"method": "regex"}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client)

    await mod.run_benchmark(args=_make_args(tmp_path=tmp_path, llm=True))

    payload = client.last_payload
    assert payload is not None
    content = payload["messages"][0]["content"]
    assert "Today is 2023/05/20 (Sat) 02:21." in content
    today_idx = content.index("Today is 2023/05/20 (Sat) 02:21.")
    context_idx = content.index("Context:")
    assert today_idx < context_idx


@pytest.mark.asyncio
async def test_off_arm_no_today_line(monkeypatch, tmp_path):
    """off: answer prompt does NOT contain 'Today is' line."""
    mod = _runner_with_fakes(monkeypatch, "off")

    client = _CapturingHTTPClient()
    client.add_response("Alice moved to Paris in 2020")

    monkeypatch.setattr(mod, "VectorMemory", lambda **kw: _FakeVectorMemory())
    monkeypatch.setattr(mod, "ArchiveStore", lambda **kw: _FakeArchive())
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", lambda **kw: _FakeKG())
    monkeypatch.setattr(mod, "ContextAssembler", _FakeAssembler)
    monkeypatch.setattr(mod, "process_conversation_turn", AsyncMock(return_value={"method": "regex"}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client)

    await mod.run_benchmark(args=_make_args(tmp_path=tmp_path, llm=True))

    payload = client.last_payload
    assert payload is not None
    content = payload["messages"][0]["content"]
    assert "Today is" not in content


@pytest.mark.asyncio
async def test_sessions_arm_no_today_line(monkeypatch, tmp_path):
    """sessions: answer prompt does NOT contain 'Today is' line."""
    mod = _runner_with_fakes(monkeypatch, "sessions")

    client = _CapturingHTTPClient()
    client.add_response("Alice moved to Paris in 2020")

    monkeypatch.setattr(mod, "VectorMemory", lambda **kw: _FakeVectorMemory())
    monkeypatch.setattr(mod, "ArchiveStore", lambda **kw: _FakeArchive())
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", lambda **kw: _FakeKG())
    monkeypatch.setattr(mod, "ContextAssembler", _FakeAssembler)
    monkeypatch.setattr(mod, "process_conversation_turn", AsyncMock(return_value={"method": "regex"}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client)

    await mod.run_benchmark(args=_make_args(tmp_path=tmp_path, llm=True))

    payload = client.last_payload
    assert payload is not None
    content = payload["messages"][0]["content"]
    assert "Today is" not in content


# ---------------------------------------------------------------------------
# 4. retrieve_context: chronological ordering + note line for sessions/both/labelled
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retrieve_context_chronological_order_with_note(monkeypatch):
    """sessions/both/labelled: vector results sorted by session_date ascending, with note line."""
    mod = _load_runner()
    monkeypatch.setenv("TAOSMD_LME_DATE_FRAME", "sessions")
    monkeypatch.setattr(mod, "ContextAssembler", _FakeAssembler)

    haystack_dates = [
        "2023/05/20 (Sat) 02:21",  # latest
        "2023/05/10 (Wed) 10:00",  # earliest
        "2023/05/15 (Mon) 14:30",  # middle
    ]

    fake_results = [
        {"text": "chunk from session 2 (latest)", "metadata": {"session_date": haystack_dates[0]}, "similarity": 0.9},
        {"text": "chunk from session 0 (earliest)", "metadata": {"session_date": haystack_dates[1]}, "similarity": 0.8},
        {"text": "chunk from session 1 (middle)", "metadata": {"session_date": haystack_dates[2]}, "similarity": 0.7},
    ]
    monkeypatch.setattr(mod, "retrieve_vector_results", AsyncMock(return_value=fake_results))

    archive = _FakeArchive()
    kg = _FakeKG()

    ctx = await mod.retrieve_context(
        "test question", kg, archive, None,
        llm_client=None, graph_expansion=0, retrieval_path="retrieve",
        date_frame="sessions", haystack_dates=haystack_dates
    )

    assert "Memories below are in chronological order, oldest first." in ctx

    earliest_idx = ctx.index("chunk from session 0 (earliest)")
    middle_idx = ctx.index("chunk from session 1 (middle)")
    latest_idx = ctx.index("chunk from session 2 (latest)")
    assert earliest_idx < middle_idx < latest_idx


@pytest.mark.asyncio
async def test_retrieve_context_no_ordering_for_off(monkeypatch):
    """off: vector results NOT reordered, no note line."""
    mod = _load_runner()
    monkeypatch.setenv("TAOSMD_LME_DATE_FRAME", "off")
    monkeypatch.setattr(mod, "ContextAssembler", _FakeAssembler)

    haystack_dates = [
        "2023/05/20 (Sat) 02:21",
        "2023/05/10 (Wed) 10:00",
        "2023/05/15 (Mon) 14:30",
    ]

    fake_results = [
        {"text": "chunk from session 2 (latest)", "metadata": {"session_date": haystack_dates[0]}, "similarity": 0.9},
        {"text": "chunk from session 0 (earliest)", "metadata": {"session_date": haystack_dates[1]}, "similarity": 0.8},
        {"text": "chunk from session 1 (middle)", "metadata": {"session_date": haystack_dates[2]}, "similarity": 0.7},
    ]
    monkeypatch.setattr(mod, "retrieve_vector_results", AsyncMock(return_value=fake_results))

    archive = _FakeArchive()
    kg = _FakeKG()

    ctx = await mod.retrieve_context(
        "test question", kg, archive, None,
        llm_client=None, graph_expansion=0, retrieval_path="retrieve",
        date_frame="off", haystack_dates=haystack_dates
    )

    assert "Memories below are in chronological order, oldest first." not in ctx

    latest_idx = ctx.index("chunk from session 2 (latest)")
    earliest_idx = ctx.index("chunk from session 0 (earliest)")
    middle_idx = ctx.index("chunk from session 1 (middle)")
    assert latest_idx < earliest_idx < middle_idx


@pytest.mark.asyncio
async def test_retrieve_context_no_ordering_for_question(monkeypatch):
    """question: vector results NOT reordered, no note line."""
    mod = _load_runner()
    monkeypatch.setenv("TAOSMD_LME_DATE_FRAME", "question")
    monkeypatch.setattr(mod, "ContextAssembler", _FakeAssembler)

    haystack_dates = [
        "2023/05/20 (Sat) 02:21",
        "2023/05/10 (Wed) 10:00",
        "2023/05/15 (Mon) 14:30",
    ]

    fake_results = [
        {"text": "chunk from session 2 (latest)", "metadata": {"session_date": haystack_dates[0]}, "similarity": 0.9},
        {"text": "chunk from session 0 (earliest)", "metadata": {"session_date": haystack_dates[1]}, "similarity": 0.8},
        {"text": "chunk from session 1 (middle)", "metadata": {"session_date": haystack_dates[2]}, "similarity": 0.7},
    ]
    monkeypatch.setattr(mod, "retrieve_vector_results", AsyncMock(return_value=fake_results))

    archive = _FakeArchive()
    kg = _FakeKG()

    ctx = await mod.retrieve_context(
        "test question", kg, archive, None,
        llm_client=None, graph_expansion=0, retrieval_path="retrieve",
        date_frame="question", haystack_dates=haystack_dates
    )

    assert "Memories below are in chronological order, oldest first." not in ctx
    latest_idx = ctx.index("chunk from session 2 (latest)")
    earliest_idx = ctx.index("chunk from session 0 (earliest)")
    middle_idx = ctx.index("chunk from session 1 (middle)")
    assert latest_idx < earliest_idx < middle_idx


# ---------------------------------------------------------------------------
# 5. labelled arm: rewrite prefixes at read time
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_labelled_rewrites_latest_to_most_recent(monkeypatch):
    """labelled: latest session_date gets 'most recent', others 'historical'."""
    mod = _load_runner()
    monkeypatch.setenv("TAOSMD_LME_DATE_FRAME", "labelled")
    monkeypatch.setattr(mod, "ContextAssembler", _FakeAssembler)

    haystack_dates = [
        "2023/05/10 (Wed) 10:00",  # earliest
        "2023/05/15 (Mon) 14:30",  # middle
        "2023/05/20 (Sat) 02:21",  # latest
    ]

    fake_results = [
        {"text": "[Session date: 2023/05/10 (Wed) 10:00]\nchunk A", "metadata": {"session_date": haystack_dates[0]}, "similarity": 0.9},
        {"text": "[Session date: 2023/05/15 (Mon) 14:30]\nchunk B", "metadata": {"session_date": haystack_dates[1]}, "similarity": 0.8},
        {"text": "[Session date: 2023/05/20 (Sat) 02:21]\nchunk C", "metadata": {"session_date": haystack_dates[2]}, "similarity": 0.7},
    ]
    monkeypatch.setattr(mod, "retrieve_vector_results", AsyncMock(return_value=fake_results))

    archive = _FakeArchive()
    kg = _FakeKG()

    ctx = await mod.retrieve_context(
        "test question", kg, archive, None,
        llm_client=None, graph_expansion=0, retrieval_path="retrieve",
        date_frame="labelled", haystack_dates=haystack_dates
    )

    assert "[Session date: 2023/05/20 (Sat) 02:21, most recent]" in ctx
    assert "[Session date: 2023/05/10 (Wed) 10:00, historical]" in ctx
    assert "[Session date: 2023/05/15 (Mon) 14:30, historical]" in ctx
    assert "Where memories disagree, the most recent one is current and the earlier ones are historical." in ctx


@pytest.mark.asyncio
async def test_labelled_two_chunks_same_latest_date_both_most_recent(monkeypatch):
    """labelled: two chunks sharing the latest date both get 'most recent'."""
    mod = _load_runner()
    monkeypatch.setenv("TAOSMD_LME_DATE_FRAME", "labelled")
    monkeypatch.setattr(mod, "ContextAssembler", _FakeAssembler)

    haystack_dates = [
        "2023/05/10 (Wed) 10:00",  # earliest
        "2023/05/20 (Sat) 02:21",  # latest
        "2023/05/20 (Sat) 02:21",  # also latest (same date)
    ]

    fake_results = [
        {"text": "[Session date: 2023/05/10 (Wed) 10:00]\nchunk A", "metadata": {"session_date": haystack_dates[0]}, "similarity": 0.9},
        {"text": "[Session date: 2023/05/20 (Sat) 02:21]\nchunk B", "metadata": {"session_date": haystack_dates[1]}, "similarity": 0.8},
        {"text": "[Session date: 2023/05/20 (Sat) 02:21]\nchunk C", "metadata": {"session_date": haystack_dates[2]}, "similarity": 0.7},
    ]
    monkeypatch.setattr(mod, "retrieve_vector_results", AsyncMock(return_value=fake_results))

    archive = _FakeArchive()
    kg = _FakeKG()

    ctx = await mod.retrieve_context(
        "test question", kg, archive, None,
        llm_client=None, graph_expansion=0, retrieval_path="retrieve",
        date_frame="labelled", haystack_dates=haystack_dates
    )

    assert ctx.count("[Session date: 2023/05/20 (Sat) 02:21, most recent]") == 2
    assert "[Session date: 2023/05/10 (Wed) 10:00, historical]" in ctx


@pytest.mark.asyncio
async def test_both_arm_no_rewrite(monkeypatch):
    """both: prefixes NOT rewritten, uses base note line."""
    mod = _load_runner()
    monkeypatch.setenv("TAOSMD_LME_DATE_FRAME", "both")
    monkeypatch.setattr(mod, "ContextAssembler", _FakeAssembler)

    haystack_dates = [
        "2023/05/10 (Wed) 10:00",
        "2023/05/20 (Sat) 02:21",
    ]

    fake_results = [
        {"text": "[Session date: 2023/05/10 (Wed) 10:00]\nchunk A", "metadata": {"session_date": haystack_dates[0]}, "similarity": 0.9},
        {"text": "[Session date: 2023/05/20 (Sat) 02:21]\nchunk B", "metadata": {"session_date": haystack_dates[1]}, "similarity": 0.8},
    ]
    monkeypatch.setattr(mod, "retrieve_vector_results", AsyncMock(return_value=fake_results))

    archive = _FakeArchive()
    kg = _FakeKG()

    ctx = await mod.retrieve_context(
        "test question", kg, archive, None,
        llm_client=None, graph_expansion=0, retrieval_path="retrieve",
        date_frame="both", haystack_dates=haystack_dates
    )

    assert "[Session date: 2023/05/10 (Wed) 10:00]\nchunk A" in ctx
    assert "[Session date: 2023/05/20 (Sat) 02:21]\nchunk B" in ctx
    assert "Memories below are in chronological order, oldest first." in ctx
    assert "Where memories disagree" not in ctx


# ---------------------------------------------------------------------------
# 6. Error handling: missing dates, mismatched lengths, unparseable dates
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_missing_question_date_exits_1(monkeypatch, tmp_path):
    """Missing question_date with non-off arm -> exit 1 with question_id and value."""
    mod = _load_runner()
    monkeypatch.setenv("TAOSMD_LME_DATE_FRAME", "question")
    monkeypatch.setattr(mod, "VectorMemory", lambda **kw: _FakeVectorMemory())
    monkeypatch.setattr(mod, "ArchiveStore", lambda **kw: _FakeArchive())
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", lambda **kw: _FakeKG())
    monkeypatch.setattr(mod, "ContextAssembler", _FakeAssembler)
    monkeypatch.setattr(mod, "process_conversation_turn", AsyncMock(return_value={"method": "regex"}))
    item = _sample_item()
    del item["question_date"]
    monkeypatch.setattr(mod, "load_dataset", lambda: [item])

    with pytest.raises(SystemExit) as exc:
        await mod.run_benchmark(args=_make_args(tmp_path=tmp_path))
    assert exc.value.code == 1


@pytest.mark.asyncio
async def test_mismatched_haystack_dates_length_exits_1(monkeypatch, tmp_path):
    """haystack_dates length != haystack_sessions length -> exit 1."""
    mod = _load_runner()
    monkeypatch.setenv("TAOSMD_LME_DATE_FRAME", "sessions")
    monkeypatch.setattr(mod, "VectorMemory", lambda **kw: _FakeVectorMemory())
    monkeypatch.setattr(mod, "ArchiveStore", lambda **kw: _FakeArchive())
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", lambda **kw: _FakeKG())
    monkeypatch.setattr(mod, "ContextAssembler", _FakeAssembler)
    monkeypatch.setattr(mod, "process_conversation_turn", AsyncMock(return_value={"method": "regex"}))
    item = _sample_item(haystack_dates=["2023/05/10 (Wed) 10:00"])
    monkeypatch.setattr(mod, "load_dataset", lambda: [item])

    with pytest.raises(SystemExit) as exc:
        await mod.run_benchmark(args=_make_args(tmp_path=tmp_path))
    assert exc.value.code == 1


@pytest.mark.asyncio
async def test_unparseable_date_exits_1(monkeypatch, tmp_path):
    """Unparseable date in haystack_dates -> exit 1."""
    mod = _load_runner()
    monkeypatch.setenv("TAOSMD_LME_DATE_FRAME", "sessions")
    monkeypatch.setattr(mod, "VectorMemory", lambda **kw: _FakeVectorMemory())
    monkeypatch.setattr(mod, "ArchiveStore", lambda **kw: _FakeArchive())
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", lambda **kw: _FakeKG())
    monkeypatch.setattr(mod, "ContextAssembler", _FakeAssembler)
    monkeypatch.setattr(mod, "process_conversation_turn", AsyncMock(return_value={"method": "regex"}))
    item = _sample_item(haystack_dates=["not a date", "2023/05/15 (Mon) 14:30", "2023/05/20 (Sat) 02:21"])
    monkeypatch.setattr(mod, "load_dataset", lambda: [item])

    with pytest.raises(SystemExit) as exc:
        await mod.run_benchmark(args=_make_args(tmp_path=tmp_path))
    assert exc.value.code == 1


@pytest.mark.asyncio
async def test_off_arm_ignores_missing_dates(monkeypatch, tmp_path):
    """off arm: missing/invalid dates do not cause exit."""
    mod = _load_runner()
    monkeypatch.setenv("TAOSMD_LME_DATE_FRAME", "off")
    monkeypatch.setattr(mod, "VectorMemory", lambda **kw: _FakeVectorMemory())
    monkeypatch.setattr(mod, "ArchiveStore", lambda **kw: _FakeArchive())
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", lambda **kw: _FakeKG())
    monkeypatch.setattr(mod, "ContextAssembler", _FakeAssembler)
    monkeypatch.setattr(mod, "process_conversation_turn", AsyncMock(return_value={"method": "regex"}))
    item = _sample_item()
    del item["question_date"]
    item["haystack_dates"] = ["not a date"] * 3
    monkeypatch.setattr(mod, "load_dataset", lambda: [item])

    await mod.run_benchmark(args=_make_args(tmp_path=tmp_path))


# ---------------------------------------------------------------------------
# 7. Results JSON includes date_frame in config block
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_results_json_includes_date_frame(monkeypatch, tmp_path):
    """Result JSON config block includes 'date_frame' next to retrieval_path and graph_expansion."""
    mod = _load_runner()
    monkeypatch.setenv("TAOSMD_LME_DATE_FRAME", "labelled")
    monkeypatch.setattr(mod, "VectorMemory", lambda **kw: _FakeVectorMemory())
    monkeypatch.setattr(mod, "ArchiveStore", lambda **kw: _FakeArchive())
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", lambda **kw: _FakeKG())
    monkeypatch.setattr(mod, "ContextAssembler", _FakeAssembler)
    monkeypatch.setattr(mod, "process_conversation_turn", AsyncMock(return_value={"method": "regex"}))
    monkeypatch.setattr(mod, "load_dataset", lambda: [_sample_item()])

    out_file = tmp_path / "results.json"
    await mod.run_benchmark(args=_make_args(tmp_path=tmp_path, out=str(out_file)))

    with open(out_file) as f:
        result = json.load(f)

    assert "date_frame" in result
    assert result["date_frame"] == "labelled"
    keys = list(result.keys())
    assert "retrieval_path" in keys
    assert "graph_expansion" in keys
    assert "date_frame" in keys


# ---------------------------------------------------------------------------
# 8. off arm produces byte-identical prompt to original ANSWER_PROMPT
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_off_arm_prompt_matches_original(monkeypatch, tmp_path):
    """off arm: captured llm_answer prompt equals master's ANSWER_PROMPT.format output for same inputs."""
    mod = _load_runner()
    monkeypatch.delenv("TAOSMD_LME_DATE_FRAME", raising=False)

    client = _CapturingHTTPClient()
    client.add_response("Alice moved to Paris in 2020")

    monkeypatch.setattr(mod, "VectorMemory", lambda **kw: _FakeVectorMemory())
    monkeypatch.setattr(mod, "ArchiveStore", lambda **kw: _FakeArchive())
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", lambda **kw: _FakeKG())
    monkeypatch.setattr(mod, "ContextAssembler", _FakeAssembler)
    monkeypatch.setattr(mod, "process_conversation_turn", AsyncMock(return_value={"method": "regex"}))
    monkeypatch.setattr(mod, "load_dataset", lambda: [_sample_item()])
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client)

    await mod.run_benchmark(args=_make_args(tmp_path=tmp_path, llm=True))

    payload = client.last_payload
    assert payload is not None
    content = payload["messages"][0]["content"]
    # With all three streams enabled and empty FTS/vector results, the context
    # is "ASSEMBLER_CONTEXT  " (two trailing spaces from the join).
    # Hard-coded expected from master's output for these exact inputs.
    expected = (
        "Based on the following context from past conversations, answer the question.\n"
        'If the answer is not in the context, say "I don\'t know."\n'
        "Answer concisely in 1-2 sentences. /no_think\n\n"
        "Context:\n"
        "ASSEMBLER_CONTEXT  \n\n"
        "Question: When did Alice move to Paris?\n\n"
        "Answer:"
    )
    assert content == expected


# ---------------------------------------------------------------------------
# 9. test_date_order_on_default_retrieve_path (parametrized over 'sessions' and 'labelled')
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("date_frame", ["sessions", "labelled"])
async def test_date_order_on_default_retrieve_path(date_frame, monkeypatch):
    """Build three hits via real _adapt_vector with dates out of order; pass through real retrieve_context; assert ascending date order, note line, labelled prefix rewrite."""
    mod = _load_runner()
    from taosmd.retrieval import _adapt_vector

    haystack_dates = [
        "2023/05/20 (Sat) 02:21",  # latest
        "2023/05/10 (Wed) 10:00",  # earliest
        "2023/05/15 (Mon) 14:30",  # middle
    ]

    raw_rows = [
        {"id": "1", "text": "[Session date: 2023/05/20 (Sat) 02:21]\nchunk latest", "similarity": 0.9, "metadata": {"session_date": haystack_dates[0]}, "created_at": "2023-05-20T02:21:00"},
        {"id": "2", "text": "[Session date: 2023/05/10 (Wed) 10:00]\nchunk earliest", "similarity": 0.8, "metadata": {"session_date": haystack_dates[1]}, "created_at": "2023-05-10T10:00:00"},
        {"id": "3", "text": "[Session date: 2023/05/15 (Mon) 14:30]\nchunk middle", "similarity": 0.7, "metadata": {"session_date": haystack_dates[2]}, "created_at": "2023-05-15T14:30:00"},
    ]

    hits = _adapt_vector(raw_rows)
    monkeypatch.setattr(mod, "retrieve_vector_results", AsyncMock(return_value=hits))
    monkeypatch.setattr(mod, "ContextAssembler", _FakeAssembler)

    archive = _FakeArchive()
    kg = _FakeKG()

    ctx = await mod.retrieve_context(
        "test question", kg, archive, None,
        llm_client=None, graph_expansion=0, retrieval_path="retrieve",
        date_frame=date_frame, haystack_dates=haystack_dates,
        streams=["vector"]  # Only vector stream
    )

    assert "Memories below are in chronological order, oldest first." in ctx

    earliest_idx = ctx.index("chunk earliest")
    middle_idx = ctx.index("chunk middle")
    latest_idx = ctx.index("chunk latest")
    assert earliest_idx < middle_idx < latest_idx

    if date_frame == "labelled":
        assert "[Session date: 2023/05/20 (Sat) 02:21, most recent]" in ctx
        assert "[Session date: 2023/05/10 (Wed) 10:00, historical]" in ctx
        assert "[Session date: 2023/05/15 (Mon) 14:30, historical]" in ctx
        assert "Where memories disagree, the most recent one is current and the earlier ones are historical." in ctx


# ---------------------------------------------------------------------------
# 10. Negative: hits with no session_date give context WITHOUT note line
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hits_without_session_date_no_note_line(monkeypatch):
    """Hits with no session_date at either level give context WITHOUT the note line."""
    mod = _load_runner()
    monkeypatch.setenv("TAOSMD_LME_DATE_FRAME", "sessions")
    monkeypatch.setattr(mod, "ContextAssembler", _FakeAssembler)

    haystack_dates = [
        "2023/05/20 (Sat) 02:21",
        "2023/05/10 (Wed) 10:00",
        "2023/05/15 (Mon) 14:30",
    ]

    # NO hits have session_date in metadata
    fake_results = [
        {"text": "chunk A no date", "metadata": {}, "similarity": 0.9},
        {"text": "chunk B no date", "metadata": {}, "similarity": 0.8},
        {"text": "chunk C no date", "metadata": {}, "similarity": 0.7},
    ]
    monkeypatch.setattr(mod, "retrieve_vector_results", AsyncMock(return_value=fake_results))

    archive = _FakeArchive()
    kg = _FakeKG()

    ctx = await mod.retrieve_context(
        "test question", kg, archive, None,
        llm_client=None, graph_expansion=0, retrieval_path="retrieve",
        date_frame="sessions", haystack_dates=haystack_dates,
        streams=["vector"]
    )

    assert "Memories below are in chronological order, oldest first." not in ctx


# ---------------------------------------------------------------------------
# 11. Streams interaction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_date_frame_sessions_with_assembler_fts_no_vector(monkeypatch):
    """date_frame='sessions' with streams=['assembler','fts'] gives context with no note line and no vector text."""
    mod = _load_runner()
    monkeypatch.setenv("TAOSMD_LME_DATE_FRAME", "sessions")
    monkeypatch.setattr(mod, "ContextAssembler", _FakeAssembler)

    haystack_dates = [
        "2023/05/20 (Sat) 02:21",
        "2023/05/10 (Wed) 10:00",
        "2023/05/15 (Mon) 14:30",
    ]

    fake_results = [
        {"text": "chunk A", "metadata": {"session_date": haystack_dates[0]}, "similarity": 0.9},
        {"text": "chunk B", "metadata": {"session_date": haystack_dates[1]}, "similarity": 0.8},
        {"text": "chunk C", "metadata": {"session_date": haystack_dates[2]}, "similarity": 0.7},
    ]
    monkeypatch.setattr(mod, "retrieve_vector_results", AsyncMock(return_value=fake_results))

    archive = _FakeArchive()
    kg = _FakeKG()

    ctx = await mod.retrieve_context(
        "test question", kg, archive, None,
        llm_client=None, graph_expansion=0, retrieval_path="retrieve",
        date_frame="sessions", haystack_dates=haystack_dates,
        streams=["assembler", "fts"]
    )

    assert "Memories below are in chronological order, oldest first." not in ctx
    assert "chunk A" not in ctx
    assert "chunk B" not in ctx
    assert "chunk C" not in ctx
    assert "ASSEMBLER_CONTEXT" in ctx


@pytest.mark.asyncio
async def test_date_frame_sessions_with_vector_only(monkeypatch):
    """date_frame='sessions' with streams=['vector'] gives note line plus ordered vector text only."""
    mod = _load_runner()
    monkeypatch.setenv("TAOSMD_LME_DATE_FRAME", "sessions")
    monkeypatch.setattr(mod, "ContextAssembler", _FakeAssembler)

    haystack_dates = [
        "2023/05/20 (Sat) 02:21",  # latest
        "2023/05/10 (Wed) 10:00",  # earliest
        "2023/05/15 (Mon) 14:30",  # middle
    ]

    fake_results = [
        {"text": "chunk latest", "metadata": {"session_date": haystack_dates[0]}, "similarity": 0.9},
        {"text": "chunk earliest", "metadata": {"session_date": haystack_dates[1]}, "similarity": 0.8},
        {"text": "chunk middle", "metadata": {"session_date": haystack_dates[2]}, "similarity": 0.7},
    ]
    monkeypatch.setattr(mod, "retrieve_vector_results", AsyncMock(return_value=fake_results))

    archive = _FakeArchive()
    kg = _FakeKG()

    ctx = await mod.retrieve_context(
        "test question", kg, archive, None,
        llm_client=None, graph_expansion=0, retrieval_path="retrieve",
        date_frame="sessions", haystack_dates=haystack_dates,
        streams=["vector"]
    )

    assert "Memories below are in chronological order, oldest first." in ctx
    earliest_idx = ctx.index("chunk earliest")
    middle_idx = ctx.index("chunk middle")
    latest_idx = ctx.index("chunk latest")
    assert earliest_idx < middle_idx < latest_idx
    assert "ASSEMBLER_CONTEXT" not in ctx


# ---------------------------------------------------------------------------
# 12. test_no_results_written_to_source_tree
# ---------------------------------------------------------------------------

def test_no_results_written_to_source_tree(monkeypatch, tmp_path):
    """Snapshot benchmarks/results before and after running one arm; assert sets are equal."""
    import os

    results_dir = REPO_ROOT / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    before = set(os.listdir(results_dir))

    mod = _load_runner()
    monkeypatch.setenv("TAOSMD_LME_DATE_FRAME", "sessions")
    monkeypatch.setattr(mod, "VectorMemory", lambda **kw: _FakeVectorMemory())
    monkeypatch.setattr(mod, "ArchiveStore", lambda **kw: _FakeArchive())
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", lambda **kw: _FakeKG())
    monkeypatch.setattr(mod, "ContextAssembler", _FakeAssembler)
    monkeypatch.setattr(mod, "process_conversation_turn", AsyncMock(return_value={"method": "regex"}))
    monkeypatch.setattr(mod, "load_dataset", lambda: [_sample_item()])

    out_file = tmp_path / "results.json"
    import asyncio
    asyncio.run(mod.run_benchmark(args=_make_args(out=str(out_file))))

    after = set(os.listdir(results_dir))
    assert before == after, f"Results directory changed: before={before}, after={after}"


# ---------------------------------------------------------------------------
# Mutation tests (M1, M2, M3) - these will fail initially on the broken code
# ---------------------------------------------------------------------------
#
# M1: make _hit_session_date read only the flat level
# M2: emit the note line unconditionally
# M3: make 'labelled' behave like 'both'
#
# These are documented here for the PR body; the actual mutations are applied
# manually during verification by editing benchmarks/longmemeval_runner.py
#
# RED-FIRST verification (run on origin/master before this fix):
# $ git stash
# $ python3 -m pytest tests/test_longmemeval_date_frame.py -v
# Expected failures on origin/master (without the fix):
#   - test_retrieve_context_chronological_order_with_note: note line not emitted, no reordering
#   - test_labelled_rewrites_latest_to_most_recent: prefixes not rewritten, wrong note line
#   - test_date_order_on_default_retrieve_path[sessions]: uses real _adapt_vector, fails without helper
#   - test_date_order_on_default_retrieve_path[labelled]: same
#   - test_hits_without_session_date_no_note_line: note line incorrectly emitted
#   - test_off_arm_prompt_matches_original: tautological comparison (passes but invalid)
#   - test_no_results_written_to_source_tree: results written to benchmarks/results/
#
# Mutation testing (M1/M2/M3 applied manually, tests must fail):
# M1: Edit _hit_session_date in taosmd/retrieval.py to only check flat metadata:
#     def _hit_session_date(hit): return hit.get("metadata", {}).get("session_date")
#     -> test_retrieve_context_chronological_order_with_note FAILS (nested session_date not read)
#     -> test_date_order_on_default_retrieve_path[sessions] FAILS
#     -> test_date_order_on_default_retrieve_path[labelled] FAILS
#     -> test_labelled_rewrites_latest_to_most_recent FAILS
#
# M2: Edit _apply_date_framing to emit note_line unconditionally:
#     Remove the `has_parseable_date` check, always prepend note_line
#     -> test_hits_without_session_date_no_note_line FAILS (note line emitted for undated hits)
#     -> test_retrieve_context_no_ordering_for_off FAILS
#     -> test_retrieve_context_no_ordering_for_question FAILS
#
# M3: Edit _apply_date_framing to make 'labelled' skip prefix rewrite:
#     Change `if date_frame == "labelled":` to `if False:`
#     -> test_labelled_rewrites_latest_to_most_recent FAILS (prefixes not rewritten)
#     -> test_labelled_two_chunks_same_latest_date_both_most_recent FAILS
#     -> test_date_order_on_default_retrieve_path[labelled] FAILS
#     -> test_both_arm_no_rewrite still PASSES (both arm unchanged)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])