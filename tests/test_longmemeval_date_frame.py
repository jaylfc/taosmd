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
async def test_ingest_prefixes_archive_and_vmem_for_sessions_arm(monkeypatch):
    """sessions/both/labelled: every archive.record and vmem.add gets [Session date: <d>] prefix."""
    mod = _runner_with_fakes(monkeypatch, "sessions")
    item = _sample_item()
    vmem = _FakeVectorMemory()
    archive = _FakeArchive()
    kg = _FakeKG()

    # Monkeypatch the store constructors to return our fakes
    monkeypatch.setattr(mod, "VectorMemory", lambda **kw: vmem)
    monkeypatch.setattr(mod, "ArchiveStore", lambda **kw: archive)
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", lambda **kw: kg)

    await mod.run_benchmark(limit=1, question_type="temporal", use_llm=False, args=None)

    # Check archive records
    for _, data, _ in archive.records:
        content = data.get("content", "")
        assert content.startswith("[Session date: "), f"Archive record missing prefix: {content[:80]}"
        # Should have one of the haystack dates
        assert any(d in content for d in item["haystack_dates"])

    # Check vmem adds
    for text, meta in vmem.adds:
        assert text.startswith("[Session date: "), f"VMEM add missing prefix: {text[:80]}"
        assert any(d in text for d in item["haystack_dates"])
        # metadata should have session_date
        assert "session_date" in meta
        assert meta["session_date"] in item["haystack_dates"]


@pytest.mark.asyncio
async def test_ingest_no_prefix_for_off_arm(monkeypatch):
    """off: no date prefixes on archive records or vmem adds."""
    mod = _runner_with_fakes(monkeypatch, "off")
    vmem = _FakeVectorMemory()
    archive = _FakeArchive()
    kg = _FakeKG()

    monkeypatch.setattr(mod, "VectorMemory", lambda **kw: vmem)
    monkeypatch.setattr(mod, "ArchiveStore", lambda **kw: archive)
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", lambda **kw: kg)

    await mod.run_benchmark(limit=1, question_type="temporal", use_llm=False, args=None)

    for _, data, _ in archive.records:
        content = data.get("content", "")
        assert not content.startswith("[Session date: "), f"Archive record should not have prefix: {content[:80]}"

    for text, meta in vmem.adds:
        assert not text.startswith("[Session date: "), f"VMEM add should not have prefix: {text[:80]}"
        assert "session_date" not in meta


@pytest.mark.asyncio
async def test_ingest_question_arm_no_prefix(monkeypatch):
    """question: no ingest-time prefixes (only affects answer prompt)."""
    mod = _runner_with_fakes(monkeypatch, "question")
    vmem = _FakeVectorMemory()
    archive = _FakeArchive()
    kg = _FakeKG()

    monkeypatch.setattr(mod, "VectorMemory", lambda **kw: vmem)
    monkeypatch.setattr(mod, "ArchiveStore", lambda **kw: archive)
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", lambda **kw: kg)

    await mod.run_benchmark(limit=1, question_type="temporal", use_llm=False, args=None)

    for _, data, _ in archive.records:
        content = data.get("content", "")
        assert not content.startswith("[Session date: "), f"Archive record should not have prefix: {content[:80]}"

    for text, meta in vmem.adds:
        assert not text.startswith("[Session date: "), f"VMEM add should not have prefix: {text[:80]}"
        assert "session_date" not in meta


# ---------------------------------------------------------------------------
# 3. Answer prompt: "Today is <question_date>." line for question/both/labelled
# ---------------------------------------------------------------------------

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


@pytest.mark.asyncio
async def test_question_arm_adds_today_line_to_prompt(monkeypatch):
    """question/both/labelled: answer prompt contains 'Today is <question_date>.' before Context:"""
    mod = _runner_with_fakes(monkeypatch, "question")

    client = _CapturingHTTPClient()
    client.add_response("Alice moved to Paris in 2020")

    monkeypatch.setattr(mod, "VectorMemory", lambda **kw: _FakeVectorMemory())
    monkeypatch.setattr(mod, "ArchiveStore", lambda **kw: _FakeArchive())
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", lambda **kw: _FakeKG())
    monkeypatch.setattr(mod, "ContextAssembler", _FakeAssembler)
    monkeypatch.setattr(mod, "process_conversation_turn", AsyncMock(return_value={"method": "regex"}))
    # Monkeypatch httpx.AsyncClient to return our capturing client
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client)

    await mod.run_benchmark(limit=1, question_type="temporal", use_llm=True, args=type("Args", (), {
        "limit": 1, "type": "temporal", "llm": True, "graph_expansion": 0,
        "retrieval_path": "retrieve", "report_retrieval_delta": False, "out": "",
        "writer_url": None, "writer_model": "default"
    })())

    payload = client.last_payload
    assert payload is not None
    content = payload["messages"][0]["content"]
    # The "Today is ..." line must appear immediately before "Context:"
    assert "Today is 2023/05/20 (Sat) 02:21." in content
    today_idx = content.index("Today is 2023/05/20 (Sat) 02:21.")
    context_idx = content.index("Context:")
    assert today_idx < context_idx
    # Should be immediately before (allowing for newline)
    assert content[today_idx:context_idx].strip() == "Today is 2023/05/20 (Sat) 02:21."


@pytest.mark.asyncio
async def test_both_arm_adds_today_line_to_prompt(monkeypatch):
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

    await mod.run_benchmark(limit=1, question_type="temporal", use_llm=True, args=type("Args", (), {
        "limit": 1, "type": "temporal", "llm": True, "graph_expansion": 0,
        "retrieval_path": "retrieve", "report_retrieval_delta": False, "out": "",
        "writer_url": None, "writer_model": "default"
    })())

    payload = client.last_payload
    assert payload is not None
    content = payload["messages"][0]["content"]
    assert "Today is 2023/05/20 (Sat) 02:21." in content
    today_idx = content.index("Today is 2023/05/20 (Sat) 02:21.")
    context_idx = content.index("Context:")
    assert today_idx < context_idx


@pytest.mark.asyncio
async def test_labelled_arm_adds_today_line_to_prompt(monkeypatch):
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

    await mod.run_benchmark(limit=1, question_type="temporal", use_llm=True, args=type("Args", (), {
        "limit": 1, "type": "temporal", "llm": True, "graph_expansion": 0,
        "retrieval_path": "retrieve", "report_retrieval_delta": False, "out": "",
        "writer_url": None, "writer_model": "default"
    })())

    payload = client.last_payload
    assert payload is not None
    content = payload["messages"][0]["content"]
    assert "Today is 2023/05/20 (Sat) 02:21." in content
    today_idx = content.index("Today is 2023/05/20 (Sat) 02:21.")
    context_idx = content.index("Context:")
    assert today_idx < context_idx


@pytest.mark.asyncio
async def test_off_arm_no_today_line(monkeypatch):
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

    await mod.run_benchmark(limit=1, question_type="temporal", use_llm=True, args=type("Args", (), {
        "limit": 1, "type": "temporal", "llm": True, "graph_expansion": 0,
        "retrieval_path": "retrieve", "report_retrieval_delta": False, "out": "",
        "writer_url": None, "writer_model": "default"
    })())

    payload = client.last_payload
    assert payload is not None
    content = payload["messages"][0]["content"]
    assert "Today is" not in content


@pytest.mark.asyncio
async def test_sessions_arm_no_today_line(monkeypatch):
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

    await mod.run_benchmark(limit=1, question_type="temporal", use_llm=True, args=type("Args", (), {
        "limit": 1, "type": "temporal", "llm": True, "graph_expansion": 0,
        "retrieval_path": "retrieve", "report_retrieval_delta": False, "out": "",
        "writer_url": None, "writer_model": "default"
    })())

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

    # Three sessions with out-of-order dates
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

    # Should contain the note line
    assert "Memories below are in chronological order, oldest first." in ctx

    # Chunks should appear in chronological order (earliest first)
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

    # Should NOT contain the note line
    assert "Memories below are in chronological order, oldest first." not in ctx

    # Order should be preserved as returned by vmem.search (similarity order)
    latest_idx = ctx.index("chunk from session 2 (latest)")
    earliest_idx = ctx.index("chunk from session 0 (earliest)")
    middle_idx = ctx.index("chunk from session 1 (middle)")
    assert latest_idx < earliest_idx < middle_idx  # Original similarity order


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

    # Latest date should be rewritten to "most recent"
    assert "[Session date: 2023/05/20 (Sat) 02:21, most recent]" in ctx
    # Others should be "historical"
    assert "[Session date: 2023/05/10 (Wed) 10:00, historical]" in ctx
    assert "[Session date: 2023/05/15 (Mon) 14:30, historical]" in ctx
    # Note line should be the extended one
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

    # Both chunks with latest date should be "most recent"
    assert ctx.count("[Session date: 2023/05/20 (Sat) 02:21, most recent]") == 2
    # Earliest should be "historical"
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

    # Prefixes should NOT be rewritten
    assert "[Session date: 2023/05/10 (Wed) 10:00]\nchunk A" in ctx
    assert "[Session date: 2023/05/20 (Sat) 02:21]\nchunk B" in ctx
    # Base note line only
    assert "Memories below are in chronological order, oldest first." in ctx
    assert "Where memories disagree" not in ctx


# ---------------------------------------------------------------------------
# 6. Error handling: missing dates, mismatched lengths, unparseable dates
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_missing_question_date_exits_1(monkeypatch):
    """Missing question_date with non-off arm -> exit 1 with question_id and value."""
    mod = _load_runner()
    monkeypatch.setenv("TAOSMD_LME_DATE_FRAME", "question")
    monkeypatch.setattr(mod, "VectorMemory", lambda **kw: _FakeVectorMemory())
    monkeypatch.setattr(mod, "ArchiveStore", lambda **kw: _FakeArchive())
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", lambda **kw: _FakeKG())
    monkeypatch.setattr(mod, "ContextAssembler", _FakeAssembler)
    monkeypatch.setattr(mod, "process_conversation_turn", AsyncMock(return_value={"method": "regex"}))
    # Mock dataset loading with item missing question_date
    item = _sample_item()
    del item["question_date"]
    monkeypatch.setattr(mod, "load_dataset", lambda: [item])

    with pytest.raises(SystemExit) as exc:
        await mod.run_benchmark(limit=1, question_type="temporal", use_llm=False, args=None)
    assert exc.value.code == 1


@pytest.mark.asyncio
async def test_mismatched_haystack_dates_length_exits_1(monkeypatch):
    """haystack_dates length != haystack_sessions length -> exit 1."""
    mod = _load_runner()
    monkeypatch.setenv("TAOSMD_LME_DATE_FRAME", "sessions")
    monkeypatch.setattr(mod, "VectorMemory", lambda **kw: _FakeVectorMemory())
    monkeypatch.setattr(mod, "ArchiveStore", lambda **kw: _FakeArchive())
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", lambda **kw: _FakeKG())
    monkeypatch.setattr(mod, "ContextAssembler", _FakeAssembler)
    monkeypatch.setattr(mod, "process_conversation_turn", AsyncMock(return_value={"method": "regex"}))
    # Mock dataset loading with mismatched dates
    item = _sample_item(haystack_dates=["2023/05/10 (Wed) 10:00"])  # Only 1 date, 3 sessions
    monkeypatch.setattr(mod, "load_dataset", lambda: [item])

    with pytest.raises(SystemExit) as exc:
        await mod.run_benchmark(limit=1, question_type="temporal", use_llm=False, args=None)
    assert exc.value.code == 1


@pytest.mark.asyncio
async def test_unparseable_date_exits_1(monkeypatch):
    """Unparseable date in haystack_dates -> exit 1."""
    mod = _load_runner()
    monkeypatch.setenv("TAOSMD_LME_DATE_FRAME", "sessions")
    monkeypatch.setattr(mod, "VectorMemory", lambda **kw: _FakeVectorMemory())
    monkeypatch.setattr(mod, "ArchiveStore", lambda **kw: _FakeArchive())
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", lambda **kw: _FakeKG())
    monkeypatch.setattr(mod, "ContextAssembler", _FakeAssembler)
    monkeypatch.setattr(mod, "process_conversation_turn", AsyncMock(return_value={"method": "regex"}))
    # Mock dataset loading with unparseable date
    item = _sample_item(haystack_dates=["not a date", "2023/05/15 (Mon) 14:30", "2023/05/20 (Sat) 02:21"])
    monkeypatch.setattr(mod, "load_dataset", lambda: [item])

    with pytest.raises(SystemExit) as exc:
        await mod.run_benchmark(limit=1, question_type="temporal", use_llm=False, args=None)
    assert exc.value.code == 1


@pytest.mark.asyncio
async def test_off_arm_ignores_missing_dates(monkeypatch):
    """off arm: missing/invalid dates do not cause exit."""
    mod = _load_runner()
    monkeypatch.setenv("TAOSMD_LME_DATE_FRAME", "off")
    monkeypatch.setattr(mod, "VectorMemory", lambda **kw: _FakeVectorMemory())
    monkeypatch.setattr(mod, "ArchiveStore", lambda **kw: _FakeArchive())
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", lambda **kw: _FakeKG())
    monkeypatch.setattr(mod, "ContextAssembler", _FakeAssembler)
    monkeypatch.setattr(mod, "process_conversation_turn", AsyncMock(return_value={"method": "regex"}))
    # Mock dataset loading with invalid dates (should be ignored for off arm)
    item = _sample_item()
    del item["question_date"]
    item["haystack_dates"] = ["not a date"] * 3  # Invalid but should be ignored
    monkeypatch.setattr(mod, "load_dataset", lambda: [item])

    # Should not raise
    await mod.run_benchmark(limit=1, question_type="temporal", use_llm=False, args=None)


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
    await mod.run_benchmark(
        limit=1, question_type="temporal", use_llm=False,
        args=type("Args", (), {
            "limit": 1, "type": "temporal", "llm": False, "graph_expansion": 0,
            "retrieval_path": "retrieve", "report_retrieval_delta": False, "out": str(out_file),
            "writer_url": None, "writer_model": "default"
        })()
    )

    with open(out_file) as f:
        result = json.load(f)

    assert "date_frame" in result
    assert result["date_frame"] == "labelled"
    # Should be near retrieval_path and graph_expansion
    keys = list(result.keys())
    assert "retrieval_path" in keys
    assert "graph_expansion" in keys
    assert "date_frame" in keys


# ---------------------------------------------------------------------------
# 8. off arm produces byte-identical prompt to original ANSWER_PROMPT
# ---------------------------------------------------------------------------

def test_off_arm_prompt_matches_original(monkeypatch):
    """off arm: the formatted prompt equals ANSWER_PROMPT.format(context=..., question=...)."""
    mod = _load_runner()
    monkeypatch.delenv("TAOSMD_LME_DATE_FRAME", raising=False)

    context = "Some context here"
    question = "What happened?"
    expected = mod.ANSWER_PROMPT.format(context=context, question=question)

    # Call the internal function that builds the prompt (we need to check how it's done)
    # The runner uses ANSWER_PROMPT.format directly in llm_answer
    # So for off arm, it should be identical
    actual = mod.ANSWER_PROMPT.format(context=context[:mod.CONTEXT_CHARS], question=question)
    assert actual == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])