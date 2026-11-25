"""Tests for taosmd.answer_context: the dated, budgeted, oldest-first context.

Offline: the vector embedder is replaced by a deterministic hash vector (same
as tests/test_api.py). Where a test needs a known ranking it fixes the hits
with ``_ranked`` (real hits taken from the real store, in a chosen order), so
the expected strings below are exact.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import threading
from pathlib import Path
import urllib.error
import urllib.request

import pytest

import taosmd
from taosmd import api as taosmd_api
from taosmd import http_server

ac = importlib.import_module("taosmd.answer_context")

NOW = "2023/06/10 (Sat) 10:00"
GUIDANCE = (
    "Answer from the dated context. You may combine facts and calculate from the "
    "dates (differences, order, how long ago relative to the current date). Say "
    "you don't know only if a fact you need is missing."
)

# 3 sessions on 3 dates, 4 turns each.
SESSIONS = {
    "s1": ("2023/05/20 (Sat) 02:21", [
        ("user", "I adopted a dog named Rex."),
        ("assistant", "Congrats! What breed is Rex?"),
        ("user", "A beagle, he is two years old."),
        ("assistant", "Beagles love long walks."),
    ]),
    "s2": ("2023/05/22 (Mon) 09:00", [
        ("user", "Booked flights to Paris for August."),
        ("assistant", "Paris in August is warm."),
        ("user", "I will stay near the Louvre."),
        ("assistant", "Great area for museums."),
    ]),
    "s3": ("2023/06/01 (Thu) 18:30", [
        ("user", "Rex learned to sit today."),
        ("assistant", "Training a beagle takes patience."),
        ("user", "Next we try roll over."),
        ("assistant", "Good luck with that."),
    ]),
}


def _patch_embedder(stores: dict) -> None:
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "taosmd-data"
    d.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    orig = taosmd_api._ensure_stores

    async def _ensure(data_dir=None):
        stores = await orig(data_dir)
        _patch_embedder(stores)
        return stores

    monkeypatch.setattr(taosmd_api, "_ensure_stores", _ensure)
    yield str(d)
    for stores in list(taosmd_api._stores_cache.values()):
        for store in (stores.get("archive"), stores.get("vector"), stores.get("kg")):
            if store and hasattr(store, "close"):
                try:
                    asyncio.run(store.close())
                except Exception:
                    pass


def _ingest_sessions(data_dir, agent="a", project=None, sessions=SESSIONS):
    for sid, (date, turns) in sessions.items():
        rows = [
            {"role": role, "content": text, "session_id": sid, "turn_index": i,
             "session_date": date}
            for i, (role, text) in enumerate(turns)
        ]
        asyncio.run(taosmd.ingest(rows, agent=agent, project=project, data_dir=data_dir))


def _row(data_dir, text, agent="a"):
    """The stored hit whose text is exactly ``text`` (ranking-independent)."""
    hits = asyncio.run(taosmd_api.search(text, agent=agent, data_dir=data_dir, limit=50))
    exact = [h for h in hits if h["text"] == text]
    assert exact, f"no stored row with text {text!r}"
    return exact[0]


def _ranked(monkeypatch, data_dir, texts, agent="a"):
    """Make search return the real hit of each text, in the order given."""
    async def one(t):
        # Pick the exact row from a wide result list: ranking depends on the
        # embedder (CI falls back to a keyword-only one), the text does not.
        hits = await taosmd_api.search(t, agent=agent, data_dir=data_dir, limit=50)
        exact = [h for h in hits if h["text"] == t]
        assert exact, f"no stored row with text {t!r}"
        return exact[0]

    fixed = [asyncio.run(one(t)) for t in texts]

    async def fake_search(query, **kw):
        return list(fixed)

    monkeypatch.setattr(ac._api, "search", fake_search)


def _ctx(data_dir, **kw):
    kw.setdefault("budget_tokens", 100_000)
    kw.setdefault("now", NOW)
    kw.setdefault("neighbors", 1)
    kw.setdefault("grow", False)   # window tests; the whole-session pass has its own tests
    return asyncio.run(taosmd.answer_context("q", agent=kw.pop("agent", "a"), data_dir=data_dir, **kw))


def _block(sid, lo, hi):
    date, turns = SESSIONS[sid]
    d = date[:10].replace("/", "-") + " (" + date[12:15] + ")"
    return "\n".join([f"Session {d}"] + [f"{r}: {t}" for r, t in turns[lo:hi]])


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------

def test_golden_output(monkeypatch, data_dir):
    """Windows only (grow off). Hits are ranked newest-session first; the context is oldest first, each hit
    widened by one turn either side, dated headers, one blank line between blocks."""
    _ingest_sessions(data_dir)
    _ranked(monkeypatch, data_dir, [
        SESSIONS["s3"][1][1][1],   # s3 turn 1
        SESSIONS["s1"][1][1][1],   # s1 turn 1
        SESSIONS["s2"][1][2][1],   # s2 turn 2
    ])
    out = _ctx(data_dir)
    expected = (
        "Current date: 2023-06-10 (Sat)\n\n"
        + _block("s1", 0, 3) + "\n\n"
        + _block("s2", 1, 4) + "\n\n"
        + _block("s3", 0, 3)
    )
    assert out["context"] == expected
    assert out["guidance"] == GUIDANCE
    assert out["sessions_included"] == 3
    assert out["sessions_dropped"] == 0
    assert out["hits_total"] == 3
    assert out["expanded"] is True
    assert out["tokens_used"] == ac.estimate_tokens(expected)
    assert out["tokens_used"] <= out["budget_available"]
    assert set(out) == {"context", "guidance", "sessions_included", "sessions_dropped",
                        "hits_total", "tokens_used", "budget_available", "expanded",
                        "included", "dropped"}
    assert [b["session_id"] for b in out["included"]] == ["s1", "s2", "s3"]
    assert [b["date"] for b in out["included"]] == [
        "2023-05-20 (Sat)", "2023-05-22 (Mon)", "2023-06-01 (Thu)"]
    assert [t["turn_index"] for t in out["included"][0]["turns"]] == [0, 1, 2]
    assert [b["span"] for b in out["included"]] == ["window"] * 3
    assert all(b["tokens"] > 0 for b in out["included"])
    assert all(t["vector_id"] is not None and t["archive_id"] is not None
               for b in out["included"] for t in b["turns"])
    assert out["dropped"] == []


def test_neighbors_zero_gives_hit_alone_in_its_dated_block(monkeypatch, data_dir):
    _ingest_sessions(data_dir)
    _ranked(monkeypatch, data_dir, [SESSIONS["s1"][1][1][1]])
    out = _ctx(data_dir, neighbors=0)
    assert out["context"] == (
        "Current date: 2023-06-10 (Sat)\n\nSession 2023-05-20 (Sat)\n"
        "assistant: Congrats! What breed is Rex?"
    )
    assert out["expanded"] is False


def test_two_hits_of_one_session_share_one_block(monkeypatch, data_dir):
    _ingest_sessions(data_dir)
    _ranked(monkeypatch, data_dir, [SESSIONS["s1"][1][0][1], SESSIONS["s1"][1][3][1]])
    out = _ctx(data_dir, neighbors=1)
    assert out["sessions_included"] == 1
    assert out["context"].endswith(_block("s1", 0, 4))


def test_real_search_path_end_to_end(data_dir):
    """No patched ranking: the real search finds the turn and widens it."""
    _ingest_sessions(data_dir)
    out = asyncio.run(taosmd.answer_context(
        SESSIONS["s2"][1][1][1], agent="a", data_dir=data_dir, budget_tokens=100_000,
        now=NOW, neighbors=1, limit=1))
    # plenty of budget: the hit's session is grown to the whole session
    assert out["context"] == "Current date: 2023-06-10 (Sat)\n\n" + _block("s2", 0, 4)


def test_unknown_session_date_header(monkeypatch, data_dir):
    async def fake_search(query, **kw):
        return [{"text": "no date here", "source": "vector", "timestamp": 0,
                 "confidence": 0.9, "metadata": {"role": "user"}}]

    monkeypatch.setattr(ac._api, "search", fake_search)
    out = _ctx(data_dir)
    assert out["context"] == "Current date: 2023-06-10 (Sat)\n\nSession (date unknown)\nuser: no date here"


def test_undated_session_sorts_before_dated(monkeypatch, data_dir):
    _ingest_sessions(data_dir)
    hit = _row(data_dir, SESSIONS["s1"][1][0][1])

    async def fake_search(query, **kw):
        return [hit, {"text": "floating", "source": "vector", "timestamp": 0,
                      "confidence": 0.5, "metadata": {}}]

    monkeypatch.setattr(ac._api, "search", fake_search)
    out = _ctx(data_dir, neighbors=0)
    assert out["context"].index("Session (date unknown)") < out["context"].index("Session 2023-05-20")


def test_facts_excluded_by_default_and_appended_on_request(monkeypatch, data_dir):
    _ingest_sessions(data_dir)
    turn = _row(data_dir, SESSIONS["s1"][1][0][1])
    fact = {"text": "Rex is_a dog", "source": "kg", "timestamp": 0, "confidence": 1.0, "metadata": {}}

    async def fake_search(query, **kw):
        return [fact, turn]

    monkeypatch.setattr(ac._api, "search", fake_search)
    default = _ctx(data_dir, neighbors=0)
    assert "Rex is_a dog" not in default["context"]
    assert default["hits_total"] == 2
    with_facts = _ctx(data_dir, neighbors=0, include_facts=True)
    assert with_facts["context"].endswith("\n\nKnown facts\nRex is_a dog")


def test_rows_without_session_fall_back_to_the_hit_alone(data_dir):
    """Data written before sessions were recorded: no neighbours, expanded False."""
    asyncio.run(taosmd.ingest(
        [{"role": "user", "content": "old row one"}, {"role": "user", "content": "old row two"}],
        agent="a", data_dir=data_dir))
    out = asyncio.run(taosmd.answer_context(
        "old row one", agent="a", data_dir=data_dir, budget_tokens=100_000, now=NOW, limit=1))
    assert "old row one" in out["context"]
    assert "old row two" not in out["context"]
    assert out["expanded"] is False


def test_empty_store(data_dir):
    out = asyncio.run(taosmd.answer_context(
        "anything", agent="a", data_dir=data_dir, budget_tokens=10_000, now=NOW))
    assert out["context"] == ""
    assert out["hits_total"] == 0
    assert out["sessions_included"] == 0
    assert out["tokens_used"] == 0
    assert out["expanded"] is False


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------

def _chars(s: str) -> int:
    return len(s)


def _budget_for(texts_blocks, extra=0):
    """Budget (tokens = characters) that fits exactly the given blocks."""
    ctx = "Current date: 2023-06-10 (Sat)\n\n" + "\n\n".join(texts_blocks)
    return len(ctx) + extra


def test_budget_fitting_two_of_three_drops_the_third_whole(monkeypatch, data_dir):
    _ingest_sessions(data_dir)
    _ranked(monkeypatch, data_dir, [
        SESSIONS["s1"][1][1][1], SESSIONS["s2"][1][1][1], SESSIONS["s3"][1][1][1]])
    b1, b2, b3 = _block("s1", 0, 3), _block("s2", 0, 3), _block("s3", 0, 3)
    out = _ctx(data_dir, tokenizer=_chars, budget_tokens=_budget_for([b1, b2]))
    assert out["sessions_included"] == 2
    assert out["sessions_dropped"] == 1
    assert out["context"] == "Current date: 2023-06-10 (Sat)\n\n" + b1 + "\n\n" + b2
    # dropped whole: not one line of it is present
    for line in b3.splitlines():
        assert line not in out["context"]
    assert out["tokens_used"] <= out["budget_available"]


def test_dropped_ids_are_reported_and_never_in_the_context(monkeypatch, data_dir):
    _ingest_sessions(data_dir)
    _ranked(monkeypatch, data_dir, [
        SESSIONS["s1"][1][1][1], SESSIONS["s2"][1][1][1], SESSIONS["s3"][1][1][1]])
    b1, b2 = _block("s1", 0, 3), _block("s2", 0, 3)
    out = _ctx(data_dir, tokenizer=_chars, budget_tokens=_budget_for([b1, b2]))
    assert [b["session_id"] for b in out["included"]] == ["s1", "s2"]
    assert [b["session_id"] for b in out["dropped"]] == ["s3"]
    dropped_turns = out["dropped"][0]["turns"]
    assert [t["turn_index"] for t in dropped_turns] == [0, 1, 2]
    included_ids = {t["vector_id"] for b in out["included"] for t in b["turns"]}
    assert all(t["vector_id"] not in included_ids for t in dropped_turns)
    # the dropped block's text is not in the context; the included ids are
    for _, text in SESSIONS["s3"][1][:3]:
        assert text not in out["context"]
    for sid in ("s1", "s2"):
        for _, text in SESSIONS[sid][1][:3]:
            assert text in out["context"]
    assert out["sessions_dropped"] == len(out["dropped"])
    assert out["sessions_included"] == len(out["included"])


def test_block_that_does_not_fit_is_skipped_and_the_next_still_tried(monkeypatch, data_dir):
    big = {"big": ("2023/05/01 (Mon) 10:00", [("user", "x" * 5000)])}
    _ingest_sessions(data_dir, sessions={**big, "s1": SESSIONS["s1"]})
    _ranked(monkeypatch, data_dir, ["x" * 5000, SESSIONS["s1"][1][1][1]])
    b1 = _block("s1", 0, 3)
    out = _ctx(data_dir, tokenizer=_chars, budget_tokens=_budget_for([b1], 10))
    assert out["sessions_included"] == 1
    assert out["sessions_dropped"] == 1
    assert "xxxx" not in out["context"]
    assert b1 in out["context"]


def test_block_larger_than_whole_budget_is_dropped_never_cut(monkeypatch, data_dir):
    _ingest_sessions(data_dir)
    _ranked(monkeypatch, data_dir, [SESSIONS["s1"][1][1][1]])
    out = _ctx(data_dir, tokenizer=_chars, budget_tokens=60)
    assert out["context"] == ""
    assert out["sessions_included"] == 0
    assert out["sessions_dropped"] == 1
    assert out["tokens_used"] == 0


def test_tokens_used_never_exceeds_available_at_any_budget(monkeypatch, data_dir):
    _ingest_sessions(data_dir)
    _ranked(monkeypatch, data_dir, [
        SESSIONS["s1"][1][1][1], SESSIONS["s2"][1][1][1], SESSIONS["s3"][1][1][1]])
    seen = set()
    for budget in range(0, 1200, 7):
        out = _ctx(data_dir, budget_tokens=budget)
        assert out["tokens_used"] <= out["budget_available"]
        assert out["sessions_included"] + out["sessions_dropped"] == 3
        seen.add(out["sessions_included"])
    assert seen == {0, 1, 2, 3}


def test_budget_tokens_is_the_evidence_budget_exactly(monkeypatch, data_dir):
    """Nothing is subtracted inside answer_context: the context may use exactly
    budget_tokens, one token less drops the block, and budget_available echoes it."""
    _ingest_sessions(data_dir)
    _ranked(monkeypatch, data_dir, [SESSIONS["s1"][1][1][1]])
    fits = _budget_for([_block("s1", 0, 3)])
    out = _ctx(data_dir, tokenizer=_chars, budget_tokens=fits)
    assert out["sessions_included"] == 1
    assert out["tokens_used"] == fits == out["budget_available"]
    out = _ctx(data_dir, tokenizer=_chars, budget_tokens=fits - 1)
    assert out["sessions_included"] == 0 and out["budget_available"] == fits - 1


def test_the_reserves_are_not_answer_context_parameters(data_dir):
    with pytest.raises(TypeError):
        asyncio.run(taosmd.answer_context("q", agent="a", data_dir=data_dir, budget_tokens=10,
                                          answer_reserve=512))
    with pytest.raises(TypeError):
        asyncio.run(taosmd.answer_context("q", agent="a", data_dir=data_dir, budget_tokens=10,
                                          reasoning_reserve=1))


def test_default_estimate_is_chars_over_3_5_rounded_up():
    assert ac.estimate_tokens("") == 0
    assert ac.estimate_tokens("a") == 1
    assert ac.estimate_tokens("a" * 7) == 2
    assert ac.estimate_tokens("a" * 8) == 3


def test_default_budget_tokens_arithmetic():
    assert ac.default_budget_tokens(10_000) == 4000
    assert ac.default_budget_tokens(10_000, used_tokens=7_000) == 2488
    assert ac.default_budget_tokens(10_000, used_tokens=9_488) == 0
    assert ac.default_budget_tokens(10_000, used_tokens=9_900) == 0
    assert ac.default_budget_tokens(10_000, used_tokens=50_000) == 0
    # a typical agent turn: ~22k already used in a 32k window
    assert ac.default_budget_tokens(32768, used_tokens=22000) == 10256
    assert ac.default_budget_tokens(32768) == 13107
    assert ac.default_budget_tokens(32768, used_tokens=22000, reasoning_reserve=3584) == 6672
    assert ac.default_budget_tokens(65536, used_tokens=22000) == 26214
    assert ac.default_budget_tokens(10_000, used_tokens=9_000, reasoning_reserve=600) == 0
    assert ac.default_budget_tokens(0) == 0
    assert ac.default_budget_tokens(10_000, share=0.5, answer_reserve=0) == 5000
    assert taosmd.default_budget_tokens is ac.default_budget_tokens


def test_used_tokens_near_the_window_gives_empty_context_with_drops_counted(monkeypatch, data_dir):
    _ingest_sessions(data_dir)
    _ranked(monkeypatch, data_dir, [SESSIONS["s1"][1][1][1], SESSIONS["s2"][1][1][1]])
    budget = ac.default_budget_tokens(8_000, used_tokens=7_900)
    assert budget == 0
    out = _ctx(data_dir, budget_tokens=budget)
    assert out["context"] == ""
    assert out["sessions_included"] == 0
    assert out["sessions_dropped"] == 2
    assert out["budget_available"] == 0


# ---------------------------------------------------------------------------
# scoping and access: the same as search
# ---------------------------------------------------------------------------

def test_agent_a_context_never_holds_agent_b_turns(data_dir):
    both = {"s1": SESSIONS["s1"]}
    _ingest_sessions(data_dir, agent="a", sessions=both)
    secret = {"s1": ("2023/05/21 (Sun) 08:00", [
        ("user", "Rex learned a secret handshake."), ("assistant", "B-private reply."),
        ("user", "B-private follow up."), ("assistant", "B-private end.")])}
    _ingest_sessions(data_dir, agent="b", sessions=secret)   # same session id on purpose
    out = asyncio.run(taosmd.answer_context(
        "Rex", agent="a", data_dir=data_dir, budget_tokens=100_000, now=NOW, limit=10, neighbors=3))
    assert "B-private" not in out["context"]
    assert "secret handshake" not in out["context"]
    assert "Rex" in out["context"]


def test_project_scope_limits_neighbours_too(data_dir):
    """Same agent, same session id reused in two projects: a search scoped to
    one project must not pull the other project's turns in as neighbours."""
    p1 = {"s1": ("2023/05/20 (Sat) 02:21", [("user", "p1 first turn about Rex"),
                                            ("assistant", "p1 second turn")])}
    asyncio.run(taosmd.ingest(
        [{"role": r, "content": t, "session_id": "s1", "turn_index": i, "session_date": p1["s1"][0]}
         for i, (r, t) in enumerate(p1["s1"][1])], agent="a", project="p1", data_dir=data_dir))
    asyncio.run(taosmd.ingest(
        [{"role": "user", "content": "p2 LEAK turn", "session_id": "s1", "turn_index": 2,
          "session_date": p1["s1"][0]},
         {"role": "user", "content": "p2 LEAK turn two", "session_id": "s1", "turn_index": 3,
          "session_date": p1["s1"][0]}], agent="a", project="p2", data_dir=data_dir))
    out = asyncio.run(taosmd.answer_context(
        "p1 first turn about Rex", agent="a", project="p1", data_dir=data_dir,
        budget_tokens=100_000, now=NOW, limit=1, neighbors=3))
    assert "p1 second turn" in out["context"]
    assert "LEAK" not in out["context"]


def test_search_scope_arguments_reach_search(monkeypatch, data_dir):
    """project, also_include, collections, collections_only and mode are handed to
    search unchanged: dropping any one of them changes what the caller can see."""
    seen = {}

    async def fake_search(query, **kw):
        seen.update(kw)
        return []

    monkeypatch.setattr(ac._api, "search", fake_search)
    _ctx(data_dir, project="p", also_include=["b"], collections=["c1"], collections_only=True,
         mode="bm25", limit=9)
    assert seen["agent"] == "a"
    assert seen["project"] == "p"
    assert seen["also_include"] == ["b"]
    assert seen["collections"] == ["c1"]
    assert seen["collections_only"] is True
    assert seen["mode"] == "bm25"
    assert seen["limit"] == 9
    assert seen["data_dir"] == data_dir


def test_cross_agent_read_in_project_matches_search(data_dir):
    _ingest_sessions(data_dir, agent="a", project="p", sessions={"s1": SESSIONS["s1"]})
    _ingest_sessions(data_dir, agent="b", project="p", sessions={"s2": SESSIONS["s2"]})
    solo = asyncio.run(taosmd.answer_context(
        "Paris", agent="a", project="p", data_dir=data_dir, budget_tokens=100_000, now=NOW, limit=10))
    shared = asyncio.run(taosmd.answer_context(
        "Paris", agent="a", project="p", also_include=["b"], data_dir=data_dir,
        budget_tokens=100_000, now=NOW, limit=10))
    assert "Paris" not in solo["context"]
    assert "Paris" in shared["context"]


def _collection_env(tmp_path, data_dir, grant_to):
    from taosmd import config as taosmd_config
    from taosmd.collections import CollectionStore, ingest_folder

    source = tmp_path / "src"
    source.mkdir()
    (source / "readme.md").write_text("# Widget\n\nThe widget frobnicates the sprocket.")
    taosmd_config.set_collections_allowed_roots([str(source)], data_dir=data_dir)
    store = CollectionStore(data_dir)
    col = store.create(name="docs", kind="docs", source_path=str(source))
    if grant_to:
        store.grant(col["id"], grant_to)
    asyncio.run(ingest_folder(col["id"], data_dir=data_dir))
    return col["id"]


def test_ungranted_collection_is_absent_and_granted_is_present(tmp_path, data_dir, monkeypatch):
    monkeypatch.delenv("TAOSMD_COLLECTIONS_ALLOWED_ROOTS", raising=False)
    cid = _collection_env(tmp_path, data_dir, grant_to="dev")
    ask = dict(data_dir=data_dir, budget_tokens=100_000, now=NOW, limit=5)
    granted = asyncio.run(taosmd.answer_context("widget frobnicates", agent="dev", collections=[cid], **ask))
    assert "frobnicates" in granted["context"]
    denied = asyncio.run(taosmd.answer_context("widget frobnicates", agent="stranger", collections=[cid], **ask))
    assert "frobnicates" not in denied["context"]
    only = asyncio.run(taosmd.answer_context(
        "widget frobnicates", agent="stranger", collections=[cid], collections_only=True, **ask))
    assert only["context"] == "" and only["hits_total"] == 0
    # and search itself says the same for the stranger
    hits = asyncio.run(taosmd_api.search("widget frobnicates", agent="stranger", collections=[cid],
                                         data_dir=data_dir))
    assert not any(h["metadata"].get("collection_id") for h in hits)


def test_scoping_helper_is_shared_with_search():
    """search resolves scope through one helper, and answer_context gets its hits
    by calling search itself rather than re-implementing the scoping."""
    import inspect

    assert "_resolve_search_agents(" in inspect.getsource(taosmd_api.search)
    assert "_api.search(" in inspect.getsource(ac.answer_context)


# ---------------------------------------------------------------------------
# ingest stores what answer_context needs
# ---------------------------------------------------------------------------

def test_ingest_records_session_fields_in_vector_metadata_and_archive(data_dir):
    asyncio.run(taosmd.ingest(
        [{"role": "user", "content": "hello there", "session_id": 7, "turn_index": 3,
          "session_date": "2023/05/20 (Sat) 02:21"}], agent="a", data_dir=data_dir))
    stores = asyncio.run(taosmd_api._ensure_stores(data_dir))
    rows = stores["vector"].get_session_rows({"7"})
    assert len(rows) == 1
    md = rows[0]["metadata"]
    assert (md["session_id"], md["turn_index"], md["session_date"], md["role"]) == (
        "7", 3, "2023/05/20 (Sat) 02:21", "user")
    hits = asyncio.run(stores["archive"].search_fts("hello", limit=3))
    assert any("session_id" in h["data_json"] for h in hits)


def test_ingest_ignores_malformed_session_fields(data_dir):
    asyncio.run(taosmd.ingest(
        [{"role": "user", "content": "odd turn", "session_id": ["x"], "turn_index": "3",
          "session_date": 5}], agent="a", data_dir=data_dir))
    stores = asyncio.run(taosmd_api._ensure_stores(data_dir))
    md = json.loads(stores["vector"]._conn.execute(
        "SELECT metadata_json FROM vector_memory").fetchone()[0])
    assert "session_id" not in md and "turn_index" not in md and "session_date" not in md


# ---------------------------------------------------------------------------
# HTTP and MCP return the library's dict
# ---------------------------------------------------------------------------

def _post(url, payload, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


class _Live:
    def __init__(self, tmp_path, monkeypatch, token=None):
        d = tmp_path / "taosmd-data"
        d.mkdir(exist_ok=True)
        self.data_dir = str(d)
        monkeypatch.setattr(taosmd_api, "_stores_cache", {})
        if token:
            monkeypatch.setenv("TAOSMD_TOKEN", token)
        else:
            monkeypatch.delenv("TAOSMD_TOKEN", raising=False)
        self.httpd = http_server.make_server("127.0.0.1", 0, data_dir=self.data_dir)
        stores = self.httpd.service_loop.run(taosmd_api._ensure_stores(self.data_dir))
        _patch_embedder(stores)
        host, port = self.httpd.server_address[:2]
        self.url = f"http://{host}:{port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)
        for s in list(taosmd_api._stores_cache.values()):
            for store in (s.get("archive"), s.get("vector"), s.get("kg")):
                if store and hasattr(store, "close"):
                    try:
                        self.httpd.service_loop.run(store.close())
                    except Exception:
                        pass
        self.httpd.service_loop.close()


@pytest.fixture
def live(tmp_path, monkeypatch):
    srv = _Live(tmp_path, monkeypatch)
    yield srv
    srv.close()


def _seed_http(live):
    for sid, (date, turns) in SESSIONS.items():
        for i, (role, text) in enumerate(turns):
            status, body = _post(f"{live.url}/ingest", {
                "text": text, "agent": "a", "role": role, "session_id": sid,
                "turn_index": i, "session_date": date})
            assert status == 200, body


def test_http_returns_the_same_dict_as_the_library(live):
    _seed_http(live)
    ask = {"query": SESSIONS["s1"][1][1][1], "agent": "a", "budget_tokens": 100_000,
           "now": NOW, "neighbors": 1, "limit": 1}
    status, body = _post(f"{live.url}/answer-context", ask)
    assert status == 200, body
    lib = live.httpd.service_loop.run(taosmd.answer_context(
        ask["query"], agent="a", data_dir=live.data_dir, budget_tokens=100_000, now=NOW,
        neighbors=1, limit=1))
    assert body == lib
    assert body["context"] == "Current date: 2023-06-10 (Sat)\n\n" + _block("s1", 0, 4)


def test_http_validation(live):
    base = {"query": "q", "agent": "a", "budget_tokens": 1000}
    assert _post(f"{live.url}/answer-context", {**base, "budget_tokens": None})[0] == 400
    assert _post(f"{live.url}/answer-context", {**base, "budget_tokens": -1})[0] == 400
    assert _post(f"{live.url}/answer-context", {**base, "query": ""})[0] == 400
    assert _post(f"{live.url}/answer-context", {**base, "agent": ""})[0] == 400
    assert _post(f"{live.url}/answer-context", {**base, "neighbors": "2"})[0] == 400
    assert _post(f"{live.url}/answer-context", {**base, "include_facts": "yes"})[0] == 400
    assert _post(f"{live.url}/answer-context", {**base, "now": "not a date"})[0] == 400
    assert _post(f"{live.url}/answer-context", {**base, "answer_reserve": 512})[0] == 400
    assert _post(f"{live.url}/answer-context", {**base, "reasoning_reserve": 0})[0] == 400


def test_http_rejects_the_same_unauthenticated_calls_as_search(tmp_path, monkeypatch):
    srv = _Live(tmp_path, monkeypatch, token="s3cret-token")
    try:
        body = {"query": "q", "agent": "a", "budget_tokens": 1000}
        assert _post(f"{srv.url}/search", {"query": "q", "agent": "a"})[0] == 401
        assert _post(f"{srv.url}/answer-context", body)[0] == 401
        assert _post(f"{srv.url}/answer-context", body, token="wrong")[0] == 401
        assert _post(f"{srv.url}/answer-context", body, token="s3cret-token")[0] == 200
    finally:
        srv.close()


def test_http_scoping_matches_search(live):
    """Agent a's HTTP context never holds agent b's turns."""
    _seed_http(live)
    for i, text in enumerate(["b secret one", "b secret two"]):
        _post(f"{live.url}/ingest", {"text": text, "agent": "b", "session_id": "s1",
                                     "turn_index": 10 + i, "session_date": "2023/05/21 (Sun) 08:00"})
    status, body = _post(f"{live.url}/answer-context", {
        "query": "Rex b secret", "agent": "a", "budget_tokens": 100_000, "now": NOW,
        "neighbors": 5, "limit": 10})
    assert status == 200
    assert "b secret" not in body["context"]


def test_mcp_returns_the_same_dict_as_the_library(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from taosmd import mcp_server

    d = tmp_path / "taosmd-data"
    d.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    mcp = mcp_server.build_server(data_dir=str(d))
    loop = mcp._taosmd_service_loop
    _patch_embedder(loop.run(taosmd_api._ensure_stores(str(d))))

    async def call(name, args):
        result = await mcp.call_tool(name, args)
        if isinstance(result, tuple):
            structured = result[1]
            if isinstance(structured, dict) and "result" in structured:
                return structured["result"]
        return json.loads(result[0].text)

    try:
        for sid, (date, turns) in SESSIONS.items():
            for i, (role, text) in enumerate(turns):
                out = asyncio.run(call("memory_ingest", {
                    "text": text, "agent": "a", "role": role, "session_id": sid,
                    "turn_index": i, "session_date": date}))
                assert out["archived"] == 1
        got = asyncio.run(call("memory_answer_context", {
            "query": SESSIONS["s3"][1][1][1], "agent": "a", "budget_tokens": 100_000,
            "now": NOW, "neighbors": 1, "limit": 1}))
        lib = loop.run(taosmd.answer_context(
            SESSIONS["s3"][1][1][1], agent="a", data_dir=str(d), budget_tokens=100_000, now=NOW,
            neighbors=1, limit=1))
        assert got == lib
        assert got["context"] == "Current date: 2023-06-10 (Sat)\n\n" + _block("s3", 0, 4)
    finally:
        for s in list(taosmd_api._stores_cache.values()):
            for store in (s.get("archive"), s.get("vector"), s.get("kg")):
                if store and hasattr(store, "close"):
                    try:
                        loop.run(store.close())
                    except Exception:
                        pass
        loop.close()


# ---------------------------------------------------------------------------
# archive hits, rebuilds, remote ingest
# ---------------------------------------------------------------------------

def test_archive_sourced_hit_joins_its_session_with_role_and_date(monkeypatch, data_dir):
    """An archive hit (80 character summary) is lifted from data_json: right role,
    its session, the session date, neighbours, and no duplicate of the vector turn."""
    from taosmd import retrieval

    long_turn = "Congrats on adopting Rex, that is wonderful news about the beagle puppy named Rex today!"
    assert len(long_turn) > 80
    sessions = {"s1": ("2023/05/20 (Sat) 02:21", [
        ("user", "I adopted a dog named Rex."), ("assistant", long_turn),
        ("user", "He is two years old.")])}
    _ingest_sessions(data_dir, sessions=sessions)
    stores = asyncio.run(taosmd_api._ensure_stores(data_dir))
    row = asyncio.run(stores["archive"].search_fts("wonderful", limit=3))[0]
    hit = taosmd_api._format_hit(retrieval._adapt_archive([row])[0])
    assert hit["source"] == "archive" and hit["text"] != long_turn

    async def fake_search(query, **kw):
        return [hit]

    monkeypatch.setattr(ac._api, "search", fake_search)
    out = _ctx(data_dir, neighbors=1)
    assert out["context"] == (
        "Current date: 2023-06-10 (Sat)\n\nSession 2023-05-20 (Sat)\n"
        f"user: I adopted a dog named Rex.\nassistant: {long_turn}\nuser: He is two years old.")
    assert out["included"][0]["session_id"] == "s1"
    assert out["included"][0]["turns"][1]["archive_id"] == row["id"]


def test_reindex_and_reconcile_keep_session_fields_and_the_archived_time(data_dir):
    _ingest_sessions(data_dir, sessions={"s1": SESSIONS["s1"]})
    stores = asyncio.run(taosmd_api._ensure_stores(data_dir))
    before = {r["metadata"]["turn_index"]: r["metadata"]
              for r in stores["vector"].get_session_rows({"s1"})}
    asyncio.run(taosmd_api.reindex(agent="a", data_dir=data_dir))
    after = {r["metadata"]["turn_index"]: r["metadata"]
             for r in stores["vector"].get_session_rows({"s1"})}
    assert set(after) == set(before) == {0, 1, 2, 3}
    for i in before:
        assert after[i]["session_id"] == "s1"
        assert after[i]["session_date"] == SESSIONS["s1"][0]
    # reconcile after losing a vector row restores it with its session fields
    stores["vector"]._conn.execute("DELETE FROM vector_memory WHERE json_extract(metadata_json, '$.turn_index') = 2")
    stores["vector"]._conn.commit()
    out = asyncio.run(taosmd_api.reconcile(agent="a", data_dir=data_dir))
    assert out["readded"] == 1
    rows = {r["metadata"]["turn_index"]: r["metadata"] for r in stores["vector"].get_session_rows({"s1"})}
    assert rows[2]["session_id"] == "s1" and rows[2]["session_date"] == SESSIONS["s1"][0]
    assert isinstance(rows[2]["timestamp"], float)  # the archived time, not the rebuild time


def test_remote_ingest_forwards_session_fields():
    from taosmd.remote import RemoteClient

    sent = {}
    rc = RemoteClient("http://fake:9999")

    async def fake_run(method, path, body=None, params=None):
        sent.update(path=path, body=body)
        return {"archived": 1}

    rc._run = fake_run  # type: ignore[method-assign]
    asyncio.run(rc.ingest(
        {"role": "user", "content": "hello", "session_id": "s1", "turn_index": 4,
         "session_date": "2023/05/20 (Sat) 02:21"}, "a", project="p"))
    assert sent["path"] == "/ingest"
    assert sent["body"] == {"text": "hello", "agent": "a", "role": "user", "session_id": "s1",
                            "turn_index": 4, "session_date": "2023/05/20 (Sat) 02:21", "project": "p"}
    asyncio.run(rc.ingest("plain", "a"))
    assert sent["body"] == {"text": "plain", "agent": "a"}


def test_product_environment_reports_a_missing_reranker(data_dir, monkeypatch):
    from benchmarks import product_context as pc
    from taosmd import cross_encoder

    class Gone:
        available = False

        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(cross_encoder, "CrossEncoderReranker", Gone)
    from taosmd import recipes
    recipe = recipes.Recipe(id="x", name="x", retrieval={"reranker": "bge-v2-m3", "fusion": "boost",
                            "strategy": "thorough"}, ingest={}, generator={}, librarian={})
    monkeypatch.setattr(recipes, "resolve_recipe", lambda agent, data_dir=None: recipe)
    env = asyncio.run(pc.product_environment(data_dir))
    assert env["reranker"] == "bge-v2-m3" and env["reranker_available"] is False
    with pytest.raises(SystemExit) as exc:
        pc.require_healthy(env)
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# coverage before depth
# ---------------------------------------------------------------------------

def _grown(data_dir, tokenizer=_chars, **kw):
    kw.setdefault("grow", True)
    return _ctx(data_dir, tokenizer=tokenizer, **kw)


def _spans(out):
    return {b["session_id"]: b["span"] for b in out["included"]}


def test_a_budget_that_fits_every_candidate_session_whole_gives_every_turn(monkeypatch, data_dir):
    """The oracle case: whole sessions fit, so every turn of every candidate is in."""
    _ingest_sessions(data_dir)
    _ranked(monkeypatch, data_dir, [
        SESSIONS["s3"][1][1][1], SESSIONS["s1"][1][2][1], SESSIONS["s2"][1][0][1]])
    out = _grown(data_dir, budget_tokens=100_000)
    for sid, (_, turns) in SESSIONS.items():
        for role, text in turns:
            assert f"{role}: {text}" in out["context"]
    assert _spans(out) == {"s1": "whole", "s2": "whole", "s3": "whole"}
    assert out["context"] == (
        "Current date: 2023-06-10 (Sat)\n\n" + _block("s1", 0, 4) + "\n\n"
        + _block("s2", 0, 4) + "\n\n" + _block("s3", 0, 4))


def test_every_session_gets_its_window_before_any_is_grown(monkeypatch, data_dir):
    """Budget = exactly the three windows. A depth-first selector would grow the
    first session and starve the third; coverage-first keeps all three."""
    _ingest_sessions(data_dir)
    _ranked(monkeypatch, data_dir, [
        SESSIONS["s1"][1][1][1], SESSIONS["s2"][1][1][1], SESSIONS["s3"][1][1][1]])
    windows = [_block(s, 0, 3) for s in ("s1", "s2", "s3")]
    out = _grown(data_dir, budget_tokens=_budget_for(windows))
    assert out["sessions_included"] == 3 and out["sessions_dropped"] == 0
    assert _spans(out) == {"s1": "window", "s2": "window", "s3": "window"}
    assert out["context"] == "Current date: 2023-06-10 (Sat)\n\n" + "\n\n".join(windows)


def test_leftover_budget_grows_sessions_in_rank_order(monkeypatch, data_dir):
    _ingest_sessions(data_dir)
    _ranked(monkeypatch, data_dir, [          # s2 is the best session
        SESSIONS["s2"][1][1][1], SESSIONS["s1"][1][1][1], SESSIONS["s3"][1][1][1]])
    extra = len("\nassistant: " + SESSIONS["s2"][1][3][1])        # s2's 4th turn
    windows = [_block(s, 0, 3) for s in ("s1", "s2", "s3")]
    out = _grown(data_dir, budget_tokens=_budget_for(windows, extra))
    assert _spans(out) == {"s1": "window", "s2": "whole", "s3": "window"}
    assert SESSIONS["s2"][1][3][1] in out["context"]
    assert SESSIONS["s1"][1][3][1] not in out["context"]
    assert out["tokens_used"] <= out["budget_available"]


def test_a_session_too_big_to_grow_keeps_its_window(monkeypatch, data_dir):
    big = {"big": ("2023/05/01 (Mon) 10:00",
                   [("user" if i % 2 == 0 else "assistant",
                     f"filler {i} " + " ".join(f"u{i}x{k}" for k in range(70)))
                    for i in range(30)])}
    _ingest_sessions(data_dir, sessions={**big, "s1": SESSIONS["s1"]})
    _ranked(monkeypatch, data_dir, [big["big"][1][5][1], SESSIONS["s1"][1][1][1]])
    out = _grown(data_dir, budget_tokens=4000)
    assert _spans(out) == {"big": "window", "s1": "whole"}
    big_block = next(b for b in out["included"] if b["session_id"] == "big")
    assert [t["turn_index"] for t in big_block["turns"]] == [4, 5, 6]


def test_grow_false_never_grows(monkeypatch, data_dir):
    _ingest_sessions(data_dir)
    _ranked(monkeypatch, data_dir, [SESSIONS["s1"][1][1][1]])
    out = _ctx(data_dir, budget_tokens=100_000)
    assert _spans(out) == {"s1": "window"}


def test_candidate_limit_grows_with_the_budget(monkeypatch, data_dir):
    seen = []

    async def fake_search(query, **kw):
        seen.append(kw["limit"])
        return []

    monkeypatch.setattr(ac._api, "search", fake_search)
    for budget in (1_000, 5_000, 10_256, 40_000):
        _ctx(data_dir, budget_tokens=budget)
    _ctx(data_dir, budget_tokens=40_000, limit=3)
    assert seen == [20, 20, 41, 160, 3]
    assert ac.default_limit(10_256) == 41


def test_budget_sized_limit_is_not_capped_by_the_recipe_pool(data_dir, monkeypatch):
    """The real search path: a recipe pool of 20 must not cap a limit of 50."""
    from taosmd import recipes

    turns = [("user", " ".join(f"p{i}x{k}" for k in range(12))) for i in range(60)]
    _ingest_sessions(data_dir, sessions={f"t{j}": ("2023/05/20 (Sat) 02:21", turns[j * 20:(j + 1) * 20])
                                          for j in range(3)})
    recipe = recipes.Recipe(
        id="pool20", name="pool20", ingest={}, generator={}, librarian={},
        retrieval={"strategy": "thorough", "limit": 5, "candidate_top_k": 20, "fusion": "boost",
                   "reranker": "none", "adjacent_neighbors": 0})
    monkeypatch.setattr(recipes, "resolve_recipe", lambda agent, data_dir=None: recipe)
    small = asyncio.run(taosmd_api.search("p3x1", agent="a", data_dir=data_dir, limit=50))
    assert len(small) < 30  # search itself stays capped by the recipe pool (20, plus a kg/archive hit)
    out = asyncio.run(taosmd.answer_context(
        "p3x1", agent="a", data_dir=data_dir, budget_tokens=100_000, now=NOW, limit=50, grow=False))
    assert out["hits_total"] == 50


def test_http_ungranted_collection_is_absent(tmp_path, monkeypatch):
    from taosmd import config as taosmd_config
    from taosmd.collections import CollectionStore, ingest_folder

    monkeypatch.delenv("TAOSMD_COLLECTIONS_ALLOWED_ROOTS", raising=False)
    srv = _Live(tmp_path, monkeypatch)
    try:
        source = tmp_path / "src"
        source.mkdir()
        (source / "readme.md").write_text("# Widget\n\nThe widget frobnicates the sprocket.")
        taosmd_config.set_collections_allowed_roots([str(source)], data_dir=srv.data_dir)
        store = CollectionStore(srv.data_dir)
        col = store.create(name="docs", kind="docs", source_path=str(source))
        store.grant(col["id"], "dev")
        srv.httpd.service_loop.run(ingest_folder(col["id"], data_dir=srv.data_dir))
        ask = {"query": "widget frobnicates", "budget_tokens": 50_000, "now": NOW,
               "collections": [col["id"]]}
        _, granted = _post(f"{srv.url}/answer-context", {**ask, "agent": "dev"})
        _, denied = _post(f"{srv.url}/answer-context", {**ask, "agent": "stranger"})
        _, only = _post(f"{srv.url}/answer-context", {**ask, "agent": "stranger", "collections_only": True})
        assert "frobnicates" in granted["context"]
        assert "frobnicates" not in denied["context"]
        assert only["context"] == "" and only["hits_total"] == 0
    finally:
        srv.close()


def test_mcp_agent_a_never_sees_agent_b(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from taosmd import mcp_server

    d = tmp_path / "taosmd-data"
    d.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    mcp = mcp_server.build_server(data_dir=str(d))
    loop = mcp._taosmd_service_loop
    _patch_embedder(loop.run(taosmd_api._ensure_stores(str(d))))

    async def call(name, args):
        result = await mcp.call_tool(name, args)
        if isinstance(result, tuple) and isinstance(result[1], dict) and "result" in result[1]:
            return result[1]["result"]
        return json.loads(result[0].text)

    try:
        for agent, tag in (("a", "alpha"), ("b", "bravo")):
            for i in range(3):
                asyncio.run(call("memory_ingest", {
                    "text": f"{tag} turn {i} about Rex", "agent": agent, "session_id": "same",
                    "turn_index": i, "session_date": "2023/05/20 (Sat) 02:21", "role": "user"}))
        out = asyncio.run(call("memory_answer_context", {
            "query": "Rex", "agent": "a", "budget_tokens": 50_000, "now": NOW, "limit": 10}))
        assert "alpha turn 1" in out["context"]
        assert "bravo" not in out["context"]
    finally:
        for s in list(taosmd_api._stores_cache.values()):
            for store in (s.get("archive"), s.get("vector"), s.get("kg")):
                if store and hasattr(store, "close"):
                    try:
                        loop.run(store.close())
                    except Exception:
                        pass
        loop.close()


# ---------------------------------------------------------------------------
# is_abstention
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "I don't know.", "I DO NOT KNOW the answer", "I\u2019m sorry, I can't say",
    "That is not in the context.", "The context does not contain it.",
    "There is no information about that.", "Hmm. I Don't Know",
])
def test_is_abstention_true(text):
    assert taosmd.is_abstention(text) is True


@pytest.mark.parametrize("text", [
    "Rex is a beagle.", "He adopted a dog in May.", "", "   ", None, 7,
    "The information was given in session two.",
])
def test_is_abstention_false(text):
    assert taosmd.is_abstention(text) is False


def test_is_abstention_matches_the_longmemeval_runner_list():
    src = (Path(__file__).resolve().parent.parent / "benchmarks" / "longmemeval_runner.py").read_text()
    for phrase in ac._ABSTENTION_PHRASES:
        assert f'"{phrase}"' in src


def test_a_standalone_hit_repeating_a_grown_session_line_is_not_printed_twice(monkeypatch, data_dir):
    _ingest_sessions(data_dir, sessions={"s1": SESSIONS["s1"]})
    session_hit = _row(data_dir, SESSIONS["s1"][1][0][1])
    twin = {"text": SESSIONS["s1"][1][3][1], "source": "vector", "timestamp": 0, "confidence": 0.4,
            "metadata": {"role": "assistant"}}   # same text as turn 3, but with no session info

    async def fake_search(query, **kw):
        return [session_hit, twin]

    monkeypatch.setattr(ac._api, "search", fake_search)
    out = _ctx(data_dir, neighbors=0, grow=True)
    assert out["context"].count(SESSIONS["s1"][1][3][1]) == 1
    assert out["sessions_included"] == 1


def test_default_budget_tokens_accepts_window_and_used_positionally():
    assert ac.default_budget_tokens(32768, 22000) == 10256
    assert ac.default_budget_tokens(32768, 22000, reasoning_reserve=1024) == 9232
    assert ac.default_budget_tokens(32768, 22000, reasoning_reserve=1536) == 8720
    assert ac.default_budget_tokens(65536) == 26214
