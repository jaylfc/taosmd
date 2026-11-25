"""Scoped, audited, reversible supersede (card tsk-supersede-fix).

A contradiction found by the extractor must never soft-hide unrelated vector
rows, and every row that is hidden must be traceable and restorable. These
tests pin that: no store-wide substring hide, regex facts do not auto-resolve
by default, an explicit supersede is agent-scoped, guarded and audited, and
``list_hidden`` / ``restore`` / ``restore_all`` always work.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time

import pytest

from taosmd import api as taosmd_api
from taosmd import config
from taosmd.cli import main
from taosmd.knowledge_graph import TemporalKnowledgeGraph
from taosmd.memory_extractor import process_conversation_turn
from taosmd.pending_decisions import PendingDecisionsStore
from taosmd.vector_memory import SupersedeRefused, VectorMemory


def _run(coro):
    return asyncio.run(coro)


def _fake_embed(vmem: VectorMemory) -> None:
    async def _embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFFFFFFFFFF
        return [((h >> (i * 3)) & 0xFF) / 255.0 - 0.5 for i in range(16)]

    vmem.embed = _embed  # type: ignore[assignment]


@pytest.fixture
def stores(tmp_path):
    vmem = VectorMemory(db_path=str(tmp_path / "vec.db"), embed_mode="onnx")
    _run(vmem.init())
    _fake_embed(vmem)
    kg = TemporalKnowledgeGraph(db_path=str(tmp_path / "kg.db"))
    _run(kg.init())
    yield vmem, kg
    _run(vmem.close())
    _run(kg.close())


def _add(vmem, text, agent):
    return _run(vmem.add(text, {"agent": agent}))


def _active_ids(vmem) -> set[int]:
    return {r[0] for r in vmem._conn.execute(
        "SELECT id FROM vector_memory WHERE valid_to IS NULL")}


def _audit(vmem):
    return [dict(r) for r in vmem._conn.execute(
        "SELECT row_id, reason, ref, caller FROM supersede_audit ORDER BY id")]


def _current_is_a(kg, name="office"):
    return [t["object_name"] for t in _run(kg.query_entity(name))
            if t["predicate"] == "is_a" and t["current"]
            and t["direction"] == "outgoing"]


def _search_texts(vmem, query, agents=None):
    hits = _run(vmem.search(query, limit=20, hybrid=False, fusion="none",
                            search_agents=agents))
    return {h["text"] for h in hits}


# ---------------------------------------------------------------------------
# The bug: a regex contradiction must not hide other rows
# ---------------------------------------------------------------------------

def test_regex_contradiction_hides_nothing_store_wide(stores):
    vmem, kg = stores
    ids = [
        _add(vmem, "agent a note: the server room is cold today", "agent-a"),
        _add(vmem, "agent a other: cold brew recipe", "agent-a"),
        _add(vmem, "agent b note: cold storage is full", "agent-b"),
        _add(vmem, "agent b other: a cold wind blew", "agent-b"),
    ]
    _run(process_conversation_turn(
        "The office is cold.", "agent-a", kg, vmem=vmem))
    _run(process_conversation_turn(
        "The office is nice.", "agent-a", kg, vmem=vmem))

    assert _active_ids(vmem) == set(ids)
    assert "agent b note: cold storage is full" in _search_texts(vmem, "cold")


def test_regex_contradiction_is_queued_as_pending(stores, tmp_path):
    vmem, kg = stores
    first = _run(process_conversation_turn(
        "The office is cold.", "agent-a", kg, vmem=vmem))
    second = _run(process_conversation_turn(
        "The office is nice.", "agent-a", kg, vmem=vmem))
    assert first["triples_created"] == 1
    assert second["method"] == "regex"
    assert second["contradictions_resolved"] == 0
    assert second["deferred"] == 1

    # The old triple is still active; the conflict waits in the queue.
    assert _current_is_a(kg) == ["cold"]
    store = PendingDecisionsStore(db_path=tmp_path / "kg.db")
    _run(store.init())
    try:
        pending = _run(store.list_pending())
    finally:
        _run(store.close())
    assert len(pending) == 1
    assert pending[0]["kind"] == "contradiction"
    assert pending[0]["new_object"] == "nice"


def test_regex_opt_in_resolves_kg_only_never_vectors(stores):
    vmem, kg = stores
    ids = [_add(vmem, "agent b note: cold storage is full", "agent-b"),
           _add(vmem, "The office is cold.", "agent-a")]
    _run(process_conversation_turn(
        "The office is cold.", "agent-a", kg, vmem=vmem,
        auto_resolve_contradictions=True))
    out = _run(process_conversation_turn(
        "The office is nice.", "agent-a", kg, vmem=vmem,
        auto_resolve_contradictions=True))
    assert out["contradictions_resolved"] == 1
    assert _current_is_a(kg) == ["nice"]
    assert _active_ids(vmem) == set(ids)
    assert _audit(vmem) == []


def test_kg_check_with_vmem_never_touches_vector_layer(stores):
    vmem, kg = stores
    rid = _add(vmem, "jay works on ProjectAlpha", "agent-a")
    _run(kg.add_triple_with_contradiction_check(
        "jay", "works_on", "ProjectAlpha", vmem=vmem))
    res = _run(kg.add_triple_with_contradiction_check(
        "jay", "works_on", "ProjectBeta", vmem=vmem))
    assert res["contradictions_resolved"] == 1
    assert _active_ids(vmem) == {rid}


# ---------------------------------------------------------------------------
# Explicit supersede: agent scope, source link, guards, audit
# ---------------------------------------------------------------------------

def test_supersede_row_is_agent_scoped_and_audited(stores):
    vmem, _ = stores
    a = _add(vmem, "agent a fact", "agent-a")
    b = _add(vmem, "agent b fact", "agent-b")
    # Wrong agent scope: the linked row belongs to someone else, nothing hides.
    assert _run(vmem.supersede(b, agent="agent-a")) is False
    assert _active_ids(vmem) == {a, b}
    assert _audit(vmem) == []
    # Right scope: only that row hides, and the audit row names it.
    assert _run(vmem.supersede(a, agent="agent-a", ref="triple-1", caller="test")) is True
    assert _active_ids(vmem) == {b}
    [row] = _audit(vmem)
    assert row == {"row_id": a, "reason": "explicit", "ref": "triple-1", "caller": "test"}


def test_supersede_matching_requires_agent_scope(stores):
    vmem, _ = stores
    _add(vmem, "the quarterly report is ready", "agent-a")
    with pytest.raises(SupersedeRefused):
        _run(vmem.supersede_matching("the quarterly report"))
    assert len(_active_ids(vmem)) == 1


def test_supersede_matching_only_hides_agents_rows_and_returns_ids(stores):
    vmem, _ = stores
    a = _add(vmem, "the quarterly report is ready", "agent-a")
    b = _add(vmem, "the quarterly report is late", "agent-b")
    legacy = _run(vmem.add("the quarterly report has no agent tag"))
    hidden = _run(vmem.supersede_matching("the quarterly report", agent="agent-a"))
    assert hidden == [a]
    assert _active_ids(vmem) == {b, legacy}
    [row] = _audit(vmem)
    assert row["reason"] == "substring"
    assert row["ref"] == "the quarterly report"
    assert row["row_id"] == a


def test_supersede_matching_refuses_short_needle(stores):
    vmem, _ = stores
    _add(vmem, "it is cold in here", "agent-a")
    with pytest.raises(SupersedeRefused):
        _run(vmem.supersede_matching("cold", agent="agent-a"))
    assert len(_active_ids(vmem)) == 1
    assert _run(vmem.supersede_matching("cold", agent="agent-a", force=True)) != []


def test_supersede_matching_refuses_too_many_rows(stores):
    vmem, _ = stores
    for i in range(21):
        _add(vmem, f"standup notes number {i}", "agent-a")
    with pytest.raises(SupersedeRefused):
        _run(vmem.supersede_matching("standup notes number", agent="agent-a"))
    assert len(_active_ids(vmem)) == 21
    assert _audit(vmem) == []
    done = _run(vmem.supersede_matching("standup notes number", agent="agent-a",
                                        max_rows=50))
    assert len(done) == 21


def test_supersede_matching_blank_needle_is_a_noop(stores):
    vmem, _ = stores
    _add(vmem, "something", "agent-a")
    assert _run(vmem.supersede_matching("   ", agent="agent-a", force=True)) == []
    assert len(_active_ids(vmem)) == 1


def test_supersede_matching_force_without_agent_is_store_wide(stores):
    vmem, _ = stores
    _add(vmem, "the quarterly report is ready", "agent-a")
    _add(vmem, "the quarterly report is late", "agent-b")
    assert len(_run(vmem.supersede_matching("the quarterly report", force=True))) == 2


# ---------------------------------------------------------------------------
# list_hidden / restore / restore_all
# ---------------------------------------------------------------------------

def test_list_hidden_marks_pre_audit_rows_unknown(stores):
    vmem, _ = stores
    old = _add(vmem, "hidden long ago " + "x" * 120, "agent-a")
    vmem._conn.execute("UPDATE vector_memory SET valid_to = 111.0 WHERE id = ?", (old,))
    vmem._conn.commit()
    new = _add(vmem, "hidden just now", "agent-b")
    _run(vmem.supersede(new, agent="agent-b", caller="test"))
    _add(vmem, "still visible", "agent-a")

    rows = {r["id"]: r for r in _run(vmem.list_hidden())}
    assert set(rows) == {old, new}
    assert rows[old]["reason"] == "unknown (before audit log)"
    assert rows[old]["agent"] == "agent-a"
    assert len(rows[old]["text"]) <= 80
    assert rows[old]["valid_to"] == 111.0
    assert rows[new]["reason"] == "explicit"
    assert [r["id"] for r in _run(vmem.list_hidden(agent="agent-b"))] == [new]


def test_restore_brings_rows_back_into_search_and_audits(stores):
    vmem, _ = stores
    rid = _add(vmem, "restore me please", "agent-a")
    _run(vmem.supersede(rid, agent="agent-a"))
    assert "restore me please" not in _search_texts(vmem, "restore me please")
    assert _run(vmem.restore([rid, 9999], caller="test")) == [rid]
    assert "restore me please" in _search_texts(vmem, "restore me please")
    assert _run(vmem.list_hidden()) == []
    assert [a["reason"] for a in _audit(vmem)] == ["explicit", "restored"]
    # Restoring an already-visible row is a clean no-op.
    assert _run(vmem.restore([rid])) == []


def test_restore_clears_hidden_by_marker(stores):
    vmem, _ = stores
    rid = _run(vmem.add("shelf hidden row", {"agent": "agent-a",
                                             "hidden_by": "shelf-archive:1"}))
    vmem._conn.execute("UPDATE vector_memory SET valid_to = 5.0 WHERE id = ?", (rid,))
    vmem._conn.commit()
    [row] = _run(vmem.list_hidden())
    assert row["reason"] == "hidden_by shelf-archive:1"
    _run(vmem.restore([rid]))
    meta = json.loads(vmem._conn.execute(
        "SELECT metadata_json FROM vector_memory WHERE id = ?", (rid,)).fetchone()[0])
    assert "hidden_by" not in meta
    assert rid in _active_ids(vmem)


def test_restore_all_since_and_agent(stores):
    vmem, _ = stores
    ids = {n: _add(vmem, f"row {n}", ag) for n, ag in
           (("a_old", "agent-a"), ("a_new", "agent-a"), ("b_new", "agent-b"))}
    t = time.time()
    for n, when in (("a_old", t - 10 * 86400), ("a_new", t - 100), ("b_new", t - 50)):
        vmem._conn.execute("UPDATE vector_memory SET valid_to = ? WHERE id = ?",
                           (when, ids[n]))
    vmem._conn.commit()
    since = time.strftime("%Y-%m-%d", time.gmtime(t - 86400))
    restored = _run(vmem.restore_all(since=since, agent="agent-a"))
    assert restored == [ids["a_new"]]
    restored = _run(vmem.restore_all(since=since))
    assert restored == [ids["b_new"]]
    assert [r["id"] for r in _run(vmem.list_hidden())] == [ids["a_old"]]


def test_audit_table_is_created_on_a_pre_existing_store(tmp_path):
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.executescript(
        "CREATE TABLE vector_memory (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "text TEXT NOT NULL, embedding TEXT NOT NULL, metadata_json TEXT NOT NULL "
        "DEFAULT '{}', created_at REAL NOT NULL, valid_to REAL);"
        "INSERT INTO vector_memory (text, embedding, created_at, valid_to) "
        "VALUES ('legacy hidden', '[]', 1.0, 2.0);")
    con.commit()
    con.close()
    vmem = VectorMemory(db_path=str(db), embed_mode="onnx")
    _run(vmem.init())
    try:
        [row] = _run(vmem.list_hidden())
        assert row["reason"] == "unknown (before audit log)"
        assert _run(vmem.restore([row["id"]])) == [row["id"]]
    finally:
        _run(vmem.close())


# ---------------------------------------------------------------------------
# api, service and CLI surface
# ---------------------------------------------------------------------------

@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "taosmd-data"
    d.mkdir()
    monkeypatch.setenv("TAOSMD_DATA_DIR", str(d))
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    config.set_collections_allowed_roots([], data_dir=str(d))
    vmem = _run(taosmd_api._ensure_stores(str(d)))["vector"]
    _fake_embed(vmem)
    _run(vmem.add("the quarterly report is ready", {"agent": "agent-a"}))
    _run(vmem.add("the quarterly report is late", {"agent": "agent-b"}))
    yield str(d)
    for s in list(taosmd_api._stores_cache.values()):
        try:
            _run(s["vector"].close())
        except Exception:
            pass


def test_api_supersede_vectors_is_scoped(data_dir):
    assert _run(taosmd_api.supersede_vectors(
        "the quarterly report", agent="agent-a", data_dir=data_dir)) == 1
    with pytest.raises(SupersedeRefused):
        _run(taosmd_api.supersede_vectors("the quarterly report", data_dir=data_dir))


def test_cli_supersede_requires_agent(data_dir, capsys):
    rc = main(["supersede", "--match", "the quarterly report"])
    assert rc == 2
    assert "agent" in capsys.readouterr().err


def test_cli_supersede_list_and_restore_roundtrip(data_dir, capsys):
    assert main(["supersede", "--agent", "agent-a", "--match", "the quarterly report"]) == 0
    capsys.readouterr()

    assert main(["supersede", "--list-hidden", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["agent"] for r in rows] == ["agent-a"]
    assert rows[0]["reason"] == "substring"
    rid = rows[0]["id"]

    assert main(["supersede", "--list-hidden", "--agent", "agent-b"]) == 0
    assert "agent-a" not in capsys.readouterr().out

    assert main(["supersede", "--restore", str(rid)]) == 0
    assert "restored 1" in capsys.readouterr().out
    assert main(["supersede", "--list-hidden", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_cli_restore_all_needs_since(data_dir, capsys):
    assert main(["supersede", "--restore-all"]) == 2
    assert main(["supersede", "--agent", "agent-a", "--match", "the quarterly report"]) == 0
    capsys.readouterr()
    assert main(["supersede", "--restore-all", "--since", "2000-01-01",
                 "--agent", "agent-a"]) == 0
    assert "restored 1" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Realistic size: 15,000 generated rows across 3 agents
# ---------------------------------------------------------------------------

def test_fifteen_thousand_rows_three_agents(data_dir, capsys):
    agents = ["agent-a", "agent-b", "agent-c"]
    stores = _run(taosmd_api._ensure_stores(data_dir))
    vmem = stores["vector"]
    _fake_embed(vmem)
    now = time.time()
    rows = [
        (f"turn {i} for {agents[i % 3]}: the office was cold, ticket {i}", "[]",
         json.dumps({"agent": agents[i % 3]}), now)
        for i in range(15000)
    ]
    vmem._conn.executemany(
        "INSERT INTO vector_memory (text, embedding, metadata_json, created_at) "
        "VALUES (?, ?, ?, ?)", rows)
    vmem._conn.commit()
    total = vmem._conn.execute("SELECT COUNT(*) FROM vector_memory").fetchone()[0]

    def hidden_count():
        return vmem._conn.execute(
            "SELECT COUNT(*) FROM vector_memory WHERE valid_to IS NOT NULL"
        ).fetchone()[0]

    # Default-path contradiction hides nothing, in any agent.
    kg = TemporalKnowledgeGraph(db_path=f"{data_dir}/knowledge-graph.db")
    _run(kg.init())
    try:
        _run(process_conversation_turn("The office is cold.", "agent-a", kg, vmem=vmem))
        out = _run(process_conversation_turn("The office is nice.", "agent-a", kg, vmem=vmem))
    finally:
        _run(kg.close())
    assert out["deferred"] == 1
    assert hidden_count() == 0

    # An explicit, source-linked correction hides exactly the linked row.
    target = vmem._conn.execute(
        "SELECT id FROM vector_memory WHERE text LIKE 'turn 7 for %'").fetchone()[0]
    assert _run(vmem.supersede(target, agent="agent-b", ref="triple-7")) is True
    assert hidden_count() == 1
    assert vmem._conn.execute("SELECT COUNT(*) FROM vector_memory").fetchone()[0] == total

    # list-hidden and restore round-trip, each under 2 s.
    t0 = time.perf_counter()
    assert main(["supersede", "--list-hidden", "--json"]) == 0
    list_secs = time.perf_counter() - t0
    listed = json.loads(capsys.readouterr().out)
    assert [(r["id"], r["agent"], r["reason"]) for r in listed] == [
        (target, "agent-b", "explicit")]

    t0 = time.perf_counter()
    assert main(["supersede", "--restore", str(target)]) == 0
    restore_secs = time.perf_counter() - t0
    assert "restored 1" in capsys.readouterr().out
    assert hidden_count() == 0
    assert list_secs < 2.0, list_secs
    assert restore_secs < 2.0, restore_secs


# ---------------------------------------------------------------------------
# auto_resolve_contradictions: None / True / False, for regex and LLM facts
# ---------------------------------------------------------------------------

def _two_turns(kg, vmem, mode, extractor):
    """Run 'office is cold' then 'office is nice'; return (first, second, pending)."""
    import taosmd.memory_extractor as me
    from unittest.mock import patch

    kwargs = {}
    if mode != "default":
        kwargs["auto_resolve_contradictions"] = mode
    if extractor == "llm":
        async def fake_llm(text, *a, **k):
            return [{"subject": "office", "predicate": "is_a",
                     "object": "cold" if "cold" in text else "nice"}]
        kwargs.update(llm_url="http://unused", http_client=object())
        ctx = patch.object(me, "extract_facts_with_llm", fake_llm)
    else:
        ctx = patch.object(me, "extract_facts_with_llm", None)
    with ctx:
        first = _run(process_conversation_turn(
            "The office is cold.", "agent-a", kg, vmem=vmem, **kwargs))
        second = _run(process_conversation_turn(
            "The office is nice.", "agent-a", kg, vmem=vmem, **kwargs))
    assert second["method"] == extractor
    try:
        rows = kg._conn.execute(
            "SELECT COUNT(*) FROM kg_pending_decisions WHERE resolved_at IS NULL"
        ).fetchone()[0]
    except sqlite3.OperationalError:  # queue table never created: nothing queued
        rows = 0
    return first, second, rows


@pytest.mark.parametrize("extractor,mode,current,queued", [
    # default: regex facts are queued, LLM facts still resolve in the graph
    ("regex", "default", ["cold"], 1),
    ("llm", "default", ["nice"], 0),
    # True: old triple invalidated, new one written, for either extractor
    ("regex", True, ["nice"], 0),
    ("llm", True, ["nice"], 0),
    # False: keep both facts, nothing invalidated, nothing queued
    ("regex", False, ["cold", "nice"], 0),
    ("llm", False, ["cold", "nice"], 0),
])
def test_auto_resolve_modes(stores, extractor, mode, current, queued):
    vmem, kg = stores
    _add(vmem, "agent b note: cold storage is full", "agent-b")
    first, second, pending = _two_turns(kg, vmem, mode, extractor)
    assert first["triples_created"] == 1
    assert sorted(_current_is_a(kg)) == sorted(current)
    assert pending == queued
    assert second["triples_created"] == (0 if queued else 1)
    assert len(_active_ids(vmem)) == 1  # the vector layer is never touched
