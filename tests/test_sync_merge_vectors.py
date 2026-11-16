"""Tests for merge_vectors (account-device-sync S4 vector half).

Content-keyed merge by sha256(namespace + normalised text). No id reuse.
Peer opened read-only via URI. One transaction, rollback on error.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3

from taosmd.vector_memory import StoreModeMismatch, VectorMemory


def _deterministic_embedder(vmem: VectorMemory) -> None:
    """Patch embed() with a deterministic 16-dim hash vector (no ONNX/QMD)."""

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFFFFFFFFFF
        return [((h >> (i * 3)) & 0xFF) / 255.0 - 0.5 for i in range(16)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


def _make_store(db_path: str, *, binary_quant: bool = False) -> VectorMemory:
    vmem = VectorMemory(
        db_path=db_path,
        embed_mode="onnx",  # falls through; we patch embed() directly
        binary_quant=binary_quant,
    )
    asyncio.run(vmem.init())
    _deterministic_embedder(vmem)
    return vmem


def _row_count(db_path: str) -> int:
    con = sqlite3.connect(db_path)
    try:
        return con.execute("SELECT COUNT(*) FROM vector_memory").fetchone()[0]
    finally:
        con.close()


def _max_id(db_path: str) -> int:
    con = sqlite3.connect(db_path)
    try:
        row = con.execute("SELECT MAX(id) FROM vector_memory").fetchone()
        return row[0] if row[0] is not None else 0
    finally:
        con.close()


def _content_key(text: str, metadata: dict) -> str:
    """Compute the content key as specified: sha256 of json.dumps([agent, project, normalised_text])."""
    agent = metadata.get("agent") or ""
    project = metadata.get("project") or ""
    normalised = " ".join(text.split()).lower()
    return hashlib.sha256(json.dumps([agent, project, normalised]).encode()).hexdigest()


def test_merge_vectors_id_collision_different_texts(tmp_path):
    """Local id 1 = 'alpha', peer id 1 = 'beta' -> both texts present, local id unchanged."""
    from taosmd.sync_merge import merge_vectors

    local_db = tmp_path / "local.db"
    peer_db = tmp_path / "peer.db"

    # Local store: id 1 = "alpha"
    vmem_local = _make_store(str(local_db))
    try:
        rid_alpha = asyncio.run(vmem_local.add("alpha", metadata={"agent": "test"}))
        assert rid_alpha == 1
    finally:
        asyncio.run(vmem_local.close())

    # Peer store: id 1 = "beta" (same id, different text)
    vmem_peer = _make_store(str(peer_db))
    try:
        rid_beta = asyncio.run(vmem_peer.add("beta", metadata={"agent": "test"}))
        assert rid_beta == 1
    finally:
        asyncio.run(vmem_peer.close())

    # Before merge: local has 1 row (id=1, text="alpha")
    assert _row_count(str(local_db)) == 1
    assert _max_id(str(local_db)) == 1

    # Naive INSERT OR REPLACE would overwrite local id 1's text with "beta"
    # This helper proves the defect exists before our fix
    def naive_merge(local_path: str, peer_path: str) -> None:
        con = sqlite3.connect(local_path)
        peer_con = sqlite3.connect(peer_path)
        try:
            peer_rows = peer_con.execute("""
                SELECT id, text, embedding, metadata_json, created_at, valid_to
                FROM vector_memory
            """).fetchall()
            for row in peer_rows:
                con.execute("""
                    INSERT OR REPLACE INTO vector_memory (id, text, embedding, metadata_json, created_at, valid_to)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, row)
            con.commit()
        finally:
            peer_con.close()
            con.close()

    # Run naive merge on a copy to demonstrate the bug
    import shutil
    local_copy = tmp_path / "local_naive.db"
    shutil.copy2(local_db, local_copy)
    naive_merge(str(local_copy), str(peer_db))
    # Naive merge overwrites local id 1's text
    con = sqlite3.connect(str(local_copy))
    row = con.execute("SELECT text FROM vector_memory WHERE id = 1").fetchone()
    con.close()
    assert row[0] == "beta", "Naive merge overwrites local id 1's text - this is the defect"

    # Now run the real merge_vectors
    result = merge_vectors(str(local_db), str(peer_db))

    # Both texts should be present
    con = sqlite3.connect(str(local_db))
    rows = con.execute("SELECT id, text FROM vector_memory ORDER BY id").fetchall()
    con.close()

    texts = {r[1] for r in rows}
    assert "alpha" in texts
    assert "beta" in texts
    assert len(rows) == 2
    # Local alpha row keeps its id (1)
    assert rows[0][0] == 1
    assert rows[0][1] == "alpha"
    # Beta gets a new id (2)
    assert rows[1][0] == 2
    assert rows[1][1] == "beta"

    assert result["added"] == 1
    assert result["deduped"] == 0
    assert result["metadata_updated"] == 0


def test_merge_vectors_same_text_different_ids_deduped(tmp_path):
    """Same text on both (different ids, whitespace/case differences) -> one row (deduped=1)."""
    from taosmd.sync_merge import merge_vectors

    local_db = tmp_path / "local.db"
    peer_db = tmp_path / "peer.db"

    vmem_local = _make_store(str(local_db))
    try:
        asyncio.run(vmem_local.add("  Alpha Text  ", metadata={"agent": "test"}))
    finally:
        asyncio.run(vmem_local.close())

    vmem_peer = _make_store(str(peer_db))
    try:
        asyncio.run(vmem_peer.add("alpha text", metadata={"agent": "test"}))  # different case/whitespace
    finally:
        asyncio.run(vmem_peer.close())

    result = merge_vectors(str(local_db), str(peer_db))

    # Should be deduped to 1 row
    assert _row_count(str(local_db)) == 1
    assert result["added"] == 0
    assert result["deduped"] == 1
    assert result["metadata_updated"] == 0


def test_merge_vectors_newer_peer_metadata_replaces_local(tmp_path):
    """Newer peer metadata replaces local metadata for the same key; older does not."""
    from taosmd.sync_merge import merge_vectors

    local_db = tmp_path / "local.db"
    peer_db = tmp_path / "peer.db"

    # Local: created_at = 1000.0, metadata = {"version": "old"}
    vmem_local = _make_store(str(local_db))
    try:
        asyncio.run(vmem_local.add("shared fact", metadata={"version": "old", "agent": "test"}))
        # Manually set created_at to an old timestamp
        vmem_local._conn.execute(
            "UPDATE vector_memory SET created_at = ? WHERE text = ?",
            (1000.0, "shared fact"),
        )
        vmem_local._conn.commit()
    finally:
        asyncio.run(vmem_local.close())

    # Peer: created_at = 2000.0 (newer), metadata = {"version": "new"}
    vmem_peer = _make_store(str(peer_db))
    try:
        asyncio.run(vmem_peer.add("shared fact", metadata={"version": "new", "agent": "test"}))
        vmem_peer._conn.execute(
            "UPDATE vector_memory SET created_at = ? WHERE text = ?",
            (2000.0, "shared fact"),
        )
        vmem_peer._conn.commit()
    finally:
        asyncio.run(vmem_peer.close())

    result = merge_vectors(str(local_db), str(peer_db))

    # Peer metadata should win (newer created_at)
    con = sqlite3.connect(str(local_db))
    row = con.execute("SELECT metadata_json FROM vector_memory WHERE text = 'shared fact'").fetchone()
    con.close()
    meta = json.loads(row[0])
    assert meta["version"] == "new"
    assert result["metadata_updated"] == 1
    assert result["added"] == 0
    assert result["deduped"] == 1


def test_merge_vectors_older_peer_metadata_does_not_replace(tmp_path):
    """Older peer metadata does not replace local metadata."""
    from taosmd.sync_merge import merge_vectors

    local_db = tmp_path / "local.db"
    peer_db = tmp_path / "peer.db"

    # Local: created_at = 2000.0 (newer), metadata = {"version": "local_new"}
    vmem_local = _make_store(str(local_db))
    try:
        asyncio.run(vmem_local.add("shared fact", metadata={"version": "local_new", "agent": "test"}))
        vmem_local._conn.execute(
            "UPDATE vector_memory SET created_at = ? WHERE text = ?",
            (2000.0, "shared fact"),
        )
        vmem_local._conn.commit()
    finally:
        asyncio.run(vmem_local.close())

    # Peer: created_at = 1000.0 (older), metadata = {"version": "peer_old"}
    vmem_peer = _make_store(str(peer_db))
    try:
        asyncio.run(vmem_peer.add("shared fact", metadata={"version": "peer_old", "agent": "test"}))
        vmem_peer._conn.execute(
            "UPDATE vector_memory SET created_at = ? WHERE text = ?",
            (1000.0, "shared fact"),
        )
        vmem_peer._conn.commit()
    finally:
        asyncio.run(vmem_peer.close())

    result = merge_vectors(str(local_db), str(peer_db))

    # Local metadata should win (newer created_at)
    con = sqlite3.connect(str(local_db))
    row = con.execute("SELECT metadata_json FROM vector_memory WHERE text = 'shared fact'").fetchone()
    con.close()
    meta = json.loads(row[0])
    assert meta["version"] == "local_new"
    assert result["metadata_updated"] == 0
    assert result["added"] == 0
    assert result["deduped"] == 1


def test_merge_vectors_equal_created_at_lexicographic_metadata_wins(tmp_path):
    """On EQUAL created_at, lexicographically larger metadata_json wins (content tie-break)."""
    from taosmd.sync_merge import merge_vectors

    local_db = tmp_path / "local.db"
    peer_db = tmp_path / "peer.db"

    # Both have same created_at = 1500.0
    # Local metadata_json = '{"version": "a"}' (lexicographically smaller)
    vmem_local = _make_store(str(local_db))
    try:
        asyncio.run(vmem_local.add("shared fact", metadata={"version": "a", "agent": "test"}))
        vmem_local._conn.execute(
            "UPDATE vector_memory SET created_at = ? WHERE text = ?",
            (1500.0, "shared fact"),
        )
        vmem_local._conn.commit()
    finally:
        asyncio.run(vmem_local.close())

    # Peer metadata_json = '{"version": "z"}' (lexicographically larger)
    vmem_peer = _make_store(str(peer_db))
    try:
        asyncio.run(vmem_peer.add("shared fact", metadata={"version": "z", "agent": "test"}))
        vmem_peer._conn.execute(
            "UPDATE vector_memory SET created_at = ? WHERE text = ?",
            (1500.0, "shared fact"),
        )
        vmem_peer._conn.commit()
    finally:
        asyncio.run(vmem_peer.close())

    result = merge_vectors(str(local_db), str(peer_db))

    # Peer metadata should win (lexicographically larger)
    con = sqlite3.connect(str(local_db))
    row = con.execute("SELECT metadata_json FROM vector_memory WHERE text = 'shared fact'").fetchone()
    con.close()
    meta = json.loads(row[0])
    assert meta["version"] == "z"
    assert result["metadata_updated"] == 1


def test_merge_vectors_store_meta_mismatch_raises(tmp_path):
    """store_meta mismatch -> StoreModeMismatch, local db unchanged (row count + max id equal before/after)."""
    from taosmd.sync_merge import merge_vectors

    local_db = tmp_path / "local.db"
    peer_db = tmp_path / "peer.db"

    # Local: binary_quant = False
    vmem_local = _make_store(str(local_db), binary_quant=False)
    try:
        asyncio.run(vmem_local.add("local fact", metadata={"agent": "test"}))
    finally:
        asyncio.run(vmem_local.close())

    # Peer: binary_quant = True (different store mode)
    vmem_peer = _make_store(str(peer_db), binary_quant=True)
    try:
        asyncio.run(vmem_peer.add("peer fact", metadata={"agent": "test"}))
    finally:
        asyncio.run(vmem_peer.close())

    # Record state before
    before_count = _row_count(str(local_db))
    before_max_id = _max_id(str(local_db))

    try:
        merge_vectors(str(local_db), str(peer_db))
        assert False, "Expected StoreModeMismatch"
    except StoreModeMismatch:
        pass

    # Local db should be unchanged
    assert _row_count(str(local_db)) == before_count
    assert _max_id(str(local_db)) == before_max_id


def test_merge_vectors_row_count_never_decreases(tmp_path):
    """Row count never decreases after merge."""
    from taosmd.sync_merge import merge_vectors

    local_db = tmp_path / "local.db"
    peer_db = tmp_path / "peer.db"

    vmem_local = _make_store(str(local_db))
    try:
        for i in range(5):
            asyncio.run(vmem_local.add(f"local fact {i}", metadata={"agent": "test"}))
    finally:
        asyncio.run(vmem_local.close())

    vmem_peer = _make_store(str(peer_db))
    try:
        for i in range(3):
            asyncio.run(vmem_peer.add(f"peer fact {i}", metadata={"agent": "test"}))
    finally:
        asyncio.run(vmem_peer.close())

    before_count = _row_count(str(local_db))
    result = merge_vectors(str(local_db), str(peer_db))
    after_count = _row_count(str(local_db))

    assert after_count >= before_count
    assert after_count == before_count + result["added"]


def test_merge_vectors_determinism_across_devices(tmp_path):
    """Same key on both sides with EQUAL created_at and different metadata_json, ids differing;
    run merge_vectors(A, B) and merge_vectors(B, A) on copies, assert both sides end with identical
    (text, metadata_json, valid_to) sets."""
    from taosmd.sync_merge import merge_vectors

    # Create base databases
    base_a = tmp_path / "base_a.db"
    base_b = tmp_path / "base_b.db"

    vmem_a = _make_store(str(base_a))
    try:
        asyncio.run(vmem_a.add("shared fact", metadata={"version": "a", "agent": "test"}))
        vmem_a._conn.execute(
            "UPDATE vector_memory SET created_at = ? WHERE text = ?",
            (1000.0, "shared fact"),
        )
        vmem_a._conn.commit()
    finally:
        asyncio.run(vmem_a.close())

    vmem_b = _make_store(str(base_b))
    try:
        asyncio.run(vmem_b.add("shared fact", metadata={"version": "z", "agent": "test"}))  # lexicographically larger
        vmem_b._conn.execute(
            "UPDATE vector_memory SET created_at = ? WHERE text = ?",
            (1000.0, "shared fact"),
        )
        vmem_b._conn.commit()
    finally:
        asyncio.run(vmem_b.close())

    # Copy for A<-B merge
    copy_a = tmp_path / "copy_a.db"
    copy_b = tmp_path / "copy_b.db"
    import shutil
    shutil.copy2(base_a, copy_a)
    shutil.copy2(base_b, copy_b)

    # Merge A <- B
    merge_vectors(str(copy_a), str(copy_b))

    # Copy for B<-A merge
    copy_a2 = tmp_path / "copy_a2.db"
    copy_b2 = tmp_path / "copy_b2.db"
    shutil.copy2(base_a, copy_a2)
    shutil.copy2(base_b, copy_b2)

    # Merge B <- A
    merge_vectors(str(copy_b2), str(copy_a2))

    # Both should have identical content
    def get_rows(db_path: str):
        con = sqlite3.connect(db_path)
        rows = con.execute("SELECT text, metadata_json, valid_to FROM vector_memory ORDER BY text").fetchall()
        con.close()
        return [(r[0], r[1], r[2]) for r in rows]

    rows_a = get_rows(str(copy_a))
    rows_b = get_rows(str(copy_b2))

    assert rows_a == rows_b
    # The lexicographically larger metadata ("z") should win
    meta = json.loads(rows_a[0][1])
    assert meta["version"] == "z"


def test_merge_vectors_valid_to_from_peer(tmp_path):
    """When peer's valid_to is set, local valid_to is set to peer's."""
    from taosmd.sync_merge import merge_vectors

    local_db = tmp_path / "local.db"
    peer_db = tmp_path / "peer.db"

    # Local: active row (valid_to = NULL)
    vmem_local = _make_store(str(local_db))
    try:
        asyncio.run(vmem_local.add("shared fact", metadata={"agent": "test"}))
    finally:
        asyncio.run(vmem_local.close())

    # Peer: superseded row (valid_to = 5000.0)
    vmem_peer = _make_store(str(peer_db))
    try:
        asyncio.run(vmem_peer.add("shared fact", metadata={"agent": "test"}))
        vmem_peer._conn.execute(
            "UPDATE vector_memory SET valid_to = ? WHERE text = ?",
            (5000.0, "shared fact"),
        )
        vmem_peer._conn.commit()
    finally:
        asyncio.run(vmem_peer.close())

    merge_vectors(str(local_db), str(peer_db))

    con = sqlite3.connect(str(local_db))
    row = con.execute("SELECT valid_to FROM vector_memory WHERE text = 'shared fact'").fetchone()
    con.close()
    assert row[0] == 5000.0


def test_merge_vectors_transaction_rollback_on_error(tmp_path):
    """If an error occurs mid-merge, the transaction rolls back and local db is unchanged."""
    from taosmd.sync_merge import merge_vectors

    local_db = tmp_path / "local.db"
    peer_db = tmp_path / "peer.db"

    vmem_local = _make_store(str(local_db))
    try:
        asyncio.run(vmem_local.add("local fact", metadata={"agent": "test"}))
    finally:
        asyncio.run(vmem_local.close())

    vmem_peer = _make_store(str(peer_db))
    try:
        asyncio.run(vmem_peer.add("peer fact", metadata={"agent": "test"}))
    finally:
        asyncio.run(vmem_peer.close())

    # Corrupt the peer database to cause an error during merge
    con = sqlite3.connect(str(peer_db))
    con.execute("DROP TABLE vector_memory")
    con.commit()
    con.close()

    before_count = _row_count(str(local_db))
    before_max_id = _max_id(str(local_db))

    try:
        merge_vectors(str(local_db), str(peer_db))
        assert False, "Expected an error"
    except Exception:
        pass

    # Local db should be unchanged
    assert _row_count(str(local_db)) == before_count
    assert _max_id(str(local_db)) == before_max_id


def _set_valid_to(db_path: str, text: str, valid_to: float | None) -> None:
    """Helper to set valid_to on a specific row by text."""
    con = sqlite3.connect(db_path)
    try:
        if valid_to is None:
            con.execute("UPDATE vector_memory SET valid_to = NULL WHERE text = ?", (text,))
        else:
            con.execute("UPDATE vector_memory SET valid_to = ? WHERE text = ?", (valid_to, text))
        con.commit()
    finally:
        con.close()


def _get_valid_to(db_path: str, text: str) -> float | None:
    """Helper to get valid_to for a specific row by text."""
    con = sqlite3.connect(db_path)
    try:
        row = con.execute("SELECT valid_to FROM vector_memory WHERE text = ?", (text,)).fetchone()
        return row[0] if row else None
    finally:
        con.close()


def test_merge_vectors_valid_to_convergence_both_directions(tmp_path):
    """valid_to must converge: when both sides have valid_to set, keep the EARLIER (min);
    when only one side has it set, use that one (closure wins); when neither, None.
    Run merge both directions on swapped copies and assert identical valid_to."""
    from taosmd.sync_merge import merge_vectors

    base_a = tmp_path / "base_a.db"
    base_b = tmp_path / "base_b.db"

    # Case 1: both have valid_to set (100.0 vs 200.0) -> min(100, 200) = 100.0
    vmem_a = _make_store(str(base_a))
    try:
        asyncio.run(vmem_a.add("shared fact", metadata={"agent": "test"}))
    finally:
        asyncio.run(vmem_a.close())

    vmem_b = _make_store(str(base_b))
    try:
        asyncio.run(vmem_b.add("shared fact", metadata={"agent": "test"}))
    finally:
        asyncio.run(vmem_b.close())

    _set_valid_to(str(base_a), "shared fact", 100.0)
    _set_valid_to(str(base_b), "shared fact", 200.0)

    # Merge A <- B
    copy_a = tmp_path / "copy_a.db"
    copy_b = tmp_path / "copy_b.db"
    import shutil
    shutil.copy2(base_a, copy_a)
    shutil.copy2(base_b, copy_b)
    merge_vectors(str(copy_a), str(copy_b))
    valid_to_a = _get_valid_to(str(copy_a), "shared fact")

    # Merge B <- A
    copy_a2 = tmp_path / "copy_a2.db"
    copy_b2 = tmp_path / "copy_b2.db"
    shutil.copy2(base_a, copy_a2)
    shutil.copy2(base_b, copy_b2)
    merge_vectors(str(copy_b2), str(copy_a2))
    valid_to_b = _get_valid_to(str(copy_b2), "shared fact")

    assert valid_to_a == valid_to_b == 100.0, f"Expected 100.0, got A={valid_to_a}, B={valid_to_b}"

    # Case 2: both have valid_to set (200.0 vs 100.0) -> min = 100.0
    base_a2 = tmp_path / "base_a2.db"
    base_b2 = tmp_path / "base_b2.db"
    vmem_a2 = _make_store(str(base_a2))
    try:
        asyncio.run(vmem_a2.add("shared fact 2", metadata={"agent": "test"}))
    finally:
        asyncio.run(vmem_a2.close())
    vmem_b2 = _make_store(str(base_b2))
    try:
        asyncio.run(vmem_b2.add("shared fact 2", metadata={"agent": "test"}))
    finally:
        asyncio.run(vmem_b2.close())
    _set_valid_to(str(base_a2), "shared fact 2", 200.0)
    _set_valid_to(str(base_b2), "shared fact 2", 100.0)

    copy_a3 = tmp_path / "copy_a3.db"
    copy_b3 = tmp_path / "copy_b3.db"
    shutil.copy2(base_a2, copy_a3)
    shutil.copy2(base_b2, copy_b3)
    merge_vectors(str(copy_a3), str(copy_b3))
    valid_to_a2 = _get_valid_to(str(copy_a3), "shared fact 2")

    copy_a4 = tmp_path / "copy_a4.db"
    copy_b4 = tmp_path / "copy_b4.db"
    shutil.copy2(base_a2, copy_a4)
    shutil.copy2(base_b2, copy_b4)
    merge_vectors(str(copy_b4), str(copy_a4))
    valid_to_b2 = _get_valid_to(str(copy_b4), "shared fact 2")

    assert valid_to_a2 == valid_to_b2 == 100.0, f"Expected 100.0, got A={valid_to_a2}, B={valid_to_b2}"

    # Case 3: local has None, peer has 150.0 -> closure wins = 150.0
    base_a3 = tmp_path / "base_a3.db"
    base_b3 = tmp_path / "base_b3.db"
    vmem_a3 = _make_store(str(base_a3))
    try:
        asyncio.run(vmem_a3.add("shared fact 3", metadata={"agent": "test"}))
    finally:
        asyncio.run(vmem_a3.close())
    vmem_b3 = _make_store(str(base_b3))
    try:
        asyncio.run(vmem_b3.add("shared fact 3", metadata={"agent": "test"}))
    finally:
        asyncio.run(vmem_b3.close())
    _set_valid_to(str(base_a3), "shared fact 3", None)
    _set_valid_to(str(base_b3), "shared fact 3", 150.0)

    copy_a5 = tmp_path / "copy_a5.db"
    copy_b5 = tmp_path / "copy_b5.db"
    shutil.copy2(base_a3, copy_a5)
    shutil.copy2(base_b3, copy_b5)
    merge_vectors(str(copy_a5), str(copy_b5))
    valid_to_a3 = _get_valid_to(str(copy_a5), "shared fact 3")

    copy_a6 = tmp_path / "copy_a6.db"
    copy_b6 = tmp_path / "copy_b6.db"
    shutil.copy2(base_a3, copy_a6)
    shutil.copy2(base_b3, copy_b6)
    merge_vectors(str(copy_b6), str(copy_a6))
    valid_to_b3 = _get_valid_to(str(copy_b6), "shared fact 3")

    assert valid_to_a3 == valid_to_b3 == 150.0, f"Expected 150.0, got A={valid_to_a3}, B={valid_to_b3}"

    # Case 4: local has 150.0, peer has None -> closure wins = 150.0
    base_a4 = tmp_path / "base_a4.db"
    base_b4 = tmp_path / "base_b4.db"
    vmem_a4 = _make_store(str(base_a4))
    try:
        asyncio.run(vmem_a4.add("shared fact 4", metadata={"agent": "test"}))
    finally:
        asyncio.run(vmem_a4.close())
    vmem_b4 = _make_store(str(base_b4))
    try:
        asyncio.run(vmem_b4.add("shared fact 4", metadata={"agent": "test"}))
    finally:
        asyncio.run(vmem_b4.close())
    _set_valid_to(str(base_a4), "shared fact 4", 150.0)
    _set_valid_to(str(base_b4), "shared fact 4", None)

    copy_a7 = tmp_path / "copy_a7.db"
    copy_b7 = tmp_path / "copy_b7.db"
    shutil.copy2(base_a4, copy_a7)
    shutil.copy2(base_b4, copy_b7)
    merge_vectors(str(copy_a7), str(copy_b7))
    valid_to_a4 = _get_valid_to(str(copy_a7), "shared fact 4")

    copy_a8 = tmp_path / "copy_a8.db"
    copy_b8 = tmp_path / "copy_b8.db"
    shutil.copy2(base_a4, copy_a8)
    shutil.copy2(base_b4, copy_b8)
    merge_vectors(str(copy_b8), str(copy_a8))
    valid_to_b4 = _get_valid_to(str(copy_b8), "shared fact 4")

    assert valid_to_a4 == valid_to_b4 == 150.0, f"Expected 150.0, got A={valid_to_a4}, B={valid_to_b4}"


def _make_legacy_store(db_path: str) -> VectorMemory:
    """Create a store with rows but no store_meta 'mode' marker (legacy dense)."""
    vmem = _make_store(db_path)
    # Remove the mode marker to simulate a legacy store
    con = sqlite3.connect(db_path)
    try:
        con.execute("DELETE FROM store_meta WHERE key = 'mode'")
        con.commit()
    finally:
        con.close()
    return vmem


def test_merge_vectors_legacy_with_rows_vs_binary_quant_raises(tmp_path):
    """Legacy store WITH rows (no marker) vs binary_quant=1 peer raises StoreModeMismatch."""
    from taosmd.sync_merge import merge_vectors
    from taosmd.vector_memory import StoreModeMismatch

    local_db = tmp_path / "local.db"
    peer_db = tmp_path / "peer.db"

    # Local: legacy store WITH rows, no marker -> assumed dense (late_interaction=0, binary_quant=0)
    vmem_local = _make_legacy_store(str(local_db))
    try:
        asyncio.run(vmem_local.add("local fact", metadata={"agent": "test"}))
    finally:
        asyncio.run(vmem_local.close())

    # Peer: binary_quant = True (mode: late_interaction=0;binary_quant=1)
    vmem_peer = _make_store(str(peer_db), binary_quant=True)
    try:
        asyncio.run(vmem_peer.add("peer fact", metadata={"agent": "test"}))
    finally:
        asyncio.run(vmem_peer.close())

    try:
        merge_vectors(str(local_db), str(peer_db))
        assert False, "Expected StoreModeMismatch"
    except StoreModeMismatch:
        pass


def test_merge_vectors_binary_quant_vs_legacy_with_rows_raises(tmp_path):
    """binary_quant=1 local vs legacy store WITH rows (no marker) peer raises StoreModeMismatch."""
    from taosmd.sync_merge import merge_vectors
    from taosmd.vector_memory import StoreModeMismatch

    local_db = tmp_path / "local.db"
    peer_db = tmp_path / "peer.db"

    # Local: binary_quant = True
    vmem_local = _make_store(str(local_db), binary_quant=True)
    try:
        asyncio.run(vmem_local.add("local fact", metadata={"agent": "test"}))
    finally:
        asyncio.run(vmem_local.close())

    # Peer: legacy store WITH rows, no marker
    vmem_peer = _make_legacy_store(str(peer_db))
    try:
        asyncio.run(vmem_peer.add("peer fact", metadata={"agent": "test"}))
    finally:
        asyncio.run(vmem_peer.close())

    try:
        merge_vectors(str(local_db), str(peer_db))
        assert False, "Expected StoreModeMismatch"
    except StoreModeMismatch:
        pass


def test_merge_vectors_empty_no_marker_local_vs_any_peer_merges(tmp_path):
    """Empty no-marker local vs any peer merges (no rows = no constraint)."""
    from taosmd.sync_merge import merge_vectors

    local_db = tmp_path / "local.db"
    peer_db = tmp_path / "peer.db"

    # Local: empty store, no marker (fresh store, no rows)
    vmem_local = _make_store(str(local_db))
    # Remove mode marker but keep it empty
    con = sqlite3.connect(str(local_db))
    try:
        con.execute("DELETE FROM store_meta WHERE key = 'mode'")
        con.commit()
    finally:
        con.close()
    asyncio.run(vmem_local.close())

    # Peer: binary_quant = True
    vmem_peer = _make_store(str(peer_db), binary_quant=True)
    try:
        asyncio.run(vmem_peer.add("peer fact", metadata={"agent": "test"}))
    finally:
        asyncio.run(vmem_peer.close())

    # Should merge without error
    result = merge_vectors(str(local_db), str(peer_db))
    assert result["added"] == 1


def test_merge_vectors_missing_store_meta_table_treated_as_no_marker(tmp_path):
    """Missing store_meta table (older store) treated same as missing marker."""
    from taosmd.sync_merge import merge_vectors

    local_db = tmp_path / "local.db"
    peer_db = tmp_path / "peer.db"

    # Local: store WITHOUT store_meta table at all (simulate older schema)
    vmem_local = _make_store(str(local_db))
    try:
        asyncio.run(vmem_local.add("local fact", metadata={"agent": "test"}))
    finally:
        asyncio.run(vmem_local.close())
    # Drop store_meta table
    con = sqlite3.connect(str(local_db))
    try:
        con.execute("DROP TABLE store_meta")
        con.commit()
    finally:
        con.close()

    # Peer: normal store with marker
    vmem_peer = _make_store(str(peer_db))
    try:
        asyncio.run(vmem_peer.add("peer fact", metadata={"agent": "test"}))
    finally:
        asyncio.run(vmem_peer.close())

    # Should merge without error (missing table treated as no marker, and local has rows -> legacy dense)
    result = merge_vectors(str(local_db), str(peer_db))
    assert result["added"] == 1
    assert result["deduped"] == 0