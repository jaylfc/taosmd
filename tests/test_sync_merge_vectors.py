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