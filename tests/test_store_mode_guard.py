"""Storage-format guard: a vector store must not be reopened in a different
mode (dense vs late-interaction vs binary-quant) than it was built in.

The formats are mutually incompatible on disk, so a mismatch must fail loud
with a re-embed instruction rather than silently serve wrong-mode results.
"""

import asyncio

import pytest

from taosmd.vector_memory import StoreModeMismatch, VectorMemory


def _build(path, **kwargs):
    vm = VectorMemory(db_path=str(path), embed_mode="qmd", **kwargs)
    asyncio.run(vm.init())
    return vm


def test_fresh_store_records_mode(tmp_path):
    vm = _build(tmp_path / "vm.db")
    row = vm._conn.execute("SELECT value FROM store_meta WHERE key='mode'").fetchone()
    assert row["value"] == "late_interaction=0;binary_quant=0"
    asyncio.run(vm.close())


def test_reopen_same_mode_ok(tmp_path):
    vm = _build(tmp_path / "vm.db")
    asyncio.run(vm.close())
    vm2 = _build(tmp_path / "vm.db")  # same dense mode
    assert vm2._conn is not None
    asyncio.run(vm2.close())


def test_reopen_late_interaction_on_dense_store_raises(tmp_path):
    vm = _build(tmp_path / "vm.db")  # dense
    asyncio.run(vm.close())
    with pytest.raises(StoreModeMismatch, match="incompatible|assumed dense"):
        _build(tmp_path / "vm.db", late_interaction=True,
               colbert_model="answerdotai/answerai-colbert-small-v1")


def test_reopen_binary_quant_on_dense_store_raises(tmp_path):
    vm = _build(tmp_path / "vm.db")
    asyncio.run(vm.close())
    with pytest.raises(StoreModeMismatch):
        _build(tmp_path / "vm.db", binary_quant=True)


def test_legacy_store_with_rows_no_marker_refuses_nondense(tmp_path):
    # Simulate a legacy store: dense rows, then drop the marker.
    vm = _build(tmp_path / "vm.db")
    vm._conn.execute(
        "INSERT INTO vector_memory (text, embedding, created_at) VALUES ('x', '[]', 1.0)"
    )
    vm._conn.execute("DELETE FROM store_meta WHERE key='mode'")
    vm._conn.commit()
    asyncio.run(vm.close())
    # Reopening in a non-dense mode is refused straight away (legacy assumed dense).
    with pytest.raises(StoreModeMismatch, match="assumed dense"):
        _build(tmp_path / "vm.db", binary_quant=True)
    # Reopening dense is fine (legacy assumed dense, matches).
    vm2 = _build(tmp_path / "vm.db")
    asyncio.run(vm2.close())
