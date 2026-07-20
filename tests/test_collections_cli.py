"""CLI tests for ``taosmd collections`` subcommands.

Mirrors the shelves/projects CLI conventions: thin argparse layer over the
service wrappers, human-readable one-line-per-row output, exit code 2 with
an error on stderr for validation failures.
"""
from __future__ import annotations

import pytest

from taosmd import config
from taosmd.cli import main
from taosmd.collections import CollectionStore


@pytest.fixture
def source_dir(tmp_path):
    d = tmp_path / "src"
    d.mkdir()
    (d / "readme.md").write_text("# Hello\n\nDocs body.")
    return d


@pytest.fixture
def data_dir(tmp_path, monkeypatch, source_dir):
    d = tmp_path / "taosmd-data"
    d.mkdir()
    monkeypatch.setenv("TAOSMD_DATA_DIR", str(d))
    monkeypatch.delenv("TAOSMD_COLLECTIONS_ALLOWED_ROOTS", raising=False)
    from taosmd import api as taosmd_api
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    config.set_collections_allowed_roots([str(source_dir)], data_dir=str(d))
    return str(d)


def test_create_and_list(data_dir, source_dir, capsys):
    rc = main([
        "collections", "create", "--name", "repo docs", "--kind", "docs",
        "--source", str(source_dir),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "col-" in out

    rc = main(["collections", "list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "repo docs" in out
    assert "created" in out


def test_create_outside_roots_fails(data_dir, tmp_path, capsys):
    outside = tmp_path / "outside"
    outside.mkdir()
    rc = main([
        "collections", "create", "--name", "x", "--kind", "docs",
        "--source", str(outside),
    ])
    assert rc == 2
    assert "allowed root" in capsys.readouterr().err


def test_link_unlink_grant_revoke(data_dir, source_dir, capsys):
    store = CollectionStore(data_dir)
    col = store.create(name="a", kind="docs", source_path=str(source_dir))
    cid = col["id"]

    assert main(["collections", "link", cid, "--type", "taos", "--id", "prj-1"]) == 0
    assert "prj-1" in store.get(cid)["links"][0]["id"]
    assert main(["collections", "unlink", cid, "--type", "taos", "--id", "prj-1"]) == 0
    assert store.get(cid)["links"] == []

    assert main(["collections", "grant", cid, "dev"]) == 0
    assert store.get(cid)["grants"] == ["dev"]
    assert main(["collections", "revoke", cid, "dev"]) == 0
    assert store.get(cid)["grants"] == []
    capsys.readouterr()


def test_index_runs_synchronously(data_dir, source_dir, capsys, monkeypatch):
    from taosmd import api as taosmd_api
    import asyncio

    stores = asyncio.run(taosmd_api._ensure_stores(data_dir))

    async def _fake_embed(text, task="search_document"):
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    stores["vector"].embed = _fake_embed

    store = CollectionStore(data_dir)
    col = store.create(name="a", kind="docs", source_path=str(source_dir))
    rc = main(["collections", "index", col["id"]])
    assert rc == 0
    out = capsys.readouterr().out
    assert "files_indexed=1" in out
    assert store.get(col["id"])["status"] == "ready"


def test_unknown_collection_exits_2(data_dir, capsys):
    rc = main(["collections", "grant", "col-000000000000", "dev"])
    assert rc == 2
    assert "not found" in capsys.readouterr().err
