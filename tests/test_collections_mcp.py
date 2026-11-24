"""MCP surface tests for collections: memory_list_collections + the
``collection`` parameter on memory_search. Skips when the optional ``mcp``
SDK is absent, matching tests/test_mcp_server.py."""
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("mcp")

from taosmd import api as taosmd_api
from taosmd import config as taosmd_config
from taosmd import mcp_server
from taosmd.collections import CollectionStore, ingest_folder


def _patch_embedder(stores: dict) -> None:
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


@pytest.fixture
def env(tmp_path, monkeypatch):
    source = tmp_path / "src"
    source.mkdir()
    (source / "readme.md").write_text("# Widget\n\nThe widget frobnicates the sprocket.")
    data_dir = tmp_path / "taosmd-data"
    data_dir.mkdir()
    monkeypatch.setenv("TAOSMD_DATA_DIR", str(data_dir))
    monkeypatch.delenv("TAOSMD_COLLECTIONS_ALLOWED_ROOTS", raising=False)
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    taosmd_config.set_collections_allowed_roots([str(source)], data_dir=str(data_dir))

    mcp = mcp_server.build_server(data_dir=str(data_dir))
    loop = mcp._taosmd_service_loop
    stores = loop.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)
    try:
        yield mcp, loop, str(data_dir), source
    finally:
        for s in list(taosmd_api._stores_cache.values()):
            for store in (s.get("archive"), s.get("vector"), s.get("kg")):
                if store and hasattr(store, "close"):
                    try:
                        loop.run(store.close())
                    except Exception:
                        pass
        loop.close()


async def _call(mcp, name: str, args: dict):
    result = await mcp.call_tool(name, args)
    if isinstance(result, tuple):
        structured = result[1]
        if isinstance(structured, dict) and "result" in structured:
            return structured["result"]
    import json
    return json.loads(result[0].text)


def test_tools_registered(env):
    mcp, *_ = env
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert "memory_list_collections" in names


def test_list_collections_and_scoped_search(env):
    mcp, loop, data_dir, source = env
    store = CollectionStore(data_dir)
    col = store.create(name="docs", kind="docs", source_path=str(source))
    store.grant(col["id"], "dev")
    loop.run(ingest_folder(col["id"], data_dir=data_dir))

    cols = asyncio.run(_call(mcp, "memory_list_collections", {}))
    assert [c["id"] for c in cols] == [col["id"]]
    assert cols[0]["status"] == "ready"

    hits = asyncio.run(_call(mcp, "memory_search", {
        "query": "widget frobnicates", "agent": "dev", "collection": col["id"],
    }))
    assert any(h["metadata"].get("file_path") == "readme.md" for h in hits)

    # No grant, no hits from the collection.
    hits = asyncio.run(_call(mcp, "memory_search", {
        "query": "widget frobnicates", "agent": "stranger", "collection": col["id"],
    }))
    assert not any(h["metadata"].get("collection_id") for h in hits)
