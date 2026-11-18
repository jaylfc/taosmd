"""Tests for taosmd.mcp_server — the MCP activation surface (#84).

Skips cleanly when the optional ``mcp`` SDK is not installed. Offline + fast:
the server is built against an isolated tmp data dir and the vector embedder is
patched (same deterministic hash vector as tests/test_http_server.py and
tests/test_api.py) so no ONNX/QMD model is needed.

We don't spin up a full MCP client; instead we exercise the underlying tool
callables that FastMCP registers, which is where all the service-layer routing
lives.
"""

from __future__ import annotations

import asyncio
import json
import sys

import pytest

pytest.importorskip("mcp")

from taosmd import api as taosmd_api
from taosmd import mcp_server


def _patch_embedder(stores: dict) -> None:
    """Deterministic 8-dim hash embedder so search finds matching text."""
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


@pytest.fixture
def server(tmp_path, monkeypatch):
    """Build an MCP server against an isolated data dir with a patched embedder.

    Yields the configured FastMCP instance. Initialises the stores on the
    server's service-loop thread (so the thread-affine SQLite connections live
    where the tools will use them), then tears everything down cleanly.
    """
    data_dir = tmp_path / "taosmd-data"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})

    mcp = mcp_server.build_server(data_dir=str(data_dir))
    loop = mcp._taosmd_service_loop

    stores = loop.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)

    try:
        yield mcp
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
    """Invoke a registered tool by name and return its decoded JSON result.

    FastMCP returns either a list of content blocks (for dict results) or a
    ``(content, {"result": ...})`` tuple (for list results) depending on the
    return type. Prefer the structured tuple payload when present (clean list);
    otherwise decode the JSON out of the single text content block (dict).
    """
    result = await mcp.call_tool(name, args)
    if isinstance(result, tuple):
        structured = result[1]
        if isinstance(structured, dict) and "result" in structured:
            return structured["result"]
        return structured
    if not result:
        return []
    return json.loads(result[0].text)


def test_import_taosmd_without_touching_mcp():
    """Importing the package works (mcp imported lazily, never at import time)."""
    import taosmd  # noqa: PLC0415

    assert hasattr(taosmd, "mcp_server")


def test_registers_expected_tools(server):
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert {
        "memory_ingest",
        "memory_search",
        "memory_pending_list",
        "memory_pending_resolve",
        "memory_stats",
    } <= names


def test_ingest_then_search_roundtrip(server):
    out = asyncio.run(_call(
        server,
        "memory_ingest",
        {"text": "The MCP server ships on the feat/mcp-server branch.", "agent": "mcp-test"},
    ))
    assert out["archived"] == 1
    assert out["agent"] == "mcp-test"

    hits = asyncio.run(_call(
        server,
        "memory_search",
        {"query": "The MCP server ships on the feat/mcp-server branch.", "agent": "mcp-test"},
    ))
    assert hits, "expected the ingested text to be retrievable"
    assert "MCP server" in hits[0]["text"]


def test_stats_reports_count(server):
    asyncio.run(_call(
        server,
        "memory_ingest",
        {"text": "Counting chunks for the stats tool.", "agent": "mcp-stats"},
    ))
    out = asyncio.run(_call(server, "memory_stats", {"agent": "mcp-stats"}))
    assert out["agent"] == "mcp-stats"
    assert out["registered"] is True
    # ingest registers the agent and stamps last_ingest_at.
    assert out["last_ingest_at"] >= 1


def test_pending_list_empty(server):
    out = asyncio.run(_call(server, "memory_pending_list", {"agent": "mcp-test"}))
    assert out == []


def test_a2a_send_with_refs_and_blocks(server):
    """a2a_send with refs+blocks -> a2a_read returns them verbatim (taOSmd #211)."""
    refs = [
        {"kind": "doc", "title": "Spec", "uri": "https://example.com/spec",
         "sha256": "deadbeef", "doc_id": "spec-1", "version": 2, "for": ["agentA"],
         "summary": "API spec"},
        {"kind": "log", "title": "CI Log", "uri": "https://example.com/log",
         "sha256": None, "doc_id": None, "version": None, "for": None,
         "summary": None},
    ]
    blocks = [
        {"kind": "tool_call", "tool": "search", "args": {"query": "hello"}},
        {"kind": "tool_result", "result": "found 3 items"},
    ]
    receipt = asyncio.run(_call(
        server,
        "a2a_send",
        {"channel": "mcp-env", "sender": "mcp-agent", "body": "mcp envelope msg",
         "refs": refs, "blocks": blocks},
    ))
    assert receipt["refs"] == refs
    assert receipt["blocks"] == blocks

    msgs = asyncio.run(_call(
        server,
        "a2a_read",
        {"channel": "mcp-env"},
    ))
    assert len(msgs) == 1
    assert msgs[0]["refs"] == refs
    assert msgs[0]["blocks"] == blocks


class TestMCPDependencyErrors:
    """The optional ``mcp`` SDK is imported lazily; when it is absent or the
    wrong major version, the server must surface a clear, one-line error
    instead of a traceback (tsk-od3z2g)."""

    @staticmethod
    def _break_fastmcp_import(monkeypatch):
        """Make ``from mcp.server.fastmcp import FastMCP`` raise ImportError.

        This is how mcp 2.x actually fails: the class was renamed and moved,
        so the import line raises ModuleNotFoundError. Setting the module to
        ``None`` in ``sys.modules`` reproduces that exactly.
        """
        monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", None)

    def test_incompatible_mcp_version_is_reported(self, tmp_path, monkeypatch):
        from taosmd.mcp_server import IncompatibleMCPVersionError

        self._break_fastmcp_import(monkeypatch)
        import importlib.metadata as md

        monkeypatch.setattr(md, "version", lambda name: "2.3.0")

        with pytest.raises(IncompatibleMCPVersionError) as exc_info:
            mcp_server.build_server(data_dir=str(tmp_path / "data"))

        msg = str(exc_info.value)
        assert "2.3.0" in msg
        assert "mcp>=1,<2" in msg
        assert "pip install taosmd[mcp]" not in msg

    def test_missing_mcp_is_still_reported_as_missing(self, tmp_path, monkeypatch):
        from taosmd.mcp_server import MissingMCPDependencyError

        self._break_fastmcp_import(monkeypatch)
        import importlib.metadata as md

        def _missing(name):
            raise md.PackageNotFoundError(name)

        monkeypatch.setattr(md, "version", _missing)

        with pytest.raises(MissingMCPDependencyError) as exc_info:
            mcp_server.build_server(data_dir=str(tmp_path / "data"))

        assert "pip install taosmd[mcp]" in str(exc_info.value)

    def test_cli_prints_one_line_for_incompatible_version(
        self, tmp_path, monkeypatch, capsys
    ):
        from taosmd.cli import main

        self._break_fastmcp_import(monkeypatch)
        import importlib.metadata as md

        monkeypatch.setattr(md, "version", lambda name: "2.3.0")

        rc = main(["mcp", "--mcp-data-dir", str(tmp_path / "data")])

        assert rc != 0
        out = capsys.readouterr()
        combined = (out.err or "") + (out.out or "")
        assert "mcp>=1,<2" in combined
        assert "Traceback" not in combined


def test_mcp_extra_is_pinned_below_2():
    """The ``mcp`` extra must pin below 2.x so a fresh install gets a working
    FastMCP (tsk-od3z2g)."""
    import tomllib

    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    assert data["project"]["optional-dependencies"]["mcp"] == ["mcp>=1,<2"]
