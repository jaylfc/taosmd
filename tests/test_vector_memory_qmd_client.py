"""Tests for qmd embed mode owning its own httpx client.

In embed_mode="qmd" no vector was ever stored: VectorMemory.init() was called
with no http_client and _embed_qmd returns [] when self._http is None, so
add() failed for every chunk and search fell back to FTS only. init() now
creates (and owns) its own httpx.AsyncClient when none is passed in qmd mode.
"""

from __future__ import annotations

import asyncio

from taosmd.vector_memory import VectorMemory


def test_qmd_init_creates_owned_client(tmp_path):
    async def go():
        vm = VectorMemory(db_path=str(tmp_path / "v.db"), embed_mode="qmd")
        await vm.init()
        assert vm._http is not None
        await vm.close()
        assert vm._http is None

    asyncio.run(go())


def test_qmd_add_stores_vector_without_explicit_client(tmp_path):
    async def go():
        vm = VectorMemory(db_path=str(tmp_path / "v.db"), embed_mode="qmd")
        await vm.init()

        class _Resp:
            status_code = 200

            def json(self):
                return {"embedding": [0.1] * 384}

        async def fake_post(*args, **kwargs):
            return _Resp()

        vm._http.post = fake_post  # type: ignore[assignment]
        rid = await vm.add("hello world")
        await vm.close()
        return rid

    rid = asyncio.run(go())
    assert rid >= 0


def test_caller_client_not_closed(tmp_path):
    async def go():
        closed = {"flag": False}

        class FakeClient:
            async def aclose(self):
                closed["flag"] = True

        client = FakeClient()
        vm = VectorMemory(db_path=str(tmp_path / "v.db"), embed_mode="qmd")
        await vm.init(http_client=client)
        await vm.close()
        assert closed["flag"] is False

    asyncio.run(go())


def test_non_qmd_mode_has_no_client(tmp_path):
    async def go():
        vm = VectorMemory(db_path=str(tmp_path / "v.db"), embed_mode="onnx")
        await vm.init()
        assert vm._http is None
        await vm.close()

    asyncio.run(go())