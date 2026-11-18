"""Tests for taosmd.llm_complete: the OpenAI-compat router (#643).

Proves the librarian stages work against an OpenAI-compat server (the
llama.cpp case from #643) while keeping the native Ollama dialect.
"""
from __future__ import annotations

import asyncio
import json

import httpx

from taosmd.llm_complete import llm_complete, set_api_family

OLLAMA_BASE = "http://ollama.test"
OPENAI_BASE = "http://llamacpp.test"


def _transport():
    calls = {"ollama_generate": 0, "openai_chat": 0, "version": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/version":
            calls["version"] += 1
            return httpx.Response(200, json={"version": "0.1"})
        if path == "/api/generate":
            calls["ollama_generate"] += 1
            return httpx.Response(200, json={"response": "OLLAMA-TEXT"})
        if path == "/v1/chat/completions":
            calls["openai_chat"] += 1
            body = json.loads(request.content) if request.content else {}
            assert body["messages"][0]["content"], "prompt must be carried"
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "OPENAI-TEXT"}}]})
        if path == "/generate-404":  # the #643 failure shape
            return httpx.Response(404)
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), calls


def test_native_ollama_dialect():
    async def run():
        client, calls = _transport()
        set_api_family(OLLAMA_BASE, "ollama")
        text = await llm_complete(client, OLLAMA_BASE, "m", "p", api="ollama")
        assert text == "OLLAMA-TEXT"
        assert calls["ollama_generate"] == 1 and calls["openai_chat"] == 0
    asyncio.run(run())


def test_openai_dialect_used_when_configured():
    calls = {"ollama_generate": 0, "openai_chat": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            calls["openai_chat"] += 1
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "OPENAI-TEXT"}}]})
        calls["ollama_generate"] += 1
        return httpx.Response(404)

    async def run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        text = await llm_complete(client, OPENAI_BASE, "m", "p", api="openai")
        await client.aclose()
        assert text == "OPENAI-TEXT"
        assert calls["openai_chat"] == 1 and calls["ollama_generate"] == 0
    asyncio.run(run())


def test_probe_falls_back_to_openai_when_no_ollama_version():
    """A llama.cpp server 404s /api/version; the router must switch to /v1
    instead of 404ing the completion like the pre-fix code (#643)."""
    async def run():
        client, calls = _transport()

        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/version":
                return httpx.Response(404)
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": []})
            if request.url.path == "/v1/chat/completions":
                return httpx.Response(200, json={
                    "choices": [{"message": {"content": "PROBED-OPENAI"}}]})
            return httpx.Response(404)

        c2 = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        text = await llm_complete(c2, "http://fresh-llamacpp.test", "m", "p")
        assert text == "PROBED-OPENAI"
    asyncio.run(run())


def test_reasoning_null_content_falls_back_to_reasoning_field():
    async def run():
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "choices": [{"message": {"content": None, "reasoning": "THOUGHT-TEXT"}}]})
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        text = await llm_complete(client, "http://r.test", "m", "p", api="openai")
        assert text == "THOUGHT-TEXT"
    asyncio.run(run())


def test_ollama_no_think_sends_think_false():
    async def run():
        async def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content) if request.content else {}
            assert body.get("think") is False, f"expected think=False, got {body}"
            return httpx.Response(200, json={"response": "OLLAMA-TEXT"})

        c = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        text = await llm_complete(c, OLLAMA_BASE, "m", "p", api="ollama", no_think=True)
        await c.aclose()
        assert text == "OLLAMA-TEXT"
    asyncio.run(run())


def test_ollama_no_think_no_think_key_when_false():
    async def run():
        async def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content) if request.content else {}
            assert "think" not in body, f"expected no 'think' key, got {body}"
            return httpx.Response(200, json={"response": "OLLAMA-TEXT"})

        c = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        text = await llm_complete(c, OLLAMA_BASE, "m", "p", api="ollama", no_think=False)
        await c.aclose()
        assert text == "OLLAMA-TEXT"
    asyncio.run(run())


def test_ollama_non_200_raises():
    async def run():
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"response": "error"})

        c = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            await llm_complete(c, OLLAMA_BASE, "m", "p", api="ollama")
            await c.aclose()
            assert False, "expected HTTPStatusError"
        except httpx.HTTPStatusError:
            pass
    asyncio.run(run())


def test_openai_non_200_raises():
    async def run():
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "server error"})

        c = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            await llm_complete(c, OPENAI_BASE, "m", "p", api="openai")
            await c.aclose()
            assert False, "expected HTTPStatusError"
        except httpx.HTTPStatusError:
            pass
    asyncio.run(run())


def test_ollama_thinking_fallback():
    async def run():
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": "", "thinking": "abc"})

        c = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        text = await llm_complete(c, OLLAMA_BASE, "m", "p", api="ollama")
        await c.aclose()
        assert text == "abc"
    asyncio.run(run())


def test_ollama_temperature_max_tokens_timeout_passed_through():
    """Ollama dialect receives temperature, max_tokens (as num_predict), and timeout."""
    async def run():
        captured = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content) if request.content else {}
            captured["body"] = body
            # timeout is passed as **kw to client.post; httpx stores it in
            # request.extensions["timeout"] as a dict when a float is given
            timeout_obj = request.extensions.get("timeout")
            captured["timeout"] = timeout_obj
            return httpx.Response(200, json={"response": "OK"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await llm_complete(
            client, OLLAMA_BASE, "m", "p",
            api="ollama", temperature=0.7, max_tokens=123, timeout=45.0,
        )
        await client.aclose()

        assert captured["body"]["options"]["temperature"] == 0.7
        assert captured["body"]["options"]["num_predict"] == 123
        # timeout is passed via httpx request extensions as a dict
        assert captured["timeout"] is not None
        assert captured["timeout"]["connect"] == 45.0
    asyncio.run(run())


def test_openai_temperature_max_tokens_timeout_passed_through():
    """OpenAI dialect receives temperature, max_tokens, and timeout."""
    async def run():
        captured = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content) if request.content else {}
            captured["body"] = body
            timeout_obj = request.extensions.get("timeout")
            captured["timeout"] = timeout_obj
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "OK"}}]})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await llm_complete(
            client, OPENAI_BASE, "m", "p",
            api="openai", temperature=0.5, max_tokens=456, timeout=30.0,
        )
        await client.aclose()

        assert captured["body"]["temperature"] == 0.5
        assert captured["body"]["max_tokens"] == 456
        assert captured["timeout"] is not None
        assert captured["timeout"]["connect"] == 30.0
    asyncio.run(run())


def test_crystallize_fallback_on_router_non_200(tmp_path):
    """CrystalStore._llm_crystallize returns its own fallback when llm_complete raises on non-200."""
    import asyncio
    from taosmd.crystallize import CrystalStore

    async def run():
        cs = CrystalStore(db_path=tmp_path / "crystals.db")
        await cs.init()
        try:
            # Stub llm_complete to raise HTTPStatusError (non-200)
            import taosmd.llm_complete as lc
            original = lc.llm_complete

            async def raising_llm_complete(*args, **kwargs):
                raise httpx.HTTPStatusError(
                    "Server error", request=httpx.Request("POST", "http://test"),
                    response=httpx.Response(500, json={"error": "boom"})
                )

            lc.llm_complete = raising_llm_complete
            try:
                narrative, outcomes, lessons, files = await cs._llm_crystallize(
                    "test conversation", "http://fake", "model"
                )
                # Should return fallback, not raise
                assert narrative.startswith("test conversation")
                assert outcomes == []
                assert lessons == []
                assert files == []
            finally:
                lc.llm_complete = original
        finally:
            await cs.close()


def test_ollama_thinking_fallback_disabled():
    async def run():
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": "", "thinking": "long reasoning text"})

        c = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        text = await llm_complete(c, OLLAMA_BASE, "m", "p", api="ollama", thinking_fallback=False)
        await c.aclose()
        assert text == ""
    asyncio.run(run())


def test_openai_reasoning_fallback_disabled():
    async def run():
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "choices": [{"message": {"content": None, "reasoning": "long reasoning text"}}]})
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        text = await llm_complete(client, "http://r.test", "m", "p", api="openai", thinking_fallback=False)
        await client.aclose()
        assert text == ""
    asyncio.run(run())
