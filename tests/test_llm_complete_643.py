"""Tests for taosmd.llm_complete — the OpenAI-compat router (#643).

Proves the librarian stages work against an OpenAI-compat server (the
llama.cpp case from #643) while keeping the native Ollama dialect.
"""
from __future__ import annotations

import asyncio

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
            body = request.json()
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
