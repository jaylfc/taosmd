"""Tests for taosmd.chain_of_memory — fragment utilization, fail-safe behaviour."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from taosmd.chain_of_memory import organize_fragments, fragments_from_context


def _run(coro):
    return asyncio.run(coro)


class _FakeResp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._p


class _FakeClient:
    """Returns a canned {ordered_fragments: [...]} JSON via /api/chat shape."""
    def __init__(self, ordered):
        self._content = json.dumps({"ordered_fragments": ordered})

    async def post(self, url, json=None, timeout=None):
        return _FakeResp({"message": {"content": self._content}})


class _BrokenClient:
    async def post(self, url, json=None, timeout=None):
        raise httpx.HTTPError("boom")


class _BadJsonClient:
    async def post(self, url, json=None, timeout=None):
        return _FakeResp({"message": {"content": "not json"}})


FRAGS = [
    "[1 Jan] Alice joined Stripe as an engineer.",
    "[2 Jan] Alice mentioned she likes coffee.",
    "[3 Jan] Alice was promoted to staff engineer at Stripe.",
]


def test_fragments_from_context_splits_lines():
    ctx = "[a] one\n[b] two\n\n[c] three\n"
    assert fragments_from_context(ctx) == ["[a] one", "[b] two", "[c] three"]


def test_organize_selects_and_orders_verbatim():
    # Model keeps fragments 0 and 2, drops the coffee noise, reorders.
    client = _FakeClient([FRAGS[0], FRAGS[2]])
    out = _run(organize_fragments("When was Alice promoted?", FRAGS,
                                  model="m", ollama_url="http://x", http_client=client))
    assert out == [FRAGS[0], FRAGS[2]]


def test_organize_rejects_invented_fragments():
    # Model returns a fragment that was never in the input -> dropped.
    client = _FakeClient(["[9 Jan] Alice quit and moved to Mars."])
    out = _run(organize_fragments("q", FRAGS,
                                  model="m", ollama_url="http://x", http_client=client))
    assert out == []


def test_organize_matches_normalised_whitespace():
    # Model reformats whitespace but content is identical -> maps back to real.
    reformatted = "  ".join(FRAGS[0].split())  # collapse to single-spaced w/ extra
    client = _FakeClient([reformatted])
    out = _run(organize_fragments("q", FRAGS,
                                  model="m", ollama_url="http://x", http_client=client))
    assert out == [FRAGS[0]]


def test_organize_accepts_prefix_trim():
    # Model trimmed a trailing clause; unique prefix match accepts it.
    trimmed = "[1 Jan] Alice joined Stripe"
    client = _FakeClient([trimmed])
    out = _run(organize_fragments("q", FRAGS,
                                  model="m", ollama_url="http://x", http_client=client))
    assert out == [FRAGS[0]]


def test_organize_dedupes():
    client = _FakeClient([FRAGS[0], FRAGS[0]])
    out = _run(organize_fragments("q", FRAGS,
                                  model="m", ollama_url="http://x", http_client=client))
    assert out == [FRAGS[0]]


def test_organize_empty_fragments_returns_empty():
    out = _run(organize_fragments("q", [],
                                  model="m", ollama_url="http://x", http_client=_FakeClient([])))
    assert out == []


def test_organize_transport_error_returns_none():
    # None signals failure so the caller falls back to the raw context.
    out = _run(organize_fragments("q", FRAGS,
                                  model="m", ollama_url="http://x", http_client=_BrokenClient()))
    assert out is None


def test_organize_bad_json_returns_none():
    out = _run(organize_fragments("q", FRAGS,
                                  model="m", ollama_url="http://x", http_client=_BadJsonClient()))
    assert out is None


def test_organize_non_list_payload_returns_none():
    class _C:
        async def post(self, url, json=None, timeout=None):
            return _FakeResp({"message": {"content": json_dumps_nonlist()}})

    def json_dumps_nonlist():
        return json.dumps({"ordered_fragments": "not a list"})

    out = _run(organize_fragments("q", FRAGS,
                                  model="m", ollama_url="http://x", http_client=_C()))
    assert out is None
