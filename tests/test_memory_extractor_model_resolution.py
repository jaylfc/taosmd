"""Model resolution at the LLM boundary of process_conversation_turn.

Two properties are pinned here:

1. When extraction_model is the "default" sentinel and no explicit pin exists,
   the function must call resolve_memory_model() (which consults the
   generator-profile registry) rather than get_memory_model() (which would
   return None on a fresh install and silently bypass the profile system).

2. The resolved profile value is a "provider:model" string (e.g.
   "ollama:qwen3.5:9b"). What reaches the HTTP payload must be the BARE model
   name ("qwen3.5:9b"): live Ollama rejects the prefixed form, and the
   extractor then silently degrades to regex extraction. An earlier revision
   of this test asserted the prefixed string reached the payload, which
   encoded that bug; it is inverted now.
"""
from __future__ import annotations

import asyncio
import logging

import pytest

from taosmd import generator_profiles as gp
from taosmd.knowledge_graph import TemporalKnowledgeGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    return asyncio.run(coro)


class _Captured(Exception):
    """Sentinel raised by the spy to short-circuit after capturing the model arg."""


def _make_spy(holder: list):
    """Return an async spy for extract_facts_with_llm that records model and raises."""
    async def _spy(text, llm_url, http_client, *, agent_name="default", model="default"):
        holder.append(model)
        raise _Captured(model)
    return _spy


class _FakeResponse:
    status_code = 200

    def json(self):
        return {
            "choices": [{"message": {"content": (
                '[{"subject": "alice", "predicate": "likes", "object": "tea"}]'
            )}}]
        }


class _PayloadCapturingClient:
    """Fake http client that records the JSON body of each POST."""

    def __init__(self, holder: list, response=None):
        self._holder = holder
        self._response = response or _FakeResponse()

    async def post(self, url, json=None, timeout=None):
        self._holder.append(json)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _pin_tier(monkeypatch, tier: str):
    monkeypatch.setattr(gp.recipes, "local_probe", lambda: {"host": {}})
    monkeypatch.setattr(gp.recipes, "tier_of", lambda info: tier)


def _clear_pins(monkeypatch):
    monkeypatch.setattr("taosmd.config.get_memory_model", lambda data_dir=None: None)
    monkeypatch.setattr("taosmd.config.get_generator_profile", lambda data_dir=None: None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_covered_tier_sends_bare_model_to_http_payload(monkeypatch, tmp_path):
    """Covered tier: the HTTP payload carries the bare model, never the provider prefix.

    The balanced profile stores "ollama:qwen3.5:9b" for gpu-12gb. The payload
    sent to /v1/chat/completions must be "qwen3.5:9b"; the prefixed form is
    rejected by live Ollama and silently degrades extraction to regex.
    """
    _pin_tier(monkeypatch, "gpu-12gb")
    _clear_pins(monkeypatch)

    payloads: list[dict] = []

    async def go():
        from taosmd.memory_extractor import process_conversation_turn

        kg = TemporalKnowledgeGraph(db_path=tmp_path / "kg.db")
        await kg.init()
        try:
            result = await process_conversation_turn(
                text="Alice likes tea.",
                agent_name=None,
                kg=kg,
                llm_url="http://localhost:11434",
                http_client=_PayloadCapturingClient(payloads),
                use_llm=True,
                extraction_model="default",
            )
        finally:
            await kg.close()
        return result

    result = _run(go())

    assert len(payloads) == 1, "expected exactly one LLM HTTP call"
    model = payloads[0]["model"]
    assert model != "default", (
        "model resolved to the 'default' sentinel; expected a profile-derived "
        "model. process_conversation_turn is not consulting resolve_memory_model()."
    )
    assert not model.startswith("ollama:"), (
        f"provider prefix leaked into the HTTP payload: {model!r}. "
        f"Ollama rejects this and extraction silently falls back to regex."
    )
    assert model == "qwen3.5:9b", (
        f"expected balanced@gpu-12gb -> bare 'qwen3.5:9b', got {model!r}"
    )
    # The 200-path parsed the LLM facts, so extraction stayed on the llm method.
    assert result["method"] == "llm"


def test_unknown_tier_falls_back_to_default(monkeypatch, tmp_path):
    """Tier absent from the balanced map: model falls through to 'default'."""
    _pin_tier(monkeypatch, "unknown-tier")
    _clear_pins(monkeypatch)

    captured: list[str] = []
    monkeypatch.setattr(
        "taosmd.memory_extractor.extract_facts_with_llm",
        _make_spy(captured),
    )

    async def go():
        from taosmd.memory_extractor import process_conversation_turn

        kg = TemporalKnowledgeGraph(db_path=tmp_path / "kg.db")
        await kg.init()
        try:
            with pytest.raises(_Captured):
                await process_conversation_turn(
                    text="Bob drinks coffee.",
                    agent_name=None,
                    kg=kg,
                    llm_url="http://localhost:11434",
                    http_client=object(),
                    use_llm=True,
                    extraction_model="default",
                )
        finally:
            await kg.close()

    _run(go())

    assert len(captured) == 1
    assert captured[0] == "default", (
        f"expected 'default' fallback for unknown tier, got {captured[0]!r}"
    )


def test_llm_failure_regex_fallback_logs_warning_once_then_debug(monkeypatch, tmp_path, caplog):
    """The regex fallback is loud (WARNING) on the first failure per (model, url), then debug.

    A dead Ollama would otherwise emit one WARNING per conversation turn.
    """
    _pin_tier(monkeypatch, "gpu-12gb")
    _clear_pins(monkeypatch)
    monkeypatch.setattr("taosmd.memory_extractor._llm_fallback_warned", set())

    payloads: list[dict] = []
    client = _PayloadCapturingClient(payloads, response=RuntimeError("connection refused"))

    async def go():
        from taosmd.memory_extractor import process_conversation_turn

        kg = TemporalKnowledgeGraph(db_path=tmp_path / "kg.db")
        await kg.init()
        try:
            first = await process_conversation_turn(
                text="Alice likes tea.",
                agent_name=None,
                kg=kg,
                llm_url="http://localhost:11434",
                http_client=client,
                use_llm=True,
                extraction_model="default",
            )
            second = await process_conversation_turn(
                text="Hello again.",
                agent_name=None,
                kg=kg,
                llm_url="http://localhost:11434",
                http_client=client,
                use_llm=True,
                extraction_model="default",
            )
        finally:
            await kg.close()
        return first, second

    with caplog.at_level(logging.DEBUG, logger="taosmd.memory_extractor"):
        first, second = _run(go())

    assert first["method"] == "regex"
    assert second["method"] == "regex"
    fallback_records = [
        r for r in caplog.records if "falling back to regex" in r.getMessage()
    ]
    assert len(fallback_records) == 2
    levels = [r.levelno for r in fallback_records]
    assert levels[0] == logging.WARNING, (
        "first LLM failure per (model, url) must be a WARNING"
    )
    assert levels[1] == logging.DEBUG, (
        "repeat LLM failures for the same (model, url) must drop to debug"
    )
