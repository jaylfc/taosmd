"""Extraction-method counters and the consecutive-fallback alert (issue #653 item 4).

A permanent, silent degrade from LLM extraction to the regex path is the most
dangerous failure mode in the shadow report: 10,063/10,063 claims came from
regex and nothing surfaced it. This test pins two things:

1. ``extraction_stats()`` counts llm / regex_fallback / regex_direct per call,
   with no double-counting when both paths come up empty.
2. After ``_FALLBACK_ALERT_AFTER`` consecutive regex fallbacks, the extractor
   logs one loud WARNING containing ``consecutive regex fallback``, and an LLM
   success resets the streak so the alert re-arms.
"""
from __future__ import annotations

import asyncio
import logging

import pytest

from taosmd.knowledge_graph import TemporalKnowledgeGraph
import taosmd.memory_extractor as me


def _run(coro):
    return asyncio.run(coro)


class _FakeResponse:
    status_code = 200

    def json(self):
        return {
            "choices": [{
                "message": {
                    "content": (
                        '[{"subject": "Alice", "predicate": "prefers", '
                        '"object": "green tea"}]'
                    )
                }
            }]
        }


class _Client:
    """Fake http client: returns *response* (object or Exception) from post()."""

    def __init__(self, response=None):
        self._response = response or _FakeResponse()

    async def post(self, url, json=None, timeout=None):
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


@pytest.fixture()
def fresh_stats(monkeypatch):
    """Reset the module-level counters and streak before every test."""
    me._reset_extraction_stats()
    yield me.extraction_stats()
    me._reset_extraction_stats()


async def _turn(kg, **kw):
    from taosmd.memory_extractor import process_conversation_turn

    defaults = dict(
        text="Alice prefers green tea.",
        agent_name=None,
        kg=kg,
        llm_url="http://localhost:11434",
        http_client=_Client(),
        use_llm=True,
        extraction_model="default",
    )
    defaults.update(kw)
    return await process_conversation_turn(**defaults)


def _go(**kw):
    async def inner(tmp_path):
        kg = TemporalKnowledgeGraph(db_path=tmp_path / "kg.db")
        await kg.init()
        try:
            return await _turn(kg, **kw)
        finally:
            await kg.close()
    return inner


def test_stats_count_llm_success(fresh_stats, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "taosmd.memory_extractor.resolve_memory_model",
        lambda agent=None: "qwen3.5:9b",
        raising=False,
    )
    _run(_go()(tmp_path))
    stats = me.extraction_stats()
    assert stats["llm"] == 1
    assert stats["regex_fallback"] == 0
    assert stats["regex_direct"] == 0


def test_stats_count_regex_fallback_on_llm_error(fresh_stats, tmp_path):
    _run(_go(http_client=_Client(RuntimeError("connection refused")))(tmp_path))
    stats = me.extraction_stats()
    assert stats["regex_fallback"] == 1
    assert stats["llm"] == 0
    assert stats["regex_direct"] == 0


def test_stats_count_empty_llm_result_as_fallback(fresh_stats, tmp_path):
    """LLM answers but extracts nothing -> regex takes over -> counted fallback."""

    class _EmptyResponse:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "[]"}}]}

    result = _run(_go(http_client=_Client(_EmptyResponse()))(tmp_path))
    stats = me.extraction_stats()
    assert stats["regex_fallback"] == 1
    assert stats["llm"] == 0
    assert result["method"] == "regex"


def test_stats_no_double_count_when_both_paths_empty(fresh_stats, tmp_path):
    """LLM raises AND regex finds no facts: exactly one regex_fallback, not two."""
    _run(_go(
        text="hello.",
        http_client=_Client(RuntimeError("boom")),
    )(tmp_path))
    stats = me.extraction_stats()
    assert stats["regex_fallback"] == 1


def test_stats_count_regex_direct_without_llm(fresh_stats, tmp_path):
    _run(_go(llm_url="")(tmp_path))
    _run(_go(llm_url="http://x", use_llm=False)(tmp_path))
    stats = me.extraction_stats()
    assert stats["regex_direct"] == 2
    assert stats["llm"] == 0
    assert stats["regex_fallback"] == 0


def test_consecutive_regex_fallback_alerts_loud_once_then_resets(
    monkeypatch, tmp_path, caplog
):
    """After N consecutive fallbacks the extractor WARNs once; an llm success re-arms."""
    me._reset_extraction_stats()
    monkeypatch.setattr(me, "_FALLBACK_ALERT_AFTER", 3)
    monkeypatch.setattr(me, "_llm_fallback_warned", set())

    failing = _Client(RuntimeError("connection refused"))
    working = _Client()

    async def inner():
        kg = TemporalKnowledgeGraph(db_path=tmp_path / "kg.db")
        await kg.init()
        try:
            for _ in range(4):
                await _turn(kg, text="Alice prefers green tea.", http_client=failing)
            # An LLM success must reset the streak.
            monkeypatch.setattr(
                "taosmd.memory_extractor.resolve_memory_model",
                lambda agent=None: "qwen3.5:9b",
                raising=False,
            )
            await _turn(kg, text="Alice prefers green tea.", http_client=working)
            for _ in range(3):
                await _turn(kg, text="Alice prefers green tea.", http_client=failing)
        finally:
            await kg.close()

    with caplog.at_level(logging.DEBUG, logger="taosmd.memory_extractor"):
        _run(inner())

    alerts = [
        r for r in caplog.records if "consecutive regex fallback" in r.getMessage()
    ]
    assert len(alerts) == 2, (
        "expected exactly one 'consecutive regex fallback' WARNING per streak"
    )
    assert all(r.levelno == logging.WARNING for r in alerts)
    me._reset_extraction_stats()
