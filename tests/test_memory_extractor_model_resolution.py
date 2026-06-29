"""Regression test: process_conversation_turn uses resolve_memory_model, not get_memory_model.

Verifies the actual code path at memory_extractor.py L259-263: when extraction_model
is "default" and no explicit pin exists, the function must call resolve_memory_model()
(which consults the generator-profile registry) rather than get_memory_model() (which
would return None on a fresh install, silently yielding the "default" sentinel and
bypassing the profile system).

These tests FAIL on the pre-fix code that used get_memory_model() because:
- get_memory_model is monkeypatched to return None
- on the buggy path: resolved_model = get_memory_model() or "default" -> "default"
- on the fixed path: resolved_model = resolve_memory_model() -> "ollama:qwen3.5:9b"
  (balanced profile, gpu-12gb tier), so Test 1 rejects the "default" sentinel.
"""
from __future__ import annotations

import asyncio

import pytest

from taosmd import config
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_covered_tier_resolves_real_model(monkeypatch, tmp_path):
    """Test 1: covered tier -> extract_facts_with_llm receives the profile model, not 'default'.

    This is the regression assertion: on the buggy pre-fix code (get_memory_model()),
    get_memory_model is None so the path yields "default". The fixed code calls
    resolve_memory_model() which returns "ollama:qwen3.5:9b", so Test 1 passes ONLY
    on the fixed code.
    """
    # Deterministic tier resolution: balanced profile, gpu-12gb coverage.
    monkeypatch.setattr(gp.recipes, "local_probe", lambda: {"host": {}})
    monkeypatch.setattr(gp.recipes, "tier_of", lambda info: "gpu-12gb")

    # No explicit memory-model pin and no explicit generator-profile pin.
    monkeypatch.setattr(
        "taosmd.config.get_memory_model", lambda data_dir=None: None
    )
    monkeypatch.setattr(
        "taosmd.config.get_generator_profile", lambda data_dir=None: None
    )

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
                    text="Alice likes tea.",
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

    assert len(captured) == 1, "spy was not called exactly once"
    model = captured[0]
    assert model != "default", (
        f"model resolved to sentinel 'default'; expected a profile-derived model. "
        f"This indicates process_conversation_turn is still using get_memory_model() "
        f"instead of resolve_memory_model()."
    )
    assert "qwen3.5" in model, (
        f"expected balanced@gpu-12gb -> qwen3.5:9b, got {model!r}"
    )


def test_unknown_tier_falls_back_to_default(monkeypatch, tmp_path):
    """Test 2: tier absent from the balanced map -> model falls through to 'default'."""
    monkeypatch.setattr(gp.recipes, "local_probe", lambda: {"host": {}})
    monkeypatch.setattr(gp.recipes, "tier_of", lambda info: "unknown-tier")

    monkeypatch.setattr(
        "taosmd.config.get_memory_model", lambda data_dir=None: None
    )
    monkeypatch.setattr(
        "taosmd.config.get_generator_profile", lambda data_dir=None: None
    )

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
