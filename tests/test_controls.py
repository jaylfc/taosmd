"""Tests for taosmd.controls, the memory-controls registry (source of truth)."""
from __future__ import annotations

import pytest

from taosmd import controls as C


def test_defaults_include_every_control():
    d = C.default_controls()
    assert set(d) == set(C.CONTROLS)
    # prefer_verified ships on by default (E-018 tri-judge evidenced).
    assert d["prefer_verified"] == "prefer_verified"
    assert d["embedder"] == "arctic-embed-s"


def test_validate_choice():
    assert C.validate_control("reranker", "bge-v2-m3") == "bge-v2-m3"
    with pytest.raises(ValueError):
        C.validate_control("reranker", "not-a-reranker")


def test_validate_bool():
    assert C.validate_control("late_interaction", True) is True
    with pytest.raises(ValueError):
        C.validate_control("late_interaction", "yes")


def test_validate_int_range():
    assert C.validate_control("adjacent_turns", 2) == 2
    assert C.validate_control("adjacent_turns", "1") == 1  # coerced
    with pytest.raises(ValueError):
        C.validate_control("adjacent_turns", 99)


def test_unknown_control_rejected():
    with pytest.raises(ValueError):
        C.validate_control("nope", True)


def test_self_verify_is_consumer_scope():
    # taOSmd retrieves; it does not generate answers. self_verify is a documented
    # consumer-side recommendation, never a core toggle that silently does nothing.
    assert C.CONTROLS["self_verify"].scope == "consumer"


def test_schema_shape_and_pros_cons_present():
    s = C.controls_schema()
    assert {"controls", "presets"} <= set(s)
    ids = {c["id"] for c in s["controls"]}
    assert ids == set(C.CONTROLS)
    for c in s["controls"]:
        # the README/dashboard need cost + pros + cons for every control.
        assert c["cost"] and c["pros"] and c["cons"], c["id"]


def test_presets_reference_valid_controls():
    for p in C.PRESETS.values():
        for cid, val in p["values"].items():
            assert cid in C.CONTROLS, cid
            C.validate_control(cid, val)  # each preset value is valid
