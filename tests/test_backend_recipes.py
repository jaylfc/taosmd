"""Tests for the SP4 recipe methods on TaOSmdBackend (the contract seam).

Hermetic: each test gets a fresh backend rooted in a tmp dir, so recipe
config (config.json) and agent records (agents.json) write under tmp_path.
"""

from __future__ import annotations

import asyncio

import pytest

from taosmd.taosmd_backend import TaOSmdBackend
from taosmd import recipes as _recipes
from taosmd import agents as _agents
from taosmd import config as _config


@pytest.fixture
def backend(tmp_path):
    # data_dir is derived as the parent of settings_db_path, i.e. tmp_path.
    return TaOSmdBackend(settings_db_path=str(tmp_path / "memory-settings.db"))


def test_schema_matches_core(backend):
    schema = asyncio.run(backend.get_recipe_schema())
    assert schema == _recipes.recipe_schema()
    assert schema["type"] == "object"


def test_list_recipes(backend):
    recipes = asyncio.run(backend.list_recipes())
    ids = {r["id"] for r in recipes}
    assert {"maxsim-rerank-9b", "rrf-9b", "fast-8b", "lite-pi"} <= ids
    assert all("metadata" in r for r in recipes)


def test_get_recipe(backend):
    assert asyncio.run(backend.get_recipe("rrf-9b"))["id"] == "rrf-9b"
    assert asyncio.run(backend.get_recipe("nope")) is None


def test_recommend_ranks_and_annotates(backend):
    gpu12 = {"host": {"gpu": {"type": "nvidia", "vram_mb": 12000},
                      "npu": {"type": "none"}, "cpu": {"cores": 8}, "ram_mb": 32000}}
    ranked = asyncio.run(backend.recommend(gpu12))
    assert ranked[0]["id"] == "maxsim-rerank-9b"
    assert "rationale" in ranked[0] and ranked[0]["rationale"]


def test_apply_recipe_agent_then_global(backend, tmp_path):
    _agents.ensure_agent("bob", data_dir=str(tmp_path))

    res = asyncio.run(backend.apply_recipe("rrf-9b", agent="bob"))
    assert res["applied_recipe_id"] == "rrf-9b"
    assert res["recipe"]["id"] == "rrf-9b"
    assert _agents.get_applied_recipe("bob", data_dir=str(tmp_path)) == "rrf-9b"

    glob = asyncio.run(backend.apply_recipe("fast-8b"))  # agent=None -> global default
    assert glob["applied_recipe_id"] == "fast-8b"
    assert _config.get_default_recipe(data_dir=str(tmp_path)) == "fast-8b"


def test_apply_unknown_recipe_raises(backend):
    with pytest.raises(ValueError):
        asyncio.run(backend.apply_recipe("does-not-exist"))


def test_create_recipe_not_implemented(backend):
    with pytest.raises(NotImplementedError):
        asyncio.run(backend.create_recipe({}))
