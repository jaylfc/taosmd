from __future__ import annotations
import pytest
from taosmd import recipes


def test_recipe_dataclass_roundtrips_to_dict():
    r = recipes.Recipe(
        id="x", name="X",
        retrieval={"strategy": "thorough", "limit": 5, "candidate_top_k": 20,
                   "fusion": "boost", "reranker": "none",
                   "adjacent_neighbors": 0, "llm_reranker": False},
        ingest={"extraction": True, "extraction_model": "", "embed_verbatim": True},
        generator={"model": ""},
        librarian={"fanout": "low", "worker_aware": True},
        metadata={"tier": "gpu-12gb", "scores": {}, "pros": [], "cons": [],
                  "est_latency": "medium", "est_footprint": "medium", "source": "t"},
    )
    d = r.to_dict()
    assert d["id"] == "x"
    assert d["retrieval"]["fusion"] == "boost"
    assert recipes.Recipe.from_dict(d).to_dict() == d


def test_recipe_schema_is_valid_object_with_four_sections():
    schema = recipes.recipe_schema()
    assert schema["type"] == "object"
    for section in ("retrieval", "ingest", "generator", "librarian", "metadata"):
        assert section in schema["properties"], section
    fusion = schema["properties"]["retrieval"]["properties"]["fusion"]
    assert fusion["enum"] == ["boost", "rrf", "mem0_additive"]
    fanout = schema["properties"]["librarian"]["properties"]["fanout"]
    assert fanout["enum"] == ["off", "low", "med", "high"]
