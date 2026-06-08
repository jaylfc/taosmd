"""Recipes: first-class named config bundles for taosmd retrieval + ingest.

A Recipe declares four typed config sections (retrieval, ingest, generator,
librarian) plus a read-only metadata block (benchmark scores, tier, pros/cons).
The schema export lets any consumer render a recipe generically; the registry
holds the recipes we have actually benchmarked; resolution + application make a
fresh install run a real recipe. See docs/superpowers/specs/2026-06-08-recipes-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Recipe:
    id: str
    name: str
    retrieval: dict
    ingest: dict
    generator: dict
    librarian: dict
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Recipe":
        return cls(
            id=d["id"], name=d["name"],
            retrieval=dict(d["retrieval"]), ingest=dict(d["ingest"]),
            generator=dict(d["generator"]), librarian=dict(d["librarian"]),
            metadata=dict(d.get("metadata", {})),
        )


def recipe_schema() -> dict:
    """JSON Schema for the recipe config bundle (renderable generically)."""
    return {
        "type": "object",
        "required": ["id", "name", "retrieval", "ingest", "generator", "librarian"],
        "properties": {
            "id": {"type": "string", "description": "Stable recipe slug"},
            "name": {"type": "string", "description": "Human label"},
            "retrieval": {
                "type": "object",
                "properties": {
                    "strategy": {"type": "string",
                                 "enum": ["thorough", "fast", "minimal", "custom"],
                                 "default": "thorough",
                                 "description": "Retrieval strategy"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50,
                              "default": 5, "description": "Final result count"},
                    "candidate_top_k": {"type": "integer", "minimum": 5,
                                        "maximum": 100, "default": 20,
                                        "description": "Pool size before rerank"},
                    "fusion": {"type": "string",
                               "enum": ["boost", "rrf", "mem0_additive"],
                               "default": "boost",
                               "description": "Hybrid fusion mode"},
                    "reranker": {"type": "string",
                                 "enum": ["none", "bge-v2-m3"], "default": "none",
                                 "description": "Cross-encoder reranker"},
                    "adjacent_neighbors": {"type": "integer", "minimum": 0,
                                           "maximum": 4, "default": 0,
                                           "description": "Positional neighbours per hit"},
                    "llm_reranker": {"type": "boolean", "default": False,
                                     "description": "Listwise LLM second pass"},
                },
            },
            "ingest": {
                "type": "object",
                "properties": {
                    "extraction": {"type": "boolean", "default": True,
                                   "description": "Run LLM enrichment on ingest"},
                    "extraction_model": {"type": "string", "default": "",
                                         "description": "provider:model or '' to inherit"},
                    "embed_verbatim": {"type": "boolean", "default": True,
                                       "description": "Archive + embed raw turns"},
                },
            },
            "generator": {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "default": "",
                              "description": "provider:model for synthesis, '' to inherit"},
                },
            },
            "librarian": {
                "type": "object",
                "properties": {
                    "fanout": {"type": "string",
                               "enum": ["off", "low", "med", "high"], "default": "low",
                               "description": "Per-layer fan-out level"},
                    "worker_aware": {"type": "boolean", "default": True,
                                     "description": "Scale fanout by worker capabilities"},
                },
            },
            "metadata": {
                "type": "object",
                "description": "Read-only display + ranking input",
                "properties": {
                    "tier": {"type": "string",
                             "enum": ["pi-npu", "cpu", "gpu-4gb", "gpu-8gb",
                                      "gpu-12gb", "unconstrained"]},
                    "scores": {"type": "object",
                               "description": "judge -> float"},
                    "pros": {"type": "array", "items": {"type": "string"}},
                    "cons": {"type": "array", "items": {"type": "string"}},
                    "est_latency": {"type": "string",
                                    "enum": ["low", "medium", "high"]},
                    "est_footprint": {"type": "string",
                                      "enum": ["low", "medium", "high"]},
                    "source": {"type": "string"},
                },
            },
        },
    }
