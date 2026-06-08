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


# Built-in recipes. Scores are copied verbatim from docs/benchmarks.md
# "Full-1540 leader (tri-judge, Jun 2026)". Do not invent numbers.
_REGISTRY: dict[str, Recipe] = {}


def _register(r: Recipe) -> None:
    _REGISTRY[r.id] = r


_register(Recipe(
    id="maxsim-rerank-9b",
    name="MaxSim + rerank (12 GB GPU)",
    retrieval={"strategy": "thorough", "limit": 5, "candidate_top_k": 50,
               "fusion": "mem0_additive", "reranker": "bge-v2-m3",
               "adjacent_neighbors": 2, "llm_reranker": False},
    ingest={"extraction": True, "extraction_model": "", "embed_verbatim": True},
    generator={"model": "ollama:qwen3.5:9b"},
    librarian={"fanout": "med", "worker_aware": True},
    metadata={"tier": "gpu-12gb",
              "scores": {"gemma4:e2b": 0.748, "llama3.1:8b": 0.394,
                         "qwen3:4b-instruct-2507": 0.659},
              "pros": ["Highest recall on every judge",
                       "Best on a 12 GB GPU where the reranker fits"],
              "cons": ["Needs the bge-v2-m3 cross-encoder (a model download)",
                       "Higher per-query latency than RRF"],
              "est_latency": "high", "est_footprint": "high",
              "source": "docs/benchmarks.md full-1540 tri-judge"}))

_register(Recipe(
    id="rrf-9b",
    name="RRF (12 GB GPU, no reranker)",
    retrieval={"strategy": "thorough", "limit": 5, "candidate_top_k": 20,
               "fusion": "rrf", "reranker": "none",
               "adjacent_neighbors": 2, "llm_reranker": True},
    ingest={"extraction": True, "extraction_model": "", "embed_verbatim": True},
    generator={"model": "ollama:qwen3.5:9b"},
    librarian={"fanout": "med", "worker_aware": True},
    metadata={"tier": "gpu-12gb",
              "scores": {"gemma4:e2b": 0.723, "llama3.1:8b": 0.390,
                         "qwen3:4b-instruct-2507": 0.634},
              "pros": ["Leader-class recall with no reranker download",
                       "Lower latency than MaxSim+rerank"],
              "cons": ["About 0.025 below the leader on lenient + qwen judges"],
              "est_latency": "medium", "est_footprint": "medium",
              "source": "docs/benchmarks.md full-1540 tri-judge"}))

_register(Recipe(
    id="fast-8b",
    name="Fast (4 GB GPU / realtime)",
    retrieval={"strategy": "thorough", "limit": 5, "candidate_top_k": 20,
               "fusion": "rrf", "reranker": "none",
               "adjacent_neighbors": 2, "llm_reranker": False},
    ingest={"extraction": True, "extraction_model": "", "embed_verbatim": True},
    generator={"model": "ollama:llama3.1:8b"},
    librarian={"fanout": "low", "worker_aware": True},
    metadata={"tier": "gpu-4gb",
              "scores": {},
              "pros": ["About 2.4x faster than the 9B leader, only ~0.02 lower",
                       "Fits a 4 GB GPU; good for multiple agents on one card"],
              "cons": ["Not full-1540 tri-judged; subset/cross-tier figure only"],
              "est_latency": "low", "est_footprint": "low",
              "source": "docs/benchmarks.md generator candidates (subset)"}))

_register(Recipe(
    id="lite-pi",
    name="Lite (no-LLM ingest, Pi / CPU)",
    retrieval={"strategy": "fast", "limit": 5, "candidate_top_k": 20,
               "fusion": "boost", "reranker": "none",
               "adjacent_neighbors": 2, "llm_reranker": False},
    ingest={"extraction": False, "extraction_model": "", "embed_verbatim": True},
    generator={"model": ""},
    librarian={"fanout": "low", "worker_aware": False},
    metadata={"tier": "pi-npu",
              "scores": {},
              "pros": ["No LLM extraction cost per turn; safe on Pi 4B CPU",
                       "Archive + embed + retrieve unchanged, so recall holds"],
              "cons": ["No enriched facts/events; relies on raw embedding recall"],
              "est_latency": "low", "est_footprint": "low",
              "source": "Midas-class no-LLM-ingest, low tier"}))


def get_recipe(recipe_id: str) -> Recipe | None:
    return _REGISTRY.get(recipe_id)


def list_recipes() -> list[Recipe]:
    return list(_REGISTRY.values())
