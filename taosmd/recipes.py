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
              "scores": {"gemma4:e2b": 0.636, "llama3.1:8b": 0.433,
                         "qwen3:4b-instruct-2507": 0.608},
              "pros": ["About 2.4x faster than the 9B leader, fits a 4 GB GPU",
                       "Edges the leader on the strict llama judge (noise band)"],
              "cons": ["About 0.04 to 0.05 below the leader on the other two judges"],
              "est_latency": "low", "est_footprint": "low",
              "source": "docs/benchmarks.md full-1540 tri-judge"}))

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
              "scores": {"gemma4:e2b": 0.607, "llama3.1:8b": 0.353,
                         "qwen3:4b-instruct-2507": 0.568},
              "pros": ["No LLM extraction cost per turn; safe on Pi 4B CPU",
                       "Archive + embed + retrieve unchanged, so recall holds"],
              "cons": ["No enriched facts/events; relies on raw embedding recall",
                       "About 0.04 to 0.07 below the leader across judges"],
              "est_latency": "low", "est_footprint": "low",
              "source": "docs/benchmarks.md full-1540 tri-judge"}))


def get_recipe(recipe_id: str) -> Recipe | None:
    return _REGISTRY.get(recipe_id)


def list_recipes() -> list[Recipe]:
    return list(_REGISTRY.values())


import os
import shutil
import subprocess


def _detect_gpu() -> dict:
    """Best-effort, dependency-free GPU sniff."""
    # NVIDIA via nvidia-smi
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                vram = int(out.stdout.strip().splitlines()[0].strip())
                return {"type": "nvidia", "vram_mb": vram}
        except Exception:
            pass
    # Apple Metal (unified memory): present on darwin with arm64
    import platform
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return {"type": "metal", "vram_mb": _total_ram_mb()}
    return {"type": "none", "vram_mb": 0}


def _detect_npu() -> dict:
    # Rockchip RKNPU exposes /sys/kernel/debug/rknpu or a devfreq node.
    for p in ("/sys/kernel/debug/rknpu", "/sys/class/devfreq/fdab0000.npu"):
        if os.path.exists(p):
            return {"type": "rknpu"}
    return {"type": "none"}


def _total_ram_mb() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 * 1024))
    except (ValueError, OSError, AttributeError):
        return 0


def local_probe() -> dict:
    """Minimal, dependency-free hardware sniff.

    Returns the same ``device_info`` shape taOS produces (host section only,
    no cluster), so ``recommend()`` has one input vocabulary. Standalone never
    needs taOS to run.
    """
    import platform
    return {"host": {
        "cpu": {"arch": platform.machine(), "cores": os.cpu_count() or 1},
        "ram_mb": _total_ram_mb(),
        "npu": _detect_npu(),
        "gpu": _detect_gpu(),
    }}


def tier_of(device_info: dict) -> str:
    """Classify a device_info dict into a coarse tier string."""
    host = device_info.get("host", {})
    gpu = host.get("gpu", {})
    npu = host.get("npu", {})
    vram = gpu.get("vram_mb", 0) or 0
    if gpu.get("type") not in (None, "none") and vram >= 11000:
        return "gpu-12gb"
    if gpu.get("type") not in (None, "none") and vram >= 7000:
        return "gpu-8gb"
    if gpu.get("type") not in (None, "none") and vram >= 3500:
        return "gpu-4gb"
    if npu.get("type") not in (None, "none"):
        return "pi-npu"
    return "cpu"


# Tier capability order: a recipe whose tier needs MORE than the device has
# must never outrank one that fits. Lower index = lighter requirement.
_TIER_RANK = {"pi-npu": 0, "cpu": 0, "gpu-4gb": 1, "gpu-8gb": 2,
              "gpu-12gb": 3, "unconstrained": 4}


def _fits(recipe_tier: str, device_tier: str) -> bool:
    """True if a recipe built for recipe_tier can run on device_tier."""
    # CPU/pi devices cannot run gpu-tier recipes; gpu devices can run lighter.
    dev = _TIER_RANK.get(device_tier, 0)
    req = _TIER_RANK.get(recipe_tier, 0)
    if device_tier in ("cpu", "pi-npu"):
        return recipe_tier in ("cpu", "pi-npu")
    return req <= dev


def _score_for(recipe: Recipe) -> float:
    """A single comparable quality number for ranking (lenient judge, else 0)."""
    scores = recipe.metadata.get("scores", {})
    return scores.get("gemma4:e2b", 0.0)


def recommend(device_info: dict | None = None) -> list[Recipe]:
    """Rank the built-in registry best-first for the given (or probed) device.

    Recipes that do not fit the device tier are sorted last; among fitting
    recipes, higher benchmarked quality ranks first. Returns Recipe objects.
    """
    if device_info is None:
        device_info = local_probe()
    device_tier = tier_of(device_info)

    def key(r: Recipe):
        fits = _fits(r.metadata.get("tier", "cpu"), device_tier)
        # fits first (True sorts before False via not-fits=0), then quality desc
        return (0 if fits else 1, -_score_for(r))

    return sorted(list_recipes(), key=key)


from . import config as _config
from . import agents as _agents

# Which librarian tasks count as "LLM extraction" for the lite path.
_EXTRACTION_TASKS = ("fact_extraction", "preference_extraction",
                     "intake_classification", "catalog_enrichment")


def _librarian_for(recipe: Recipe) -> dict:
    """Build the agent librarian block this recipe implies (write-through)."""
    base = _agents._default_librarian()
    base["fanout"]["default"] = recipe.librarian.get("fanout", "low")
    base["fanout"]["auto_scale"] = recipe.librarian.get("worker_aware", True)
    if not recipe.ingest.get("extraction", True):
        base["enabled"] = False
        for t in _EXTRACTION_TASKS:
            if t in base["tasks"]:
                base["tasks"][t] = False
    return base


def apply_recipe(agent: str, recipe_id: str, data_dir=None) -> Recipe:
    """Apply a recipe to an agent: write its knobs through to the stores.

    Writes applied_recipe_id + the flattened retrieval_config + the implied
    librarian block to the agent record, and the generator model to config
    when the recipe names one. Raises ValueError on an unknown id.
    """
    recipe = get_recipe(recipe_id)
    if recipe is None:
        raise ValueError(f"unknown recipe id: {recipe_id!r}")
    _agents.set_agent_recipe_config(
        agent, recipe_id=recipe_id,
        retrieval_config=dict(recipe.retrieval),
        librarian=_librarian_for(recipe), data_dir=data_dir)
    gen = recipe.generator.get("model", "")
    # Only seed the global memory model from the recipe when the user has not
    # already chosen one. The fresh-install path (resolve_recipe -> apply_recipe
    # for recommend()[0]) must not clobber a model the user configured.
    if gen and _config.get_memory_model(data_dir=data_dir) is None:
        _config.set_memory_model(gen, data_dir=data_dir)
    return recipe


def resolve_recipe(agent: str, data_dir=None) -> Recipe:
    """Resolve the active recipe: per-agent -> global default -> recommend()[0].

    On a fresh agent with nothing configured, lazily materialise recommend()[0]
    via apply_recipe so the choice is stored and visible (write-through).
    """
    rid = _agents.get_applied_recipe(agent, data_dir=data_dir)
    if rid and get_recipe(rid):
        # Merge stored (possibly manually-edited) retrieval_config over the recipe.
        recipe = get_recipe(rid)
        stored = _agents.get_agent_retrieval_config(agent, data_dir=data_dir)
        if stored:
            merged = Recipe.from_dict(recipe.to_dict())
            merged.retrieval.update(stored)
            return merged
        return recipe
    gid = _config.get_default_recipe(data_dir=data_dir)
    if gid and get_recipe(gid):
        return apply_recipe(agent, gid, data_dir=data_dir)
    top = recommend(None)[0]
    return apply_recipe(agent, top.id, data_dir=data_dir)


import threading
from pathlib import Path

# HuggingFace repo for the bge reranker v2-m3 ONNX export.
_RERANKER_REPO = "BAAI/bge-reranker-v2-m3"
_RERANKER_DOWNLOADS: dict[str, str] = {}  # onnx_path -> "downloading"|"ready"|"error"


def _reranker_present(onnx_path: str) -> bool:
    for c in (Path(onnx_path) / "model.onnx", Path(onnx_path) / "onnx" / "model.onnx"):
        if c.exists():
            return True
    return False


def _fetch_reranker_onnx(dest: str, on_progress) -> str | None:
    """Download the bge-v2-m3 ONNX into dest, reporting progress. Network IO.

    Uses huggingface_hub (already a transitive dep via transformers) with
    tqdm progress mapped to on_progress events. Raises on failure.
    """
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    on_progress({"phase": "start", "pct": 0, "repo": _RERANKER_REPO})
    path = snapshot_download(
        repo_id=_RERANKER_REPO,
        allow_patterns=["*.onnx", "*.json", "tokenizer*", "*.txt"],
        local_dir=dest,
    )
    on_progress({"phase": "done", "pct": 100})
    return path


def ensure_reranker_model(onnx_path: str = "models/cross-encoder-onnx",
                          on_progress=None, block: bool = False) -> str:
    """Ensure the bge-v2-m3 ONNX is present; download with visible progress.

    Returns "ready" if present, "downloading" if a background fetch is in
    flight (block=False), or "error". Never blocks a caller unless block=True.
    Progress events are dicts {phase, pct, ...} passed to on_progress.
    """
    on_progress = on_progress or (lambda e: None)
    if _reranker_present(onnx_path):
        return "ready"
    if _RERANKER_DOWNLOADS.get(onnx_path) == "downloading":
        return "downloading"

    # Mark "downloading" on the main thread BEFORE starting the worker, so a
    # concurrent caller observes the in-flight status and does not spawn a
    # duplicate download (closes the TOCTOU window between start() and _run()).
    _RERANKER_DOWNLOADS[onnx_path] = "downloading"

    def _run():
        try:
            _fetch_reranker_onnx(onnx_path, on_progress)
            _RERANKER_DOWNLOADS[onnx_path] = "ready"
        except Exception as exc:  # noqa: BLE001
            _RERANKER_DOWNLOADS[onnx_path] = "error"
            on_progress({"phase": "error", "error": str(exc)})

    if block:
        _run()
        return _RERANKER_DOWNLOADS.get(onnx_path, "error")
    threading.Thread(target=_run, daemon=True).start()
    return "downloading"
