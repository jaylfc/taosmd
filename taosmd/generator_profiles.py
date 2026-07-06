"""Task-aware generator profiles.

A GeneratorProfile maps a workload to the best answer/memory generator PER
hardware tier. The active profile overrides the retrieval recipe's generator at
resolution time (see resolve_generator, added in a later task). An empty model
string is the retrieval-only (no local LLM) backend. Profiles are pure data, so
a new workload-to-model mapping is one row.

Design: docs/superpowers/specs/2026-06-24-task-aware-generator-profiles-design.md
"""
from dataclasses import dataclass, field
from . import recipes
from . import config as _config
from . import agents as _agents

# Highest VRAM first. Mirrors recipes.tier_of's vocabulary.
TIER_ORDER: tuple[str, ...] = ("gpu-12gb", "gpu-8gb", "gpu-4gb", "pi-npu", "cpu")

# Provider tokens recognised at the model boundary. Registry values are
# "provider:model" strings; backends want the bare model name.
KNOWN_PROVIDERS: tuple[str, ...] = ("ollama",)


def split_provider(resolved: str) -> tuple[str, str]:
    """Split a resolved ``provider:model`` string into ``(provider, model)``.

    Only a LEADING known provider token counts: model names legitimately
    contain colons (``qwen3.5:9b`` is a bare model, not a provider), so
    anything that doesn't start with a known provider passes through
    unchanged as ``("", resolved)``. Empty input yields ``("", "")``.

    Use this at HTTP call sites so the bare model name is what reaches the
    backend (live Ollama rejects the prefixed form); keep the prefixed
    string for storage and CLI display.
    """
    if not resolved:
        return ("", "")
    head, sep, rest = resolved.partition(":")
    if sep and head.lower() in KNOWN_PROVIDERS:
        return (head.lower(), rest)
    return ("", resolved)


@dataclass
class GeneratorProfile:
    id: str
    label: str
    workload: str
    models: dict  # tier -> "provider:model" ("" == retrieval-only / none)
    evidence: dict = field(default_factory=dict)
    notes: str = ""


_REGISTRY: dict[str, GeneratorProfile] = {}


def _register(p: GeneratorProfile) -> None:
    _REGISTRY[p.id] = p


_register(GeneratorProfile(
    id="balanced",
    label="Balanced (multi-session and long-context)",
    workload="multi-session conversational and long-context recall; the LoCoMo "
             "and BEAM leader. The default.",
    models={
        "gpu-12gb": "ollama:qwen3.5:9b",
        "gpu-8gb": "ollama:qwen3.5:9b",
        "gpu-4gb": "ollama:llama3.1:8b",
        "pi-npu": "",
        "cpu": "",
    },
    evidence={"source": "docs/benchmarks.md LoCoMo full-1540 tri-judge + BEAM"},
    notes="Mirrors the shipped per-tier recipe generators, so this default "
          "reproduces today's behaviour.",
))

_register(GeneratorProfile(
    id="factual-recall",
    label="Factual recall (single-fact QA)",
    workload="single-fact retrieval QA, LongMemEval style. NOT for "
             "conversational or long-context use.",
    models={
        "gpu-12gb": "ollama:granite4.1:8b",
        "gpu-8gb": "ollama:granite4.1:8b",
        "gpu-4gb": "ollama:llama3.1:8b",
    },
    evidence={
        "source": "docs/research-report.md F-016 / E-024 (granite4.1:8b "
                  "full-500 + LoCoMo probe) + N-017 (gemma4:12b prior pick) + "
                  "F-015 / E-023 (8 and 4GB full-500)",
        "scores": {"gpu-12gb": "granite4.1:8b 54.8 Qwen / 60.4 llama "
                               "(full-500, F-016)",
                   "gpu-8gb": "granite4.1:8b 54.8 Qwen / 60.4 llama vs prior "
                              "llama3.1:8b 49.2 / 54.4 (E-023)"},
    },
    notes="Loses on conversational and long-context workloads (LoCoMo 0.475 "
          "vs 0.515 external judge, E-024 kill criterion), so pick it only "
          "for single-fact QA. granite4.1:8b at 4.9GB q4 (Apache-2.0) is a "
          "statistical tie with the prior 12GB pick gemma4:12b (54.8/60.4 vs "
          "53.8/61.4); the VRAM headroom and license break the tie, and the "
          "same weights serve the 8GB tier. The 4GB pick stays llama3.1:8b "
          "(granite does not fit 4GB).",
))


def get_profile(profile_id: str) -> GeneratorProfile | None:
    return _REGISTRY.get(profile_id)


def list_profiles() -> list[GeneratorProfile]:
    return list(_REGISTRY.values())


def default_profile_id() -> str:
    return "balanced"


def resolve_generator(agent: str | None = None, *, fallback: str | None = None,
                      data_dir=None) -> str:
    """Resolve the active generator model string for the current tier.

    Precedence: explicit pin > active profile (per-agent, then global,
    default balanced) for the detected tier > fallback > "".
    A "" result means retrieval-only (no local generator).
    """
    pin = _config.get_memory_model(data_dir)
    if pin:
        return pin
    pid = None
    if agent:
        try:
            pid = _agents.get_agent_generator_profile(agent, data_dir=data_dir)
        except _agents.AgentNotFoundError:
            pid = None
    pid = pid or _config.get_generator_profile(data_dir) or default_profile_id()
    prof = get_profile(pid)
    if prof is not None:
        tier = recipes.tier_of(recipes.local_probe())
        if tier in prof.models:
            return prof.models[tier]
    return fallback or ""
