# Task-aware Generator Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a data-driven, tier-aware generator-profile registry so a user can select the answer/memory generator by workload (for example single-fact recall) without changing the safe default, covering the whole hardware range including retrieval-only on tiny devices.

**Architecture:** A new `taosmd/generator_profiles.py` registry sits alongside the retrieval recipes. Each profile maps a workload to a generator model per hardware tier (an empty string means retrieval-only). A `resolve_generator` function applies precedence pin > active profile (per-agent, then global) > caller fallback. `config.resolve_memory_model` delegates to it, so existing consumers route through profiles with no signature change, and `apply_recipe` stops auto-seeding `memory_model` so a profile is never shadowed by what looks like a user pin.

**Tech Stack:** Python 3, the existing taosmd package (config.py JSON store, agents.py registry, recipes.py tier detection, controls.py typed-control registry), pytest.

## Global Constraints

- Default profile is `balanced` and must reproduce today's per-tier recipe generators exactly. Never silently change the default generator.
- `balanced.models` values, verbatim: `gpu-12gb` and `gpu-8gb` = `ollama:qwen3.5:9b`; `gpu-4gb` = `ollama:llama3.1:8b`; `pi-npu` = `""`; `cpu` = `""`.
- `factual-recall.models` values, verbatim (8GB and 4GB confirmed by E-023 / F-015): `gpu-12gb` = `ollama:gemma4:12b`; `gpu-8gb` = `ollama:llama3.1:8b`; `gpu-4gb` = `ollama:llama3.1:8b`. No `pi-npu` or `cpu` key (falls through).
- Tier vocabulary and ordering come from `recipes.tier_of` / `recipes.local_probe`: `gpu-12gb` > `gpu-8gb` > `gpu-4gb` > `pi-npu` > `cpu`. Do not invent a new tier concept.
- Model strings are `provider:model` (for example `ollama:qwen3.5:9b`); `""` means the `none` / retrieval-only backend.
- No em dashes anywhere in code comments, docstrings, or docs. No AI attribution in commits. Commit author is jaylfc.
- Number provenance in docs: cite benchmarks.md or the research report; never invent a number.
- Scope is the `local` and `none` backends only. Do not add remote or cloud generation here.

---

### Task 1: Generator-profile registry and seeds

**Files:**
- Create: `taosmd/generator_profiles.py`
- Test: `tests/test_generator_profiles.py`

**Interfaces:**
- Consumes: `taosmd.recipes` (read-only, for the tier order constant in a later task).
- Produces: `GeneratorProfile` dataclass with fields `id: str`, `label: str`, `workload: str`, `models: dict[str, str]`, `evidence: dict`, `notes: str`; `get_profile(profile_id: str) -> GeneratorProfile | None`; `list_profiles() -> list[GeneratorProfile]`; `default_profile_id() -> str`; module constant `TIER_ORDER: tuple[str, ...]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generator_profiles.py
from taosmd import generator_profiles as gp


def test_default_is_balanced():
    assert gp.default_profile_id() == "balanced"
    assert gp.get_profile("balanced") is not None


def test_registry_integrity():
    profiles = gp.list_profiles()
    assert {p.id for p in profiles} >= {"balanced", "factual-recall"}
    for p in profiles:
        assert p.models, f"{p.id} has an empty models map"
        for tier in p.models:
            assert tier in gp.TIER_ORDER, f"{p.id} has bad tier {tier!r}"


def test_balanced_mirrors_recipe_generators():
    # backward-compat guard: balanced must equal today's per-tier recipe gens
    bal = gp.get_profile("balanced").models
    assert bal["gpu-12gb"] == "ollama:qwen3.5:9b"
    assert bal["gpu-8gb"] == "ollama:qwen3.5:9b"
    assert bal["gpu-4gb"] == "ollama:llama3.1:8b"
    assert bal["pi-npu"] == ""
    assert bal["cpu"] == ""


def test_factual_recall_tiers():
    fr = gp.get_profile("factual-recall").models
    assert fr["gpu-12gb"] == "ollama:gemma4:12b"
    assert fr["gpu-8gb"] == "ollama:llama3.1:8b"
    assert fr["gpu-4gb"] == "ollama:llama3.1:8b"
    assert "pi-npu" not in fr  # falls through to the recipe (retrieval-only)


def test_unknown_profile_is_none():
    assert gp.get_profile("nope") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_generator_profiles.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'taosmd.generator_profiles'`

- [ ] **Step 3: Write minimal implementation**

```python
# taosmd/generator_profiles.py
"""Task-aware generator profiles.

A GeneratorProfile maps a workload to the best answer/memory generator PER
hardware tier. The active profile overrides the retrieval recipe's generator at
resolution time (see resolve_generator, added in a later task). An empty model
string is the retrieval-only (no local LLM) backend. Profiles are pure data, so
a new workload-to-model mapping is one row.

Design: docs/superpowers/specs/2026-06-24-task-aware-generator-profiles-design.md
"""
from dataclasses import dataclass, field

# Highest VRAM first. Mirrors recipes.tier_of's vocabulary.
TIER_ORDER: tuple[str, ...] = ("gpu-12gb", "gpu-8gb", "gpu-4gb", "pi-npu", "cpu")


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
        "gpu-12gb": "ollama:gemma4:12b",
        "gpu-8gb": "ollama:llama3.1:8b",
        "gpu-4gb": "ollama:llama3.1:8b",
    },
    evidence={
        "source": "docs/research-report.md N-017 (12GB full-500) + "
                  "F-015 / E-023 (8 and 4GB full-500)",
        "scores": {"gpu-12gb": "53.8 Qwen / 61.4 llama (full-500)",
                   "gpu-8gb": "llama3.1:8b 49.2 Qwen / 54.4 llama (E-023)"},
    },
    notes="Loses on conversational and long-context workloads (LoCoMo 0.63 vs "
          "0.68, BEAM 46 vs 49), so pick it only for single-fact QA. The 8GB "
          "and 4GB picks are llama3.1:8b, confirmed by E-023 (beats qwen3.5:9b "
          "on the cross-family judge).",
))


def get_profile(profile_id: str) -> GeneratorProfile | None:
    return _REGISTRY.get(profile_id)


def list_profiles() -> list[GeneratorProfile]:
    return list(_REGISTRY.values())


def default_profile_id() -> str:
    return "balanced"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_generator_profiles.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add taosmd/generator_profiles.py tests/test_generator_profiles.py
git commit -m "feat(profiles): generator-profile registry with balanced + factual-recall seeds"
```

---

### Task 2: Global profile storage in config.py

**Files:**
- Modify: `taosmd/config.py` (add after `set_memory_model`, near line 115; add exports near line 482)
- Test: `tests/test_generator_profile_config.py`

**Interfaces:**
- Consumes: existing `config._read`, `config._write` (atomic JSON store).
- Produces: `config.get_generator_profile(data_dir=None) -> str | None`; `config.set_generator_profile(profile_id: str, clear: bool = False, data_dir=None) -> None`. Stored under the top-level key `generator_profile`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generator_profile_config.py
from taosmd import config


def test_generator_profile_roundtrip(tmp_path):
    assert config.get_generator_profile(data_dir=tmp_path) is None
    config.set_generator_profile("factual-recall", data_dir=tmp_path)
    assert config.get_generator_profile(data_dir=tmp_path) == "factual-recall"
    config.set_generator_profile("", clear=True, data_dir=tmp_path)
    assert config.get_generator_profile(data_dir=tmp_path) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_generator_profile_config.py -q`
Expected: FAIL with `AttributeError: module 'taosmd.config' has no attribute 'get_generator_profile'`

- [ ] **Step 3: Write minimal implementation**

Add the key constant near `_MEMORY_MODEL_KEY` (line 30):

```python
_GENERATOR_PROFILE_KEY = "generator_profile"
```

Add the functions after `set_memory_model` (after line 115):

```python
def get_generator_profile(data_dir=None) -> str | None:
    """Return the active global generator-profile id, or None if unset."""
    pid = _read(data_dir).get(_GENERATOR_PROFILE_KEY)
    if isinstance(pid, str) and pid.strip():
        return pid
    return None


def set_generator_profile(profile_id: str, clear: bool = False, data_dir=None) -> None:
    """Persist the active global generator-profile id.

    Args:
        profile_id: a registered profile id. Ignored when clear is True.
        clear: when True, remove the setting (unset).

    Raises:
        ValueError: when clear is False and profile_id is not a non-empty str.
    """
    data = _read(data_dir)
    if clear:
        data.pop(_GENERATOR_PROFILE_KEY, None)
        _write(data, data_dir)
        return
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise ValueError("profile_id must be a non-empty string")
    data[_GENERATOR_PROFILE_KEY] = profile_id.strip()
    _write(data, data_dir)
```

Add to the `__all__` list (near line 482):

```python
    "get_generator_profile",
    "set_generator_profile",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_generator_profile_config.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add taosmd/config.py tests/test_generator_profile_config.py
git commit -m "feat(profiles): global generator-profile config storage"
```

---

### Task 3: Per-agent profile override in agents.py

**Files:**
- Modify: `taosmd/agents.py` (add a class method near `set_agent_recipe_config` line 301, and module-level wrappers near line 566)
- Test: `tests/test_generator_profile_agent.py`

**Interfaces:**
- Consumes: the existing `AgentRegistry` envelope (`self._read` / `self._write`), the existing module-level `data_dir`-to-registry wrapper pattern.
- Produces: module-level `agents.get_agent_generator_profile(name: str, data_dir=None) -> str | None`; `agents.set_agent_generator_profile(name: str, profile_id: str | None, data_dir=None) -> dict`. Stored on the agent record under `generator_profile_id`. Passing `None` or `""` clears it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generator_profile_agent.py
from taosmd import agents


def test_agent_generator_profile_roundtrip(tmp_path):
    # register_agent (module-level) does not take data_dir; use the registry
    # directly against tmp_path, exactly like tests/test_agents.py does.
    agents.AgentRegistry(tmp_path).register_agent("alice")
    assert agents.get_agent_generator_profile("alice", data_dir=tmp_path) is None
    agents.set_agent_generator_profile("alice", "factual-recall", data_dir=tmp_path)
    assert agents.get_agent_generator_profile("alice", data_dir=tmp_path) == "factual-recall"
    agents.set_agent_generator_profile("alice", None, data_dir=tmp_path)
    assert agents.get_agent_generator_profile("alice", data_dir=tmp_path) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_generator_profile_agent.py -q`
Expected: FAIL with `AttributeError: module 'taosmd.agents' has no attribute 'get_agent_generator_profile'`

- [ ] **Step 3: Write minimal implementation**

Add these methods on `AgentRegistry`, right after `set_agent_recipe_config` (after line ~325):

```python
    def get_agent_generator_profile(self, name: str) -> str | None:
        """Return the per-agent generator-profile id, or None if unset."""
        rec = self.get_agent(name)
        pid = rec.get("generator_profile_id")
        return pid if isinstance(pid, str) and pid.strip() else None

    def set_agent_generator_profile(self, name: str, profile_id: str | None) -> dict:
        """Set or clear the per-agent generator-profile id (None/'' clears)."""
        data = self._read()
        for a in data["agents"]:
            if a["name"] == name:
                if profile_id and profile_id.strip():
                    a["generator_profile_id"] = profile_id.strip()
                else:
                    a.pop("generator_profile_id", None)
                self._write(data)
                return dict(a)
        raise AgentNotFoundError(f"agent {name!r} is not registered")
```

Add module-level wrappers next to the existing `get_applied_recipe` / `set_agent_recipe_config` wrappers (near line 566). The file already has a `_registry(data_dir)` helper (line 520) used by every wrapper; reuse it:

```python
def get_agent_generator_profile(name: str, data_dir=None) -> str | None:
    return _registry(data_dir).get_agent_generator_profile(name)


def set_agent_generator_profile(name: str, profile_id: str | None, data_dir=None) -> dict:
    return _registry(data_dir).set_agent_generator_profile(name, profile_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_generator_profile_agent.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add taosmd/agents.py tests/test_generator_profile_agent.py
git commit -m "feat(profiles): per-agent generator-profile override storage"
```

---

### Task 4: resolve_generator, delegate resolve_memory_model, stop the auto-seed

**Files:**
- Modify: `taosmd/generator_profiles.py` (add `resolve_generator`)
- Modify: `taosmd/config.py` (`resolve_memory_model` delegates; line 225-235)
- Modify: `taosmd/recipes.py` (`apply_recipe` stops auto-seeding; lines 399-401)
- Test: `tests/test_generator_resolution.py`

**Interfaces:**
- Consumes: `config.get_memory_model`, `config.get_generator_profile`, `agents.get_agent_generator_profile`, `recipes.tier_of`, `recipes.local_probe`, `get_profile`, `default_profile_id`.
- Produces: `generator_profiles.resolve_generator(agent: str | None = None, *, fallback: str | None = None, data_dir=None) -> str`. Precedence: explicit pin (`get_memory_model`) > active profile model for the detected tier (per-agent override, then global, default `balanced`) > `fallback` > `""`. A tier absent from the profile's `models` falls through to `fallback`. A present `""` is returned as-is (retrieval-only).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generator_resolution.py
import pytest
from taosmd import generator_profiles as gp


@pytest.fixture
def at_12gb(monkeypatch):
    monkeypatch.setattr(gp.recipes, "local_probe", lambda: {"host": {}})
    monkeypatch.setattr(gp.recipes, "tier_of", lambda info: "gpu-12gb")


def test_default_resolves_to_balanced_for_tier(at_12gb, tmp_path):
    # no pin, no profile set -> balanced -> qwen3.5:9b at 12gb
    assert gp.resolve_generator(data_dir=tmp_path) == "ollama:qwen3.5:9b"


def test_global_profile_overrides(at_12gb, tmp_path):
    from taosmd import config
    config.set_generator_profile("factual-recall", data_dir=tmp_path)
    assert gp.resolve_generator(data_dir=tmp_path) == "ollama:gemma4:12b"


def test_pin_beats_profile(at_12gb, tmp_path):
    from taosmd import config
    config.set_generator_profile("factual-recall", data_dir=tmp_path)
    config.set_memory_model("ollama:custom-pin", data_dir=tmp_path)
    assert gp.resolve_generator(data_dir=tmp_path) == "ollama:custom-pin"


def test_absent_tier_falls_through_to_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(gp.recipes, "tier_of", lambda info: "pi-npu")
    monkeypatch.setattr(gp.recipes, "local_probe", lambda: {"host": {}})
    from taosmd import config
    config.set_generator_profile("factual-recall", data_dir=tmp_path)
    # factual-recall has no pi-npu key -> fallback used
    assert gp.resolve_generator(fallback="ollama:recipe-gen", data_dir=tmp_path) == "ollama:recipe-gen"


def test_retrieval_only_empty_string(monkeypatch, tmp_path):
    monkeypatch.setattr(gp.recipes, "tier_of", lambda info: "pi-npu")
    monkeypatch.setattr(gp.recipes, "local_probe", lambda: {"host": {}})
    # balanced pi-npu == "" -> retrieval-only
    assert gp.resolve_generator(data_dir=tmp_path) == ""


def test_per_agent_beats_global(at_12gb, tmp_path):
    from taosmd import config, agents
    config.set_generator_profile("balanced", data_dir=tmp_path)
    agents.AgentRegistry(tmp_path).register_agent("bob")
    agents.set_agent_generator_profile("bob", "factual-recall", data_dir=tmp_path)
    assert gp.resolve_generator("bob", data_dir=tmp_path) == "ollama:gemma4:12b"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_generator_resolution.py -q`
Expected: FAIL with `AttributeError: module 'taosmd.generator_profiles' has no attribute 'resolve_generator'` (and `gp.recipes` missing)

- [ ] **Step 3: Write minimal implementation**

In `taosmd/generator_profiles.py`, add the imports at the top (after the docstring) and the function at the end:

```python
from . import recipes
from . import config as _config
from . import agents as _agents


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
        pid = _agents.get_agent_generator_profile(agent, data_dir=data_dir)
    pid = pid or _config.get_generator_profile(data_dir) or default_profile_id()
    prof = get_profile(pid)
    if prof is not None:
        tier = recipes.tier_of(recipes.local_probe())
        if tier in prof.models:
            return prof.models[tier]
    return fallback or ""
```

In `taosmd/config.py`, change `resolve_memory_model` (line 225-235) to delegate through the profile while preserving its `fallback`-or-None contract. Use a lazy import to avoid a circular import:

```python
def resolve_memory_model(fallback: str | None = None, data_dir=None) -> str | None:
    """Resolve the active generator model: pin > profile(tier) > fallback.

    Delegates to generator_profiles.resolve_generator (lazy import to avoid a
    cycle). Returns None when resolution yields the empty (retrieval-only)
    value AND no fallback was given, preserving the historical None contract.
    """
    from . import generator_profiles  # lazy: avoids config<->profiles cycle
    resolved = generator_profiles.resolve_generator(fallback=fallback, data_dir=data_dir)
    return resolved or None
```

In `taosmd/recipes.py`, remove the auto-seed in `apply_recipe` (lines 399-401). Replace:

```python
    gen = recipe.generator.get("model", "")
    # Only seed the global memory model from the recipe when the user has not
    # already chosen one. ...
    if gen and _config.get_memory_model(data_dir=data_dir) is None:
        _config.set_memory_model(gen, data_dir=data_dir)
    return recipe
```

with:

```python
    # The generator is no longer copied into config here. The active generator
    # profile (default "balanced") mirrors the recipe generators per tier and is
    # read live by generator_profiles.resolve_generator, so auto-seeding would
    # masquerade as a user pin and shadow a profile selection.
    return recipe
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_generator_resolution.py tests/test_generator_profiles.py -q`
Expected: PASS (all)

- [ ] **Step 5: Run the no-regression guard**

Confirm that with nothing configured, `resolve_memory_model` still returns the recipe generator for the tier (balanced mirrors it). Run:

Run: `.venv/bin/python -m pytest tests/ -q -k "recipe or memory_model or generator"`
Expected: PASS, no recipe/config regressions.

- [ ] **Step 6: Commit**

```bash
git add taosmd/generator_profiles.py taosmd/config.py taosmd/recipes.py tests/test_generator_resolution.py
git commit -m "feat(profiles): resolve_generator + delegate resolve_memory_model; stop apply_recipe auto-seed"
```

---

### Task 5: Surface the profile in the controls registry

**Files:**
- Modify: `taosmd/controls.py` (add a `generator_profile` entry to `CONTROLS`, near the other entries around line 49-123)
- Test: `tests/test_generator_profile_control.py`

**Interfaces:**
- Consumes: `generator_profiles.list_profiles`, `generator_profiles.default_profile_id`, the existing `Control` dataclass, `validate_control`, `controls_schema`.
- Produces: a `CONTROLS["generator_profile"]` entry, `type="choice"`, `choices` = the profile ids, `default="balanced"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generator_profile_control.py
from taosmd import controls


def test_generator_profile_control_present():
    c = controls.CONTROLS["generator_profile"]
    assert c.type == "choice"
    assert "balanced" in c.choices and "factual-recall" in c.choices
    assert c.default == "balanced"


def test_generator_profile_validates():
    assert controls.validate_control("generator_profile", "factual-recall") == "factual-recall"


def test_generator_profile_rejects_unknown():
    import pytest
    with pytest.raises(Exception):
        controls.validate_control("generator_profile", "nope")


def test_generator_profile_in_schema():
    schema = controls.controls_schema()
    ids = [c["id"] for c in schema] if isinstance(schema, list) else list(schema)
    assert "generator_profile" in ids or "generator_profile" in str(schema)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_generator_profile_control.py -q`
Expected: FAIL with `KeyError: 'generator_profile'`

- [ ] **Step 3: Write minimal implementation**

At the top of `taosmd/controls.py`, import the registry (after the existing imports):

```python
from . import generator_profiles as _gp
```

Add this entry inside the `CONTROLS` dict (after the last existing entry, for example after `self_verify`):

```python
    "generator_profile": Control(
        id="generator_profile", label="Generator profile",
        category="quality", scope="consumer", type="choice",
        config_key="generator_profile",
        default=_gp.default_profile_id(),
        choices=tuple(p.id for p in _gp.list_profiles()),
        cost="switches which answer model loads; gemma4:12b needs a 12GB GPU",
        pros="picks the generator that wins your workload (for example "
             "factual-recall = gemma4:12b for single-fact QA)",
        cons="factual-recall loses on conversational and long-context work; "
             "the default 'balanced' is the safe all-round choice",
        description="Select the answer generator by workload. Default 'balanced'.",
        benchmarks_anchor="end-to-end-judge-on-longmemeval-s-the-generation-side-number",
    ),
```

If `validate_control` reads choices from the `Control.choices` tuple it will accept profile ids automatically. If `controls_schema` enumerates `CONTROLS`, the new entry appears with no further change.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_generator_profile_control.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add taosmd/controls.py tests/test_generator_profile_control.py
git commit -m "feat(profiles): surface generator_profile in the controls registry"
```

---

### Task 6: CLI command `taosmd generator-profile`

**Files:**
- Modify: `taosmd/cli.py` (add handler functions near `_memory_model_get`/`_memory_model_set` line 118-133; add a subparser near line 1220; add dispatch near line 1722)
- Test: `tests/test_generator_profile_cli.py`

**Interfaces:**
- Consumes: `generator_profiles.list_profiles`, `get_profile`, `resolve_generator`, `config.get_generator_profile`, `config.set_generator_profile`, `agents.set_agent_generator_profile`.
- Produces: CLI verbs `generator-profile list`, `generator-profile show <id>`, `generator-profile set <id> [--agent A]`. Handlers return an int exit code (0 on success), matching the existing `_memory_model_get` convention.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generator_profile_cli.py
from taosmd import cli


def test_cli_list_and_set(tmp_path, capsys):
    rc = cli._generator_profile_list(data_dir=tmp_path)
    assert rc == 0
    out = capsys.readouterr().out
    assert "balanced" in out and "factual-recall" in out

    rc = cli._generator_profile_set("factual-recall", agent=None, data_dir=tmp_path)
    assert rc == 0
    from taosmd import config
    assert config.get_generator_profile(data_dir=tmp_path) == "factual-recall"


def test_cli_set_rejects_unknown(tmp_path):
    rc = cli._generator_profile_set("nope", agent=None, data_dir=tmp_path)
    assert rc != 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_generator_profile_cli.py -q`
Expected: FAIL with `AttributeError: module 'taosmd.cli' has no attribute '_generator_profile_list'`

- [ ] **Step 3: Write minimal implementation**

Add handler functions near `_memory_model_set` (after line 133). The older `_memory_model_*` handlers take no `data_dir` and use the default store; these new handlers deliberately accept an optional `data_dir=None` and pass it through to `config` / `agents`, which makes them unit-testable against a tmp_path. The CLI dispatch calls them without `data_dir`, so production still uses the default store.

```python
def _generator_profile_list(data_dir=None) -> int:
    from . import generator_profiles as gp
    from . import config
    active = config.get_generator_profile(data_dir=data_dir) or gp.default_profile_id()
    for p in gp.list_profiles():
        mark = "*" if p.id == active else " "
        print(f"{mark} {p.id:16} {p.label}")
    return 0


def _generator_profile_show(profile_id: str, data_dir=None) -> int:
    from . import generator_profiles as gp
    p = gp.get_profile(profile_id)
    if p is None:
        print(f"error: unknown profile {profile_id!r}", file=sys.stderr)
        return 1
    print(f"{p.id}: {p.label}")
    print(f"workload: {p.workload}")
    for tier, model in p.models.items():
        print(f"  {tier:9} {model or '(retrieval-only)'}")
    if p.notes:
        print(f"notes: {p.notes}")
    return 0


def _generator_profile_set(profile_id: str, agent=None, data_dir=None) -> int:
    from . import generator_profiles as gp
    from . import config, agents
    if gp.get_profile(profile_id) is None:
        print(f"error: unknown profile {profile_id!r}", file=sys.stderr)
        return 1
    if agent:
        agents.set_agent_generator_profile(agent, profile_id, data_dir=data_dir)
        print(f"agent {agent}: generator profile = {profile_id}")
    else:
        config.set_generator_profile(profile_id, data_dir=data_dir)
        print(f"global generator profile = {profile_id}")
    return 0
```

Add the subparser near the `memory-model` subparser (line ~1220):

```python
    gp_p = sub.add_parser("generator-profile", help="select the answer generator by workload")
    gp_sub = gp_p.add_subparsers(dest="generator_profile_cmd", required=True)
    gp_sub.add_parser("list", help="list profiles and mark the active one")
    gp_show = gp_sub.add_parser("show", help="show a profile's per-tier models")
    gp_show.add_argument("profile_id")
    gp_set = gp_sub.add_parser("set", help="set the active profile")
    gp_set.add_argument("profile_id")
    gp_set.add_argument("--agent", default=None, help="set per-agent instead of global")
```

Add dispatch near the `memory_model_cmd` dispatch (line ~1722):

```python
    if args.command == "generator-profile":
        if args.generator_profile_cmd == "list":
            return _generator_profile_list()
        if args.generator_profile_cmd == "show":
            return _generator_profile_show(args.profile_id)
        if args.generator_profile_cmd == "set":
            return _generator_profile_set(args.profile_id, agent=args.agent)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_generator_profile_cli.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Verify the CLI end to end**

Run: `.venv/bin/python -m taosmd.cli generator-profile list`
Expected: prints both profiles with `*` on `balanced`.

- [ ] **Step 6: Commit**

```bash
git add taosmd/cli.py tests/test_generator_profile_cli.py
git commit -m "feat(profiles): taosmd generator-profile list/show/set CLI"
```

---

### Task 7: Documentation

**Files:**
- Modify: `README.md` (add a short "Generator profiles" subsection where memory-model / recipes are documented)
- Modify: `docs/benchmarks.md` (cross-reference from the per-workload generator table near line 519-526)

**Interfaces:** none (docs only).

- [ ] **Step 1: Add the README subsection**

Add prose (no bullets unless the surrounding file uses them, no em dashes):

```markdown
### Generator profiles

The answer generator is selectable by workload, not just by hardware tier. The
default profile `balanced` (qwen3.5:9b, llama3.1:8b on a 4GB GPU) is the
multi-session and long-context leader and reproduces the previous behaviour. The
`factual-recall` profile (gemma4:12b on a 12GB GPU) wins single-fact retrieval QA
but loses on conversational and long-context workloads, so it is opt-in. Select
one with `taosmd generator-profile set <id>` (add `--agent NAME` for a single
agent), or from the dashboard Settings panel. On devices too small for a local
generator the profile resolves to retrieval-only. The 8GB and 4GB factual picks
are llama3.1:8b, confirmed by the E-023 low-tier bench (F-015 in the research
report).
```

- [ ] **Step 2: Add the benchmarks.md cross-reference**

Near the per-workload generator table (line ~519-526), add one line:

```markdown
These per-workload picks are now selectable as generator profiles (balanced,
factual-recall); see the README "Generator profiles" section and
docs/superpowers/specs/2026-06-24-task-aware-generator-profiles-design.md.
```

- [ ] **Step 3: Voice check**

Run: `grep -cP "\x{2014}" README.md docs/benchmarks.md`
Expected: the counts must not increase versus before this task (benchmarks.md has pre-existing em dashes; do not add any). Confirm your added lines contain none.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/benchmarks.md
git commit -m "docs(profiles): document generator profiles in README + benchmarks"
```

---

### Task 8: Full suite and follow-up note

**Files:** none (verification + a tracking note)

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS (all existing tests plus the new ones). If any pre-existing test asserted the old `apply_recipe` auto-seed behaviour, update it to assert `resolve_memory_model` resolves via the balanced profile instead (the value is identical for a default install).

- [ ] **Step 2: Record the E-023 dependency**

E-023 has LANDED and resolved to F-015 (report rev 1.48): it confirmed the 8GB and 4GB factual picks as llama3.1:8b, so the `factual-recall.models` values in Task 1 are already final, the "provisional" wording is already removed, and no data change is needed. qwen3:4b was recorded INVALID (self-verify degeneration) and gemma4:e4b weaker, so neither enters the registry. This step is therefore informational only; nothing to do.

- [ ] **Step 3: Commit any test fixups**

```bash
git add -A
git commit -m "test(profiles): reconcile suite with profile-based generator resolution"
```
