# Smart Installer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the install path from the approved spec: a shared switch/profile registry, a CLI that prints a hardware-tailored agent install prompt, and a committed static prompt, so a user sets taOSmd up correctly for their machine without reading a tier matrix.

**Architecture:** A single source of truth `taosmd/profiles.py` holds the Switch and Profile dataclasses, the registry, and the resolve/recommend/schema functions. A thin `taosmd/setup_prompt.py` renders a deterministic prompt from a probed `device_info` by reusing `recipes.local_probe/tier_of/recommend` plus the profile registry. The `taosmd setup-prompt` CLI subcommand wires the probe to the renderer and prints it. A committed `docs/INSTALL-AGENT-PROMPT.md` is the hardware-agnostic fallback. Hardware tier is applied automatically; quality and integrity switches are surfaced for explicit consent (never silently enabled).

**Tech Stack:** Python 3 stdlib only (dataclasses, argparse, json), pytest. No new dependencies. Reuses `taosmd/recipes.py` and `taosmd/config.py`.

---

## Spec

Source spec: `docs/superpowers/specs/2026-06-16-smart-installer-design.md`. This plan covers the registry, the CLI prompt generator, and the committed static prompt. The web-dash management surface is explicitly OUT (a separate follow-on plan).

## File Structure

- **Create `taosmd/profiles.py`** - the registry and policy. `Switch` + `Profile` dataclasses, the `SWITCHES` and `PROFILES` registries, accessors (`list_switches`, `get_switch`, `get_profile`, `list_profiles`), `recommend_profile(tier, needs)`, `resolve_config(profile_id, consented_switches)`, and `profiles_schema()`. No I/O, no printing. One responsibility: what can be turned on, what it costs, what the bundles are.
- **Create `taosmd/setup_prompt.py`** - pure prompt rendering. `render_setup_prompt(device_info, needs=None)` -> `str`, reusing `recipes.tier_of/recommend` and `profiles`. Deterministic given inputs. No probing, no printing (the CLI does those), so it is trivially testable.
- **Modify `taosmd/cli.py`** - register the `setup-prompt` subcommand in `_build_parser()` (around the other `sub.add_parser(...)` calls, ~line 1237 next to `install-skill`), add the `_setup_prompt_cmd(args)` handler (near the other `_*_cmd` handlers), and add one dispatch line in `main()` (~line 1508, next to `install-skill`).
- **Create `docs/INSTALL-AGENT-PROMPT.md`** - the committed static prompt: hardware-agnostic, instructs the agent to run the probe itself then follow the same flow.
- **Modify `README.md`** - point the install section at the agent prompt as the recommended path.
- **Create `tests/test_profiles.py`** - registry integrity, the Minimal consent invariant, `recommend_profile` per tier, `resolve_config` round-trips, schema shape.
- **Create `tests/test_setup_prompt.py`** - with an injected `device_info`, the prompt contains the detected tier, the recommended profile, the ask-list, and is deterministic; probe-miss degrades to cpu/Minimal.

Run all tests with: `python -m pytest tests/test_profiles.py tests/test_setup_prompt.py -v` (the repo uses pytest; `.venv/bin/python -m pytest` on hosts with the venv).

---

### Task 1: Switch and Profile dataclasses + registry + accessors

**Files:**
- Create: `taosmd/profiles.py`
- Test: `tests/test_profiles.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_profiles.py
from taosmd import profiles


def test_every_profile_references_only_real_switch_ids():
    valid = set(profiles.SWITCHES)
    for prof in profiles.PROFILES.values():
        for sid in prof.overrides:
            assert sid in valid, f"profile {prof.id} references unknown switch {sid}"


def test_minimal_profile_enables_no_consent_required_switch():
    minimal = profiles.PROFILES["minimal"]
    for sw in profiles.SWITCHES.values():
        if sw.requires_consent:
            assert minimal.overrides.get(sw.id, False) is False, (
                f"Minimal must not enable consent-required switch {sw.id}"
            )


def test_switch_categories_are_valid():
    allowed = {"hardware", "quality", "integrity"}
    for sw in profiles.SWITCHES.values():
        assert sw.category in allowed


def test_accessors_round_trip():
    assert profiles.get_switch("rerank").id == "rerank"
    assert profiles.get_switch("nope") is None
    assert profiles.get_profile("quality").id == "quality"
    assert {s.id for s in profiles.list_switches()} == set(profiles.SWITCHES)
    assert {p.id for p in profiles.list_profiles()} == set(profiles.PROFILES)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_profiles.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'taosmd.profiles'`

- [ ] **Step 3: Write minimal implementation**

```python
# taosmd/profiles.py
"""Single source of truth for installer switches and profiles.

A Switch is one capability that can be turned on, with what it costs and the
config key it writes. A Profile is a named bundle of switch states. Both the
setup-prompt generator and (later) the web dashboard read this module, so they
never drift. No I/O here: this is policy and data only.
"""
from dataclasses import dataclass, field, asdict


@dataclass
class Switch:
    id: str
    label: str
    category: str          # "hardware" | "quality" | "integrity"
    config_key: str        # dotted key written into ~/.taosmd/config.json
    on_value: object       # value written when the switch is enabled
    default: bool          # default enabled state on a fresh install
    requires_consent: bool  # quality/integrity switches must be asked
    cost: str              # short human note: latency / RAM / accuracy impact
    description: str       # one-line summary
    help: str              # longer text the dashboard shows


@dataclass
class Profile:
    id: str
    label: str
    description: str
    overrides: dict = field(default_factory=dict)  # {switch_id: bool}


# Quality + integrity switches. Hardware-tier selection is NOT a switch here;
# it is the recipe choice that recipes.recommend() already makes from the tier.
SWITCHES = {
    "arctic_embed": Switch(
        id="arctic_embed", label="Arctic-embed dense embedder", category="quality",
        config_key="vector_memory.embed_model", on_value="arctic-embed-s",
        default=True, requires_consent=False,
        cost="same dim and latency as the old default; a free retrieval upgrade",
        description="Snowflake arctic-embed-s as the dense embedder (F-010).",
        help="Beats all-MiniLM-L6-v2 by +0.157 R@K at the same 384 dimensions and "
             "latency. Fresh installs already fetch it; this records the choice.",
    ),
    "rerank": Switch(
        id="rerank", label="Cross-encoder reranking", category="quality",
        config_key="retrieval.reranker", on_value="bge-v2-m3",
        default=False, requires_consent=True,
        cost="adds a reranker pass per query (latency + a model download)",
        description="bge-v2-m3 reranking of retrieved candidates (the F-013 win).",
        help="Re-scores the candidate pool with a cross-encoder before generation. "
             "Part of the F-013 configuration. Costs one extra model and a per-query pass.",
    ),
    "self_verify": Switch(
        id="self_verify", label="Answer self-verification", category="quality",
        config_key="answer.self_verify", on_value=True,
        default=False, requires_consent=True,
        cost="roughly doubles answer latency (a second LLM pass)",
        description="CoVe-style answer self-verification (the dominant F-013 lever).",
        help="After drafting an answer, the generator verifies it against the evidence "
             "and abstains when unsupported. +17.8pp Judge on LongMemEval-S, judge-robust. "
             "Costs a second generation pass per answer.",
    ),
    "prefer_verified": Switch(
        id="prefer_verified", label="Verified-memory recall gate", category="integrity",
        config_key="claims.prefer_verified", on_value=True,
        default=False, requires_consent=True,
        cost="adds an entailment verify-pass at write time; no measured accuracy cost",
        description="The claims gate (F-011): prefer entailment-verified memories on recall.",
        help="Eliminates served-hallucination (0.040 -> 0.000) at no measured accuracy cost "
             "on LoCoMo. Demotes (never deletes) unverified claims at serve time. For users "
             "who need auditable, zero-served-hallucination memory.",
    ),
}

PROFILES = {
    "minimal": Profile(
        id="minimal", label="Minimal",
        description="Fastest and lightest: plain retrieval, no rerank, no self-verify, "
                    "no claims gate. For weak hardware or speed-first users.",
        overrides={"arctic_embed": True, "rerank": False, "self_verify": False,
                   "prefer_verified": False},
    ),
    "quality": Profile(
        id="quality", label="Quality",
        description="Best accuracy on capable hardware: arctic-embed + rerank + "
                    "self-verify. This is the F-013 configuration.",
        overrides={"arctic_embed": True, "rerank": True, "self_verify": True,
                   "prefer_verified": False},
    ),
    "integrity": Profile(
        id="integrity", label="Integrity",
        description="Quality plus the verified-memory recall gate, with provenance and "
                    "the audit surface. For auditable, zero-served-hallucination memory.",
        overrides={"arctic_embed": True, "rerank": True, "self_verify": True,
                   "prefer_verified": True},
    ),
}


def list_switches() -> list[Switch]:
    return list(SWITCHES.values())


def get_switch(switch_id: str) -> Switch | None:
    return SWITCHES.get(switch_id)


def list_profiles() -> list[Profile]:
    return list(PROFILES.values())


def get_profile(profile_id: str) -> Profile | None:
    return PROFILES.get(profile_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_profiles.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add taosmd/profiles.py tests/test_profiles.py
git commit -m "feat(profiles): switch/profile registry with consent-aware Minimal invariant"
```

---

### Task 2: recommend_profile, resolve_config, profiles_schema

**Files:**
- Modify: `taosmd/profiles.py`
- Test: `tests/test_profiles.py`

- [ ] **Step 1: Write the failing test (append to tests/test_profiles.py)**

```python
def test_recommend_profile_by_tier():
    assert profiles.recommend_profile("cpu") == "minimal"
    assert profiles.recommend_profile("pi-npu") == "minimal"
    assert profiles.recommend_profile("gpu-4gb") == "quality"
    assert profiles.recommend_profile("gpu-8gb") == "quality"
    assert profiles.recommend_profile("gpu-12gb") == "quality"


def test_recommend_profile_needs_override_to_integrity():
    # A stated audit/compliance need leans Integrity regardless of tier.
    assert profiles.recommend_profile("gpu-12gb", needs="we need an audit trail") == "integrity"
    assert profiles.recommend_profile("cpu", needs="compliance and provenance") == "integrity"


def test_resolve_config_minimal_enables_no_consent_switch():
    cfg = profiles.resolve_config("minimal", consented_switches=[])
    # arctic_embed is a free default (no consent), so it is present; the
    # consent-required keys are absent.
    assert cfg.get("vector_memory.embed_model") == "arctic-embed-s"
    assert "retrieval.reranker" not in cfg
    assert "answer.self_verify" not in cfg
    assert "claims.prefer_verified" not in cfg


def test_resolve_config_consent_required_switch_needs_explicit_consent():
    # Quality wants rerank+self_verify, but without consent they stay OFF.
    cfg_no = profiles.resolve_config("quality", consented_switches=[])
    assert "retrieval.reranker" not in cfg_no
    assert "answer.self_verify" not in cfg_no
    # With consent they are written with their on_value.
    cfg_yes = profiles.resolve_config("quality", consented_switches=["rerank", "self_verify"])
    assert cfg_yes["retrieval.reranker"] == "bge-v2-m3"
    assert cfg_yes["answer.self_verify"] is True


def test_resolve_config_unknown_profile_raises():
    import pytest
    with pytest.raises(ValueError):
        profiles.resolve_config("nope", consented_switches=[])


def test_profiles_schema_lists_switches_and_profiles():
    schema = profiles.profiles_schema()
    assert {s["id"] for s in schema["switches"]} == set(profiles.SWITCHES)
    assert {p["id"] for p in schema["profiles"]} == set(profiles.PROFILES)
    # each switch row carries the dashboard-facing fields
    row = next(s for s in schema["switches"] if s["id"] == "self_verify")
    assert row["requires_consent"] is True
    assert row["cost"] and row["help"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_profiles.py -v`
Expected: FAIL with `AttributeError: module 'taosmd.profiles' has no attribute 'recommend_profile'`

- [ ] **Step 3: Write minimal implementation (append to taosmd/profiles.py)**

```python
# Keywords in a stated need that lean the recommendation to Integrity.
_INTEGRITY_NEED_HINTS = ("audit", "compliance", "complian", "provenance",
                         "integrity", "regulat", "gdpr", "hipaa")


def recommend_profile(tier: str, needs: str | None = None) -> str:
    """Recommend a profile id from the hardware tier and any stated needs.

    A stated audit/compliance need wins (-> integrity). Otherwise capable GPU
    hardware leans quality and weak hardware leans minimal. The user always
    confirms or changes the recommendation.
    """
    if needs:
        low = needs.lower()
        if any(h in low for h in _INTEGRITY_NEED_HINTS):
            return "integrity"
    if tier in ("gpu-4gb", "gpu-8gb", "gpu-12gb"):
        return "quality"
    return "minimal"


def resolve_config(profile_id: str, consented_switches: list[str] | None = None) -> dict:
    """Return the final config dict for a profile plus the consented switches.

    Non-consent switches follow the profile (or their default). Consent-required
    switches are written ONLY when explicitly present in consented_switches, even
    if the profile wants them. This is the "never silently enabled" rule.
    """
    prof = PROFILES.get(profile_id)
    if prof is None:
        raise ValueError(f"unknown profile: {profile_id}")
    consented = set(consented_switches or [])
    out: dict = {}
    for sid, sw in SWITCHES.items():
        wants = prof.overrides.get(sid, sw.default)
        if sw.requires_consent:
            enabled = sid in consented and wants
        else:
            enabled = wants
        if enabled:
            out[sw.config_key] = sw.on_value
    return out


def profiles_schema() -> dict:
    """Renderable description of the registry for the dashboard / prompt."""
    return {
        "switches": [
            {
                "id": s.id, "label": s.label, "category": s.category,
                "config_key": s.config_key, "default": s.default,
                "requires_consent": s.requires_consent, "cost": s.cost,
                "description": s.description, "help": s.help,
            }
            for s in SWITCHES.values()
        ],
        "profiles": [
            {"id": p.id, "label": p.label, "description": p.description,
             "overrides": dict(p.overrides)}
            for p in PROFILES.values()
        ],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_profiles.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add taosmd/profiles.py tests/test_profiles.py
git commit -m "feat(profiles): recommend_profile, resolve_config (consent-gated), profiles_schema"
```

---

### Task 3: Deterministic prompt renderer

**Files:**
- Create: `taosmd/setup_prompt.py`
- Test: `tests/test_setup_prompt.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_setup_prompt.py
from taosmd import setup_prompt


GPU12 = {"host": {"cpu": {"arch": "x86_64", "cores": 16}, "ram_mb": 64000,
                  "npu": {"type": "none"},
                  "gpu": {"type": "cuda", "name": "RTX 3060", "vram_mb": 12288}}}
CPU = {"host": {"cpu": {"arch": "arm64", "cores": 8}, "ram_mb": 8000,
                "npu": {"type": "none"}, "gpu": {"type": "none", "vram_mb": 0}}}


def test_prompt_is_deterministic_for_fixed_device_info():
    a = setup_prompt.render_setup_prompt(GPU12)
    b = setup_prompt.render_setup_prompt(GPU12)
    assert a == b


def test_gpu_prompt_names_tier_recommended_profile_and_asklist():
    text = setup_prompt.render_setup_prompt(GPU12)
    assert "gpu-12gb" in text
    # capable hardware -> Quality recommended
    assert "Quality" in text
    # the consent-required switches appear as an ask-list with their cost
    assert "rerank" in text.lower()
    assert "self-verification" in text.lower() or "self_verify" in text
    # a free (non-consent) switch is reported, not asked
    assert "arctic" in text.lower()


def test_cpu_prompt_recommends_minimal():
    text = setup_prompt.render_setup_prompt(CPU)
    assert "cpu" in text
    assert "Minimal" in text


def test_stated_need_leans_integrity():
    text = setup_prompt.render_setup_prompt(GPU12, needs="we need an audit trail")
    assert "Integrity" in text


def test_probe_miss_degrades_to_cpu_minimal():
    # An empty / malformed device_info must not raise; it degrades.
    text = setup_prompt.render_setup_prompt({})
    assert "cpu" in text
    assert "Minimal" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_setup_prompt.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'taosmd.setup_prompt'`

- [ ] **Step 3: Write minimal implementation**

```python
# taosmd/setup_prompt.py
"""Render the hardware-tailored install agent prompt.

Pure and deterministic: given a device_info dict (and optional stated needs),
return the prompt text. The CLI handles probing and printing. Reuses
recipes.tier_of/recommend for the hardware tier and profiles for the bundles.
"""
from . import recipes
from . import profiles


def _safe_tier(device_info: dict) -> str:
    try:
        return recipes.tier_of(device_info or {})
    except Exception:
        return "cpu"


def _top_recipe(device_info: dict):
    try:
        ranked = recipes.recommend(device_info or {"host": {}})
        return ranked[0] if ranked else None
    except Exception:
        return None


def render_setup_prompt(device_info: dict, needs: str | None = None) -> str:
    tier = _safe_tier(device_info)
    profile_id = profiles.recommend_profile(tier, needs)
    profile = profiles.get_profile(profile_id)
    recipe = _top_recipe(device_info)
    recipe_line = f"{recipe.id} ({recipe.name})" if recipe else "the default cpu recipe"

    host = (device_info or {}).get("host", {})
    cpu = host.get("cpu", {})
    gpu = host.get("gpu", {})
    specs = (
        f"cpu={cpu.get('arch', 'unknown')} x{cpu.get('cores', '?')}, "
        f"ram_mb={host.get('ram_mb', 'unknown')}, "
        f"gpu={gpu.get('name') or gpu.get('type') or 'none'} "
        f"(vram_mb={gpu.get('vram_mb', 0)})"
    )

    # The ask-list: consent-required switches the recommended profile wants on.
    ask = [
        sw for sid, sw in profiles.SWITCHES.items()
        if sw.requires_consent and profile.overrides.get(sid, False)
    ]
    ask_lines = "\n".join(
        f"  - {sw.label} ({sw.id}): {sw.description} COST: {sw.cost}"
        for sw in ask
    ) or "  - (none for this profile)"

    # Free, non-consent switches that will be applied and merely reported.
    auto = [
        sw for sid, sw in profiles.SWITCHES.items()
        if not sw.requires_consent and profile.overrides.get(sid, sw.default)
    ]
    auto_lines = "\n".join(f"  - {sw.label} ({sw.id}): {sw.description}" for sw in auto) \
        or "  - (none)"

    return f"""\
You are setting up taOSmd (a local-first AI memory system) on this machine. Follow these steps.

DETECTED HARDWARE
  tier: {tier}
  specs: {specs}
  best-fitting recipe (apply automatically, no consent needed): {recipe_line}

RECOMMENDED PROFILE: {profile.label}
  {profile.description}

APPLY AUTOMATICALLY (hardware tier + free upgrades, just report them to the user):
{auto_lines}

ASK THE USER before enabling each of these (state the COST, get explicit yes/no):
{ask_lines}

STEPS
  1. Confirm the recommended profile with the user, or switch to Minimal / Quality / Integrity on their preference.
  2. For each switch in the ASK list, surface its cost and get explicit consent. Never enable a consent-required switch without a yes.
  3. Run the existing install scripts (scripts/install-server.sh or scripts/install-client.sh, .ps1 on Windows) and write the resolved config into ~/.taosmd/config.json.
  4. Tell the user every choice stays changeable later in the web dashboard, and how to re-run this prompt.

If a step fails (for example a permissions problem running the install scripts), report the failure and point the user at the manual install docs rather than leaving a half-configured state.
"""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_setup_prompt.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add taosmd/setup_prompt.py tests/test_setup_prompt.py
git commit -m "feat(setup-prompt): deterministic hardware-tailored install prompt renderer"
```

---

### Task 4: Wire the `setup-prompt` CLI subcommand

**Files:**
- Modify: `taosmd/cli.py` (parser registration in `_build_parser()`, a `_setup_prompt_cmd` handler, one dispatch line in `main()`)
- Test: `tests/test_setup_prompt.py` (append a CLI-level test)

- [ ] **Step 1: Write the failing test (append to tests/test_setup_prompt.py)**

```python
import json
from taosmd import cli


def test_cli_setup_prompt_with_injected_device_info(tmp_path, capsys):
    di = tmp_path / "device.json"
    di.write_text(json.dumps(
        {"host": {"cpu": {"arch": "x86_64", "cores": 16}, "ram_mb": 64000,
                  "npu": {"type": "none"},
                  "gpu": {"type": "cuda", "name": "RTX 3060", "vram_mb": 12288}}}))
    rc = cli.main(["setup-prompt", "--device-info", str(di)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "gpu-12gb" in out
    assert "Quality" in out


def test_cli_setup_prompt_needs_flag(tmp_path, capsys):
    di = tmp_path / "device.json"
    di.write_text(json.dumps({"host": {}}))
    rc = cli.main(["setup-prompt", "--device-info", str(di), "--needs", "audit trail"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Integrity" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_setup_prompt.py -v`
Expected: FAIL with `SystemExit: 2` (argparse rejects the unknown `setup-prompt` subcommand)

- [ ] **Step 3a: Add the parser registration in `_build_parser()`**

In `taosmd/cli.py`, immediately after the `install-skill` subparser block (the `install_skill_p = sub.add_parser("install-skill", ...)` group near line 1237), add:

```python
    setup_prompt_p = sub.add_parser(
        "setup-prompt",
        help="Print a hardware-tailored agent prompt for installing taOSmd",
    )
    setup_prompt_p.add_argument(
        "--device-info", metavar="FILE", default=None,
        help="Read device_info JSON from FILE instead of probing (for tests / "
             "generating a prompt for a different machine)",
    )
    setup_prompt_p.add_argument(
        "--needs", default=None,
        help="A short note on the user's stated needs (e.g. 'audit trail'); "
             "leans the recommended profile",
    )
```

- [ ] **Step 3b: Add the handler near the other `_*_cmd` functions** (for example after `_install_skill_cmd`):

```python
def _setup_prompt_cmd(args: argparse.Namespace) -> int:
    import json
    from . import recipes
    from . import setup_prompt

    if getattr(args, "device_info", None):
        try:
            with open(args.device_info, "r", encoding="utf-8") as fh:
                device_info = json.load(fh)
        except Exception as exc:
            print(f"error: could not read --device-info {args.device_info}: {exc}")
            return 1
    else:
        device_info = recipes.local_probe()

    print(setup_prompt.render_setup_prompt(device_info, getattr(args, "needs", None)))
    return 0
```

- [ ] **Step 3c: Add the dispatch line in `main()`** (next to the `install-skill` dispatch, ~line 1508):

```python
    if args.cmd == "setup-prompt":
        return _setup_prompt_cmd(args)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_setup_prompt.py -v`
Expected: PASS (all tests, including the two CLI tests)

- [ ] **Step 5: Commit**

```bash
git add taosmd/cli.py tests/test_setup_prompt.py
git commit -m "feat(cli): setup-prompt subcommand (probe + render, --device-info for tests)"
```

---

### Task 5: Committed static prompt

**Files:**
- Create: `docs/INSTALL-AGENT-PROMPT.md`

- [ ] **Step 1: Write the file**

```markdown
# Install taOSmd with an agent

This is the hardware-agnostic install prompt. Paste it into a tool-capable agent
(Claude Code, Cowork, or any agent that can run shell commands). If you have
already cloned the repo and can run the CLI, prefer the machine-tailored version:
`taosmd setup-prompt` prints a prompt already filled in with your detected
hardware. Use this static version when the CLI is not yet available.

---

You are setting up taOSmd (a local-first AI memory system) on this machine.

1. Detect the hardware. Run `python -c "import json; from taosmd import recipes; print(json.dumps(recipes.local_probe()))"` if the package is importable; otherwise inspect CPU, RAM, and GPU (and VRAM) yourself. Classify the tier: gpu-12gb / gpu-8gb / gpu-4gb / pi-npu / cpu. If detection fails, assume the cpu tier.

2. Recommend a profile and confirm it with the user:
   - Minimal: fastest and lightest. Plain retrieval, no rerank, no self-verify, no claims gate. For weak hardware or speed-first users.
   - Quality: arctic-embed + cross-encoder rerank + answer self-verification. Best accuracy on capable hardware.
   - Integrity: Quality plus the verified-memory recall gate, with provenance and the audit surface. For auditable, zero-served-hallucination memory.
   Lean Minimal on cpu/pi hardware, Quality on GPU hardware, and Integrity if the user states an audit or compliance need. The user always confirms or changes it.

3. Apply the hardware tier automatically (match the model to the silicon, no consent needed) and report it.

4. For every quality or integrity switch the profile turns on, ASK first and state the cost before enabling. Never enable one silently:
   - Cross-encoder reranking (rerank): adds a reranker pass per query (latency + a model download).
   - Answer self-verification (self_verify): roughly doubles answer latency (a second LLM pass).
   - Verified-memory recall gate (prefer_verified): adds an entailment verify-pass at write time; eliminates served-hallucination at no measured accuracy cost.
   Arctic-embed is a free, same-latency upgrade applied without asking; just report it.

5. Run the install scripts (`scripts/install-server.sh` or `scripts/install-client.sh`, `.ps1` on Windows) with the resolved choices, and write the switch states into `~/.taosmd/config.json`.

6. Close by telling the user every choice stays changeable later in the web dashboard, and how to re-run this setup.

If the install scripts fail (for example a permissions problem), report the failure and point the user at the manual install docs rather than leaving a half-configured state.
```

- [ ] **Step 2: Verify it renders and references real scripts**

Run: `ls scripts/install-server.sh scripts/install-client.sh && head -5 docs/INSTALL-AGENT-PROMPT.md`
Expected: both scripts exist and the doc shows its title. If a script path differs, correct the doc to the real path.

- [ ] **Step 3: Commit**

```bash
git add docs/INSTALL-AGENT-PROMPT.md
git commit -m "docs: committed static install-agent prompt (hardware-agnostic fallback)"
```

---

### Task 6: Point the README at the agent prompt

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Find the install section**

Run: `grep -n -i "install" README.md | head`
Identify the heading where installation is described.

- [ ] **Step 2: Add the recommended-path pointer**

Immediately under the install heading, add this paragraph (adjust surrounding prose to fit; keep taOSmd casing, no em dashes):

```markdown
**Recommended: let an agent set it up for your machine.** Run `taosmd setup-prompt`
to print an install prompt already tailored to your detected hardware, then paste
it into a tool-capable agent (Claude Code, Cowork, or similar). It picks the right
model for your silicon, recommends a feature profile, and asks before enabling
anything that changes latency or behaviour. If you have not cloned the repo yet,
use the static prompt in [docs/INSTALL-AGENT-PROMPT.md](docs/INSTALL-AGENT-PROMPT.md).
The manual steps below still work for anyone who prefers them.
```

- [ ] **Step 3: Verify**

Run: `grep -n "setup-prompt" README.md`
Expected: the new pointer line is present.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(readme): recommend the agent install prompt as the default path"
```

---

## Self-Review

**Spec coverage:**
- Shared registry `taosmd/profiles.py` (Switch + Profile + registry + recommend_profile + resolve_config + profiles_schema): Tasks 1-2. Covered.
- CLI-generated tailored prompt (`taosmd setup-prompt`, `--device-info` for tests): Tasks 3-4. Covered.
- Committed static prompt `docs/INSTALL-AGENT-PROMPT.md`: Task 5. Covered.
- Auto hardware-tier (reuse local_probe/tier_of/recommend) + ask-on-quality/integrity consent: Task 3 renderer (auto list vs ask list) + Task 2 resolve_config consent gating. Covered.
- Three profiles Minimal/Quality/Integrity with the Minimal-consent invariant enforced by a test: Tasks 1-2. Covered.
- Tests `tests/test_profiles.py` + `tests/test_setup_prompt.py`: Tasks 1-4. Covered.
- README pointer: Task 6. Covered.
- Error handling: probe miss degrades to cpu/Minimal (Task 3 `_safe_tier`/`_top_recipe` + the degrade test); install-script failure guidance is in both prompts. Covered.
- Web-dash management: OUT of scope by the spec (profiles_schema() is provided for it but the surface is a follow-on). Correct.

**Placeholder scan:** No TBDs; every code step shows complete code; every command has an expected result.

**Type consistency:** `Switch` fields (id, label, category, config_key, on_value, default, requires_consent, cost, description, help) and `Profile` fields (id, label, description, overrides) are used identically in Tasks 1-3 and the schema. `recommend_profile(tier, needs)` and `resolve_config(profile_id, consented_switches)` signatures match between definition (Task 2) and callers (Task 3 renderer, Task 4 CLI). `render_setup_prompt(device_info, needs=None)` matches between definition (Task 3) and caller (Task 4).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-17-smart-installer.md`. Two execution options:

1. **Subagent-Driven (recommended)** - a fresh subagent per task, spec-compliance then code-quality review between tasks, fast iteration.
2. **Inline Execution** - execute tasks in this session with checkpoints for review.

Branch off origin/master as `feat/smart-installer`; PR at the end, do not self-merge.
