# Smart Installer Design

Status: design for review (decisions settled with Jay; awaiting spec sign-off before writing-plans)

## Goal

Set taOSmd up correctly for a user's specific machine and needs without making them read a tier matrix. The installer is a copy-paste agent prompt, not only a shell script: the agent probes the hardware, applies the right hardware tier automatically, recommends a feature profile, asks consent on the switches that change behaviour, configures the system, and tells the user every choice stays changeable later in the web dash.

This is the front door. Most people meet taOSmd through it, so it has to be honest (no silently-enabled behaviour), forgiving (degrades cleanly on a probe miss), and consistent with what the web dash later shows.

## Why a prompt and not just a script

A shell script can match a model to silicon, but it cannot hold a short conversation about what the user actually wants (speed vs accuracy vs auditability), explain a trade-off, or adapt the install to the answer. An agent can. We already ship `scripts/install-server.sh/.ps1` and `scripts/install-client.sh/.ps1`; the smart installer drives those scripts with a resolved config rather than replacing them.

## Delivery: both forms

Jay's call was "Both", so there are two entry points that share one registry:

1. CLI-generated, machine-tailored prompt. `taosmd setup-prompt` runs the local probe and prints a prompt already filled in with the detected specs, the chosen tier and recipe, the recommended profile, and the specific switch questions worth asking on this box. The user pastes it into their agent (Claude Code, Cowork, or any tool-capable agent).
2. Committed static prompt at `docs/INSTALL-AGENT-PROMPT.md`. A hardware-agnostic version checked into the repo for someone who finds the project before installing anything. It instructs the agent to run the probe itself, then follow the same flow. It is the fallback when the CLI is not yet available.

Both forms resolve the same way because both read `taosmd/profiles.py`.

## Shared switch and profile registry

New module `taosmd/profiles.py` is the single source of truth for what can be turned on, what it costs, and what the named bundles are. Both the prompt generator and the web dash consume it, so they never drift.

A Switch has: `id`, `label`, `category` (one of `hardware`, `quality`, `integrity`), `config_key` (the key written into `~/.taosmd/config.json`), `default`, `requires_consent` (bool), `cost` (a short human note on latency / RAM / accuracy impact), `description`, and `help` (the longer text the web dash shows).

A Profile has: `id`, `label`, `description`, and `overrides` (a mapping of switch id to value).

Switches map to capabilities that already exist or are already pre-registered:
- `rerank` (quality): bge-v2-m3 reranking. The F-013 win, with self-verify.
- `self_verify` (quality): CoVe-style answer self-verification (`TAOSMD_SELF_VERIFY`). F-013, +17.8pp Judge, judge-robust.
- `arctic_embed` (quality): arctic-embed-s dense default (F-010). Fresh installs already fetch it.
- `prefer_verified` (integrity): the claims gate (F-011). Eliminates served-hallucination at no measured accuracy cost; stays opt-in until Jay flips the default.

Hardware-category switches are not user toggles; they are the recipe selection that `recipes.recommend()` already does from the detected tier.

## Three profiles

- Minimal: fastest and lightest. Plain retrieval, no rerank, no self-verify, no claims gate. For weak hardware or speed-first users. Every consent-required switch is off here, by construction (a test enforces this).
- Quality: `rerank` + `self_verify` + `arctic_embed`. Best accuracy on capable hardware. This is the F-013 configuration.
- Integrity: Quality plus `prefer_verified`, with provenance and the audit surface called out. For users who need auditable, zero-served-hallucination memory.

## Auto versus ask

- Hardware tier: auto. `recipes.local_probe()` produces `device_info`, `recipes.tier_of()` classifies it (`gpu-12gb`/`gpu-8gb`/`gpu-4gb`/`pi-npu`/`cpu`), and `recipes.recommend()` picks the best-fitting recipe. Matching a model to the silicon needs no consent, so it is applied silently and reported.
- Quality and integrity switches: ask. These change latency, RAM, or answer behaviour, so the agent surfaces each with its `cost` and gets explicit consent before enabling. This is the "never silently enabled" rule made concrete in the install path.
- Profile: recommend one, user confirms. The recommendation is a function of tier (and any stated need): weak hardware leans Minimal, capable hardware leans Quality, a stated compliance or audit need leans Integrity. The user always confirms or changes it.

## Data flow

`local_probe()` produces `device_info` -> `tier_of()` gives the tier -> `recipes.recommend(device_info)` returns the ranked fitting recipes and the installer takes the top one, and `profiles.recommend_profile(tier, needs)` gives the suggested profile -> the prompt is emitted carrying the detected specs, the chosen tier and recipe, the recommended profile, and the ask-list of quality and integrity switches with their costs -> the agent asks the user, who confirms the profile and consents (or not) to switches -> `profiles.resolve_config(profile_id, consented_switches)` returns the final config dict -> the agent runs the existing install scripts with that config and writes the switch states into `~/.taosmd/config.json` -> the agent closes by noting the web dash can change all of it later, with guidance.

## Components and files

- Create `taosmd/profiles.py`: the Switch and Profile dataclasses, the registry, `recommend_profile(tier, needs=None)`, `resolve_config(profile_id, consented_switches)` returning a config dict, and `profiles_schema()` for the dash.
- Register a `setup-prompt` subcommand in the taOSmd CLI argument parser (the same parser that holds `a2a-poll`, `recipes`, `claims`, and the rest). It runs `local_probe()`, builds the tailored prompt text from the registry, and prints it. A `--device-info FILE` flag injects a fixed probe result for tests and for generating a prompt for a different machine.
- Create `docs/INSTALL-AGENT-PROMPT.md`: the committed static prompt.
- Tests `tests/test_profiles.py`: registry integrity (every profile references only real switch ids), `recommend_profile` per tier, `resolve_config` round-trips, and the invariant that every `requires_consent` switch is off under Minimal.
- Tests `tests/test_setup_prompt.py`: with an injected `device_info`, the emitted prompt contains the detected tier, the recommended profile, and the ask-list, and is deterministic.
- Modify the CLI entrypoint to register `setup-prompt`, and the README to point at the agent prompt as the recommended install path.

## Error handling

The probe is already best-effort and dependency-free and falls back to the `cpu` tier when the GPU sniff fails. The prompt generator must never fail hard on a probe miss: it degrades to the cpu and Minimal recommendation and says so in the prompt. If the agent cannot run the install scripts (for example a permissions problem), it reports the failure and points at the manual install docs rather than leaving a half-configured state.

## Scope and decomposition

This spec covers the registry, the CLI prompt generator, and the committed static prompt. That is one implementable plan that produces a working install path on its own.

The web-dash management surface is a separate follow-on plan. It reads `profiles_schema()` and renders each switch and profile with its help text and cost, toggleable with guidance, writing changes back to `~/.taosmd/config.json` (the global memory-model config surface already exists). It is deliberately out of this plan so the install path can ship and be tested first.

## Testing posture

The registry must be internally consistent and the consent rule must be mechanically enforced, not just documented: a Minimal profile that enabled a consent-required switch would be a correctness bug, so a test asserts it cannot. Prompt generation is deterministic given a fixed `device_info`, which is how both test files pin behaviour without touching real hardware.
