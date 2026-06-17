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
