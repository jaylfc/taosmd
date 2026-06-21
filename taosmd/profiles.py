"""Single source of truth for installer switches and profiles.

A Switch is one capability that can be turned on, with what it costs and the
config key it writes. A Profile is a named bundle of switch states. Both the
setup-prompt generator and (later) the web dashboard read this module, so they
never drift. No I/O here: this is policy and data only.
"""
from dataclasses import dataclass, field


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
    off_value: object = None  # value written when disabled; None = write nothing


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
        config_key="controls.prefer_verified", on_value="prefer_verified", off_value="off",
        default=True, requires_consent=False,
        cost="no query-time cost; a safe no-op until claims are verified",
        description="The recall gate (F-011): prefer entailment-verified memories on recall. On by default.",
        help="Demotes (never deletes) claims verified as unsupported out of default recall. "
             "Eliminates served-hallucination (0.040 -> 0.000) at no measured accuracy cost, "
             "tri-judge confirmed (E-018). Ships on by default because it is a safe no-op until "
             "the verify-pass is populated; Minimal and Quality turn it off, Integrity keeps it on. "
             "Writes the same controls.prefer_verified the runtime recall gate reads.",
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
        elif sw.off_value is not None:
            # A switch whose runtime default is on (e.g. the recall gate) must be
            # written off explicitly when a profile opts out, or the runtime
            # default would silently re-enable it.
            out[sw.config_key] = sw.off_value
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
                "on_value": s.on_value, "off_value": s.off_value,
            }
            for s in SWITCHES.values()
        ],
        "profiles": [
            {"id": p.id, "label": p.label, "description": p.description,
             "overrides": dict(p.overrides)}
            for p in PROFILES.values()
        ],
    }
