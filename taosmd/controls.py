"""Single source of truth for taOSmd memory controls.

A Control is one user-facing lever: what it does, its type and allowed values,
its default, its scope, and its cost (pros, cons, resource impact). The standalone
dashboard, the GET/PUT /controls API, the persisted config store, the README
documentation, and the taOS app settings UI all derive from this module, so they
never drift. No I/O here: policy and data only.

Scope:
  runtime   per-query / retrieval behaviour; safe to toggle live, takes effect
            on the next search. prefer_verified is read from the controls config
            by the recall gate; reranker, fusion, and adjacent_turns override the
            active recipe per query. (prefer_verified, reranker, fusion,
            adjacent_turns)
  store     a store-level choice; changing it requires re-processing existing
            memories (a re-index/re-embed), so the dashboard shows it as the
            current install choice, not a live toggle. The token-level vectors
            for late-interaction are written at ingest, so it is store-scope too.
            (embedder, binary_quant, late_interaction)
  consumer  applied in the consumer's answer-generation, not in taOSmd core
            (taOSmd retrieves; it does not generate answers). Shown as an
            informational recommendation, never as a switch that does nothing.
            (self_verify)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import generator_profiles as _gp


@dataclass(frozen=True)
class Control:
    id: str
    label: str
    category: str           # "hardware" | "quality" | "integrity"
    scope: str              # "runtime" | "store" | "consumer"
    type: str               # "bool" | "choice" | "int"
    config_key: str         # dotted key under config.json "controls"
    default: object         # default value
    choices: tuple = ()     # allowed values for type=="choice"
    int_range: tuple = ()   # (min, max) inclusive for type=="int"
    cost: str = ""          # short resource note (latency / RAM / download)
    pros: str = ""          # when to turn it on
    cons: str = ""          # the trade-off
    description: str = ""    # one-line summary
    benchmarks_anchor: str = ""  # docs/benchmarks.md section anchor


CONTROLS: dict[str, Control] = {
    "prefer_verified": Control(
        id="prefer_verified", label="Verified-memory recall gate",
        category="integrity", scope="runtime", type="choice",
        config_key="controls.prefer_verified", default="prefer_verified",
        choices=("off", "prefer_verified", "strict"),
        cost="no query-time cost; needs the verify-pass populated to act (safe no-op otherwise)",
        pros="eliminates served-hallucination (0.040 to 0.000) at no measured accuracy cost, tri-judge confirmed (E-018); auditable, zero-served-hallucination recall",
        cons="acts only on claims you have verified; strict additionally drops unverified claims and over-trades recall, so it stays opt-in",
        description="Demote claims verified as unsupported out of default recall. On by default.",
        benchmarks_anchor="end-to-end-judge-on-longmemeval-s-the-generation-side-number",
    ),
    "reranker": Control(
        id="reranker", label="Cross-encoder reranking",
        category="quality", scope="runtime", type="choice",
        config_key="controls.reranker", default="off",
        choices=("off", "bge-v2-m3"),
        cost="one extra model download (bge-reranker-v2-m3) plus a cross-encoder pass per query",
        pros="the F-013 accuracy win where the hardware affords it (a GPU box, or any tier with headroom)",
        cons="adds per-query latency and a model; drop it on a Pi-class CPU tier",
        description="Re-score the candidate pool with a cross-encoder before serving. Overrides the recipe per query.",
        benchmarks_anchor="locomo--multi-session-conversational-memory-1540-qas",
    ),
    "late_interaction": Control(
        id="late_interaction", label="Late-interaction (MaxSim) retrieval",
        category="quality", scope="store", type="bool",
        config_key="vector_memory.late_interaction", default=False,
        cost="re-index to apply; then about 110 ms per query on a 16-core CPU, no GPU or reranker needed",
        pros="lifts evidence recall from 0.64 to 0.85 on LoCoMo, CPU-only",
        cons="store-level: token-level vectors are written at ingest, so turning it on or off needs a re-index, not a live toggle",
        description="Token-level MaxSim scoring; the store is built with per-token vectors (set at setup).",
        benchmarks_anchor="late-interaction-retrieval-token-level-maxsim-at-retrieval-time-tri-judge",
    ),
    "fusion": Control(
        id="fusion", label="Hybrid fusion strategy",
        category="quality", scope="runtime", type="choice",
        config_key="controls.fusion", default="rrf",
        choices=("rrf", "mem0_additive", "boost"),
        cost="none; it is a ranking-strategy choice",
        pros="rrf and mem0_additive both beat the older mem0-only guidance at full scale",
        cons="the best choice is mildly tier and task dependent; boost suits the smallest tiers",
        description="How dense and lexical hits are combined into one ranking. Overrides the recipe per query.",
        benchmarks_anchor="fusion-strategy-comparison-recall5",
    ),
    "adjacent_turns": Control(
        id="adjacent_turns", label="Conversational adjacency",
        category="quality", scope="runtime", type="int",
        config_key="controls.adjacent_turns", default=2, int_range=(0, 4),
        cost="a wider context window per hit (more tokens to the generator)",
        pros="worth about +0.089 on LoCoMo at 2; surrounding turns add context",
        cons="more context can over-budget a tiny generator; use 1 on a Pi-class tier",
        description="Include N positional neighbours around each retrieved hit. Overrides the recipe per query.",
        benchmarks_anchor="locomo--multi-session-conversational-memory-1540-qas",
    ),
    "embedder": Control(
        id="embedder", label="Dense embedder",
        category="hardware", scope="store", type="choice",
        config_key="vector_memory.embed_model", default="arctic-embed-s",
        choices=("arctic-embed-s", "minilm-onnx"),
        cost="a one-time model fetch; changing it re-embeds the whole store",
        pros="arctic-embed-s scores +0.057 judged retrieval over MiniLM at the same 384 dims and latency",
        cons="store-level: switching after ingest requires a re-index; MiniLM is the model the 97.0% headline was measured on",
        description="Which ONNX dense embedder backs retrieval (set at setup).",
        benchmarks_anchor="hardware-tiers--recommended-configurations",
    ),
    "binary_quant": Control(
        id="binary_quant", label="Binary embedding quantization",
        category="hardware", scope="store", type="bool",
        config_key="vector_memory.binary_quant", default=False,
        cost="re-embed to apply; then 32x smaller vectors and cheaper CPU distance",
        pros="32x smaller vector footprint, recall-neutral; for SBC / low-memory tiers",
        cons="store-level (a re-index to turn on or off); a slight distance approximation",
        description="Store 1 bit per dimension instead of full-precision floats.",
        benchmarks_anchor="binary-embedding-quantization--recall-neutral-ships-as-an-sbc-footprint-option",
    ),
    "self_verify": Control(
        id="self_verify", label="Answer self-verification (consumer-side)",
        category="quality", scope="consumer", type="bool",
        config_key="answer.self_verify", default=False,
        cost="a second LLM pass per answer in your answer-generation",
        pros="a CoVe-style check of the draft against the evidence; the dominant end-to-end Judge lever on LongMemEval-S (shipped config 42.8% Qwen / 51.2% llama after the N-017 judge-parser correction)",
        cons="applied in your answer-generation, not in taOSmd core (taOSmd serves memory); roughly doubles answer latency",
        description="A recommendation, not a core toggle: pair reranking with a self-verify pass in your answer-gen for the verified-answer config.",
        benchmarks_anchor="end-to-end-judge-on-longmemeval-s-the-generation-side-number",
    ),
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
}


# Named bundles. Values are {control_id: value}. prefer_verified is on globally
# by default, so Minimal is the preset that turns it off.
PRESETS: dict[str, dict] = {
    "minimal": {
        "label": "Minimal",
        "description": "Fastest and lightest: plain retrieval, no rerank, gate off.",
        "values": {"reranker": "off",
                   "prefer_verified": "off", "fusion": "rrf", "adjacent_turns": 1},
    },
    "quality": {
        "label": "Quality",
        "description": "Best accuracy where hardware affords: reranking on (pair with self-verify in your answer-gen).",
        "values": {"reranker": "bge-v2-m3",
                   "prefer_verified": "off", "fusion": "rrf", "adjacent_turns": 2},
    },
    "integrity": {
        "label": "Integrity",
        "description": "Quality plus the verified-memory gate: auditable, zero-served-hallucination recall.",
        "values": {"reranker": "bge-v2-m3",
                   "prefer_verified": "prefer_verified", "fusion": "rrf", "adjacent_turns": 2},
    },
}


def default_controls() -> dict:
    """The resolved default value for every control id."""
    return {c.id: c.default for c in CONTROLS.values()}


def validate_control(control_id: str, value) -> object:
    """Validate and coerce a value for a control. Raises ValueError on bad input."""
    c = CONTROLS.get(control_id)
    if c is None:
        raise ValueError(f"unknown control: {control_id!r}")
    if c.type == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{control_id} expects a bool, got {value!r}")
        return value
    if c.type == "choice":
        if value not in c.choices:
            raise ValueError(f"{control_id} expects one of {c.choices}, got {value!r}")
        return value
    if c.type == "int":
        try:
            iv = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{control_id} expects an int, got {value!r}") from None
        lo, hi = c.int_range
        if not (lo <= iv <= hi):
            raise ValueError(f"{control_id} must be in [{lo}, {hi}], got {iv}")
        return iv
    raise ValueError(f"{control_id} has unknown type {c.type!r}")


def controls_schema() -> dict:
    """Renderable description for the dashboard, the /controls API, and the taOS app."""
    return {
        "controls": [
            {
                "id": c.id, "label": c.label, "category": c.category,
                "scope": c.scope, "type": c.type, "config_key": c.config_key,
                "default": c.default, "choices": list(c.choices),
                "int_range": list(c.int_range), "cost": c.cost,
                "pros": c.pros, "cons": c.cons, "description": c.description,
                "benchmarks_anchor": c.benchmarks_anchor,
            }
            for c in CONTROLS.values()
        ],
        "presets": [
            {"id": pid, "label": p["label"], "description": p["description"],
             "values": dict(p["values"])}
            for pid, p in PRESETS.items()
        ],
    }
