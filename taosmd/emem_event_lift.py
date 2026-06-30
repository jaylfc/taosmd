"""EMem pass-2: lift EDUs into typed (subject, predicate, object) triples.

Pass 1 (``taosmd.emem_edu.extract_session_edus``) decomposes a conversation
session into atomic Elementary Discourse Units like "Bob traveled to Tokyo
for 5 days in March 2024 for the Global AI Innovation Symposium". That's
already the validated SH-specialist retrieval recipe; see PR #74 and
docs/benchmarks.md.

Pass 2 turns each EDU into structured KG triples whose predicates are
constrained to ``taosmd.predicate_vocab.ALLOWED_PREDICATES``. The EDU
above becomes:

  [("bob", "moved_to", "tokyo"),
   ("bob", "attended", "global ai innovation symposium 2024")]

This is opt-in via ``--emem-edu-pass2-events`` on the LoCoMo runner. The
pass-1 retrieval recipe is unchanged whether pass-2 runs or not; pass-2
exists to populate the KG side of the system (structured queries like
"who has Bob met?") which LoCoMo doesn't measure but production agents
will.

Why constrain the predicate? Because gbrain-style "closed vocab" beats
free-form predicates for query-side consistency. Same reason the closed
vocab landed in PR #76: if the librarian decides "is at" today and
"works at" tomorrow, ``query_predicate("works_at")`` misses half the
data. Constraining at extraction time prevents drift at source.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from .predicate_vocab import ALLOWED_PREDICATES, normalise


# The system prompt lists every allowed predicate inline so the model can
# pick from a fixed set rather than inventing new ones. We keep the list
# sorted + comma-separated for token efficiency. Renders at ~250 tokens,
# acceptable per-call overhead for a 5-15 s LLM round trip.
def _allowed_predicates_inline() -> str:
    return ", ".join(sorted(ALLOWED_PREDICATES))


_SYSTEM_PROMPT = """\
You are extracting structured (subject, predicate, object) triples from a single Elementary Discourse Unit (EDU), a short, atomic factual statement.

For each triple:
- subject and object must be normalised entity names. Lowercase, use the most informative name (e.g. "bob" not "he", "global ai innovation symposium 2024" not "the conference").
- predicate MUST be one of these allowed predicates and nothing else:
{predicates_inline}

If a fact's natural predicate isn't in the allowed list, EITHER pick the closest match OR drop the triple. Do not invent a new predicate. Drop triples you can't express cleanly.

Return JSON of the form:
  {{"event_type": "<short label, e.g. 'travel', 'meeting', 'employment_change'>",
    "triples": [{{"subject": "...", "predicate": "...", "object": "..."}}, ...]}}

If the EDU has no extractable triples (pleasantry, hedge, opinion without a fact), return {{"event_type": "none", "triples": []}}.
"""


_ONESHOT_INPUT = """\
EDU: Bob traveled to Tokyo for 5 days in March 2024 to attend the Global AI Innovation Symposium at Tokyo University.
"""


_ONESHOT_OUTPUT = json.dumps({
    "event_type": "travel",
    "triples": [
        {"subject": "bob", "predicate": "moved_to", "object": "tokyo"},
        {"subject": "bob", "predicate": "attended",
         "object": "global ai innovation symposium 2024"},
        {"subject": "global ai innovation symposium 2024",
         "predicate": "located_in", "object": "tokyo university"},
    ],
})


async def _ollama_chat_json(
    client: httpx.AsyncClient, ollama_url: str, model: str,
    messages: list[dict], *, timeout: float = 120.0, num_predict: int = 2048,
) -> str:
    """POST to /api/chat with format=json. Mirrors emem_edu._ollama_chat_json
    so we can stay deterministic + JSON-strict on the small open-source models
    that get used as extractors at our hardware tier.
    """
    resp = await client.post(
        f"{ollama_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": num_predict},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return (data.get("message") or {}).get("content", "")


def _system_prompt() -> str:
    return _SYSTEM_PROMPT.format(predicates_inline=_allowed_predicates_inline())


async def lift_edu_to_triples(
    edu_text: str,
    *,
    model: str | None = None,
    ollama_url: str,
    http_client: httpx.AsyncClient,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Lift a single EDU into ``{event_type, triples}`` with vocab-constrained predicates.

    Returns ``{"event_type": str, "triples": [{"subject", "predicate", "object"}]}``.
    Predicates are normalised + filtered against ``ALLOWED_PREDICATES`` here
    in the client (belt-and-braces: the system prompt asks for it, but
    smaller models occasionally invent). Triples whose predicate doesn't
    survive normalisation/filtering are silently dropped (logged at debug
    by the caller if it cares); we'd rather lose a noisy triple than write
    a free-form one and corrode the vocab.

    Returns ``{"event_type": "none", "triples": []}`` on transport / JSON
    error; caller can treat empty as "nothing extracted" without needing
    to differentiate "nothing there" from "extractor failed".

    ``model`` is a ``provider:model`` (or bare model) string. When omitted
    it resolves to the system-wide memory model (see :mod:`taosmd.config`),
    falling back to ``llama3.1:8b`` (the prior pass-2 default) when no
    global is configured.
    """
    if not model:
        from .config import resolve_memory_model  # noqa: PLC0415

        model = resolve_memory_model(fallback="llama3.1:8b")
    messages = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": _ONESHOT_INPUT},
        {"role": "assistant", "content": _ONESHOT_OUTPUT},
        {"role": "user", "content": f"EDU: {edu_text}"},
    ]

    try:
        raw = await _ollama_chat_json(
            http_client, ollama_url, model, messages,
            timeout=timeout, num_predict=2048,
        )
        parsed = json.loads(raw)
    except (httpx.HTTPError, json.JSONDecodeError):
        return {"event_type": "none", "triples": []}

    event_type = (parsed.get("event_type") or "none").strip().lower()
    out_triples: list[dict[str, str]] = []
    for t in parsed.get("triples", []) or []:
        s = (t.get("subject") or "").strip().lower()
        p_raw = (t.get("predicate") or "").strip().lower()
        o = (t.get("object") or "").strip().lower()
        if not s or not p_raw or not o:
            continue
        # Normalise through synonym table first; reject if still outside vocab.
        p = normalise(p_raw)
        if p not in ALLOWED_PREDICATES:
            continue
        out_triples.append({"subject": s, "predicate": p, "object": o})

    return {"event_type": event_type, "triples": out_triples}


async def lift_edus_to_events(
    edus: list[dict],
    *,
    model: str | None = None,
    ollama_url: str,
    http_client: httpx.AsyncClient,
    timeout: float = 120.0,
) -> list[dict[str, Any]]:
    """Lift a batch of EDUs in sequence. One LLM call per EDU.

    Returns one ``{event_type, triples, source_edu, source_turn_ids}`` dict
    per input EDU. ``source_edu`` is the original EDU text (preserved so
    KG writers can record provenance); ``source_turn_ids`` is copied from
    the pass-1 EDU output so the resulting triples remain traceable to the
    underlying conversation turns.

    Sequential calls (not concurrent) to avoid hammering Ollama under
    NUM_PARALLEL=1, which is the default LoCoMo bench config. Caller can
    parallelise across sessions if they want; within a session, pass-1
    already serialised on ingest order.
    """
    out: list[dict[str, Any]] = []
    for edu in edus:
        edu_text = (edu.get("edu_text") or "").strip()
        if not edu_text:
            continue
        result = await lift_edu_to_triples(
            edu_text,
            model=model, ollama_url=ollama_url,
            http_client=http_client, timeout=timeout,
        )
        result["source_edu"] = edu_text
        result["source_turn_ids"] = edu.get("source_turn_ids", []) or []
        out.append(result)
    return out
