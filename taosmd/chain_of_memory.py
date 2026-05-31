"""Chain-of-Memory (CoM) — lightweight fragment utilization before answering.

Adapted from arXiv:2601.14287 ("Chain-of-Memory: Lightweight Memory
Construction with Dynamic Evolution for LLM Agents"). The paper's thesis
matches what taosmd found the hard way: heavy memory *construction* (EDU
extraction, pass-2 lift, kitchen-sink retrieval stacks) gives marginal
gains; the unworked lever is *utilization* — how the retrieved fragments
are organized before the generator sees them.

CoM inserts one lightweight LLM call AFTER retrieval/adjacency and BEFORE
the answer prompt. It takes the retrieved fragments + the question and
produces a condensed, reasoning-ordered context: the fragments that form
the inference path to the answer, in order, with irrelevant noise pruned.
The answer prompt then runs on that condensed context instead of the raw
concatenation.

This is distinct from the other post-retrieval levers we have:
  - LLM-rerank scores/reorders candidates but keeps them as a flat list.
  - CoVe verifies the *answer* after the fact.
  - CoM restructures the *context* into an inference chain and prunes noise
    before the answer is drafted.

Fail-safe: any transport/parse error returns the original context unchanged,
so a CoM failure never starves the generator. Opt-in via --chain-of-memory.

NOT YET VALIDATED — prototype pending a LoCoMo bench. Default off.
"""

from __future__ import annotations

import httpx


_COM_SYSTEM = """\
You are preparing retrieved conversation memory for a question-answering step.

You are given a QUESTION and a set of retrieved memory FRAGMENTS (each a line
from a past conversation, some relevant, some noise). Your job is NOT to answer
the question. Your job is to construct the reasoning context:

1. Select only the fragments that are actually needed to answer the question.
2. Drop irrelevant fragments entirely — do not pad.
3. Order the kept fragments into the sequence a careful reader would follow to
   reason toward the answer (e.g. earliest-relevant-fact first, then the facts
   that build on it, ending with the fact that most directly answers).
4. Preserve each kept fragment's wording and any [date] prefixes verbatim — do
   not paraphrase, summarise, or invent. Copy them exactly.

Return JSON: {"ordered_fragments": ["<fragment 1>", "<fragment 2>", ...]}
If no fragment is relevant, return {"ordered_fragments": []}.
"""


async def organize_fragments(
    question: str,
    fragments: list[str],
    *,
    model: str,
    ollama_url: str,
    http_client: httpx.AsyncClient,
    timeout: float = 120.0,
    thinking_mode: bool = False,
) -> list[str] | None:
    """Select + order the fragments into an inference path for ``question``.

    Returns the kept fragments in reasoning order, or ``None`` on
    transport/parse failure so the caller can fall back to the original
    context. An empty list means "the model judged none relevant" — distinct
    from None (failure); the caller may choose to keep the originals in that
    case rather than answer on empty context.
    """
    if not fragments:
        return []

    import json
    listing = "\n".join(f"- {f}" for f in fragments)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _COM_SYSTEM},
            {"role": "user", "content": f"QUESTION: {question}\n\nFRAGMENTS:\n{listing}"},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_predict": 4096},
    }
    if not thinking_mode:
        payload["think"] = False
    try:
        resp = await http_client.post(
            f"{ollama_url.rstrip('/')}/api/chat", json=payload, timeout=timeout,
        )
        resp.raise_for_status()
        content = (resp.json().get("message") or {}).get("content", "")
        parsed = json.loads(content)
    except (httpx.HTTPError, json.JSONDecodeError, KeyError):
        return None

    out = parsed.get("ordered_fragments")
    if not isinstance(out, list):
        return None
    # Keep only strings that actually appeared in the input (the model is told
    # to copy verbatim; this guards against invented fragments). Match on a
    # normalised-whitespace basis so trivial reformatting still maps back to
    # the real fragment text.
    by_norm = {" ".join(f.split()): f for f in fragments}
    kept: list[str] = []
    seen: set[str] = set()
    for frag in out:
        if not isinstance(frag, str):
            continue
        norm = " ".join(frag.split())
        real = by_norm.get(norm)
        if real is None:
            # Fall back to a containment match — the model may have trimmed a
            # trailing clause. Accept if it's a clear prefix of exactly one.
            cands = [orig for n, orig in by_norm.items() if n.startswith(norm) and len(norm) > 20]
            real = cands[0] if len(cands) == 1 else None
        if real is not None and real not in seen:
            kept.append(real)
            seen.add(real)
    return kept


def fragments_from_context(context: str) -> list[str]:
    """Split a built context block back into individual fragment lines.

    ``_build_context`` joins hits (and adjacent-turn neighbours) with newlines;
    each non-empty line is one fragment for CoM to reason over.
    """
    return [ln for ln in context.split("\n") if ln.strip()]
