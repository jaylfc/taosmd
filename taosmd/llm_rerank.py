"""LLM-based listwise reranker.

Single-pass listwise scoring: one LLM call sees all N candidates plus the
query and returns a ranked list of candidate numbers, most-to-least
relevant. Fast (200 QAs × ~5 s/call ≈ 17 min for the rerank step) vs
pointwise (N × per-call), and the listwise prompt lets the model see
cross-document relations rather than scoring each doc in isolation.

Inspired by Mem0's llm_reranker (Apache 2.0, mem0/reranker/llm_reranker.py)
but uses a single listwise call instead of Mem0's per-candidate
pointwise scoring, chosen for our local-tier latency budget where 20
sequential ~3 s LLM calls per QA would dominate bench wall-clock.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

import httpx

logger = logging.getLogger(__name__)

LLM_RERANK_PROMPT = """You are a relevance ranker. Given a question and {n} candidate passages, output the candidate numbers in order from MOST relevant to LEAST relevant for answering the question.

Output ONLY the candidate numbers, comma-separated, on a single line. No explanation, no other text.

Question: {query}

Candidates:
{candidates}

Ranking (most to least relevant, comma-separated numbers, just the numbers):"""


def _build_candidates_block(texts: Iterable[str], char_cap: int = 280) -> str:
    """Render candidates as a numbered block, truncated to keep prompt budget."""
    lines = []
    for i, text in enumerate(texts, start=1):
        snippet = (text or "").strip().replace("\n", " ")
        if len(snippet) > char_cap:
            snippet = snippet[:char_cap].rstrip() + "…"
        lines.append(f"{i}. {snippet}")
    return "\n".join(lines)


def _parse_ranking(response: str, n: int) -> list[int]:
    """Extract a permutation of 1..n from the LLM response.

    Robust to extra prose, stray punctuation, repeated numbers, and
    out-of-range numbers. Missing candidates are appended in original
    order so the output is always a valid permutation of length n.
    """
    nums: list[int] = []
    seen: set[int] = set()
    for tok in re.findall(r"\d+", response or ""):
        try:
            x = int(tok)
        except ValueError:
            continue
        if 1 <= x <= n and x not in seen:
            nums.append(x)
            seen.add(x)
    # Append anything missing in original order so the result is always
    # a complete permutation. Better to "trust the original cross-encoder
    # ranking for the tail" than silently drop candidates the LLM didn't
    # mention.
    for x in range(1, n + 1):
        if x not in seen:
            nums.append(x)
    return nums


async def llm_listwise_rerank(
    client: httpx.AsyncClient,
    ollama_url: str,
    rerank_model: str,
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    timeout: float = 60.0,
    no_think_prefix: bool = False,
) -> list[dict]:
    """Listwise rerank a candidate list and return the top-k.

    Each candidate must have a ``text`` field (used for prompt) and is
    otherwise passed through unchanged. The reranker reorders the list
    in place and returns the top ``top_k``; downstream code can keep
    using the same hit dict shape.

    On any failure (network, parse, timeout) returns the input
    candidates truncated to ``top_k``, i.e. trusts the upstream
    cross-encoder ranking.
    """
    if not candidates:
        return []
    n = len(candidates)
    if n <= 1:
        return candidates[:top_k]

    prompt = LLM_RERANK_PROMPT.format(
        n=n,
        query=query,
        candidates=_build_candidates_block(c.get("text", "") for c in candidates),
    )
    if no_think_prefix:
        prompt = "/no_think\n\n" + prompt

    payload = {
        "model": rerank_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 200},
    }
    if not no_think_prefix:
        payload["think"] = False

    try:
        resp = await client.post(
            f"{ollama_url.rstrip('/')}/api/generate",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        text = (resp.json().get("response") or "").strip()
    except Exception as e:
        logger.debug("LLM listwise rerank failed: %s; trusting upstream order", e)
        return candidates[:top_k]

    order = _parse_ranking(text, n)
    reranked = [candidates[i - 1] for i in order]
    return reranked[:top_k]


__all__ = ["llm_listwise_rerank", "LLM_RERANK_PROMPT"]
