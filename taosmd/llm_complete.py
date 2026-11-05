"""Shared completion router for the librarian LLM calls (#643).

The librarian stages (session enrichment, crystallize, reflect, query
expansion, LLM rerank) hardcoded Ollama's native /api/generate, so any
OpenAI-compatible server (llama.cpp, vLLM, Ollama's own /v1 surface)
404'd and the stage silently degraded to heuristics. This module picks
the API family from llm_api config (or endpoint probing) and issues the
same completion in either dialect.

llm_api values:
  "ollama"  - native /api/generate (fastest, keeps options/think)
  "openai"  - /v1/chat/completions
  "" (auto) - probe: GET <base>/api/version; 2xx -> ollama, else openai.
              Probe results are cached per base URL for the process.
"""
from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

# base URL -> "ollama" | "openai"
_probe_cache: dict[str, str] = {}


def set_api_family(base_url: str, family: str) -> None:
    """Pin the dialect for a base URL (config seam / tests)."""
    _probe_cache[base_url.rstrip("/")] = family


async def _family(base_url: str, client: httpx.AsyncClient) -> str:
    base = base_url.rstrip("/")
    cached = _probe_cache.get(base)
    if cached:
        return cached
    try:
        if (await client.get(f"{base}/api/version", timeout=5)).status_code == 200:
            fam = "ollama"
        elif (await client.get(f"{base}/v1/models", timeout=5)).status_code == 200:
            fam = "openai"
        else:
            # Neither signature answered: keep the historical default so
            # existing Ollama installs behave exactly as before.
            fam = "ollama"
    except Exception:
        fam = "ollama"
    _probe_cache[base] = fam
    logger.info("llm_complete: treating %s as %s API", base, fam)
    return fam


async def llm_complete(
    client: httpx.AsyncClient,
    llm_url: str,
    model: str,
    prompt: str,
    *,
    api: str = "",
    temperature: float = 0.2,
    max_tokens: int = 400,
    no_think: bool = False,
    timeout: float | None = None,
) -> str:
    """One completion, either dialect. Raises on HTTP/network failure.

    Returns the response text. For reasoning backends that answer with
    content=null (some llama.cpp patches / thinking modes), falls back to
    the reasoning field so the caller still gets usable text rather than
    an empty parse.
    """
    base = llm_url.rstrip("/")
    fam = api if api in ("ollama", "openai") else await _family(base, client)
    kw = {"timeout": timeout} if timeout is not None else {}

    if fam == "ollama":
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if no_think:
            payload["think"] = False
        resp = await client.post(f"{base}/api/generate", json=payload, **kw)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response") or data.get("thinking") or ""
    else:
        resp = await client.post(
            f"{base}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            **kw,
        )
        resp.raise_for_status()
        choice = resp.json()["choices"][0]["message"]
        text = choice.get("content") or choice.get("reasoning") or ""
    return text


__all__ = ["llm_complete", "set_api_family"]
