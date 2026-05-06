"""ENGRAM-style turn classifier.

Routes each conversation turn into one or more memory types
{episodic, semantic, procedural} as a 3-bit mask. Consumed at retrieval
time to run per-type top-k searches that are merged + deduplicated.

Based on "ENGRAM: Effective, Lightweight Memory Orchestration for
Conversational Agents" (arXiv:2511.12960). The paper showed that
collapsing the typed stores into a single undifferentiated store dropped
overall LoCoMo performance from 77.55 to 46.56 — typed separation is the
load-bearing piece, not any specific embedder or retriever choice.

Default backend is Ollama (qwen3:4b at temperature 0). Cache by SHA-256
of the turn text to avoid re-classifying the same string across
benchmark re-runs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

EPISODIC = 1
SEMANTIC = 2
PROCEDURAL = 4
ALL_TYPES = EPISODIC | SEMANTIC | PROCEDURAL

CLASSIFY_PROMPT = """You are classifying a conversation turn into one or more memory types.

Output a single JSON object with three boolean fields:
- "episodic": true if the turn describes events with a temporal anchor (e.g., "yesterday", "last week", "just got back", "this morning"). Specific occurrences in time.
- "semantic": true if the turn states stable facts, preferences, or attributes about people, places, or things (e.g., "I work at Stripe", "she lives in Tokyo", "I prefer tea"). No time anchor; treated as currently true.
- "procedural": true if the turn describes instructions, workflows, how-to knowledge, or repeatable steps (e.g., "the way I make coffee is to bloom for 30 seconds first", "to log in, click the icon and enter your code").

A turn can be true for multiple types. At least one type must be true.

Turn: {turn_text}

Output JSON only, no preamble:"""


class EngramRouter:
    """LLM-backed classifier mapping each turn to a memory-type bitmask.

    Args:
        client: An ``httpx.AsyncClient`` for issuing requests.
        ollama_url: Base URL for the Ollama server. Default
            ``http://localhost:11434``.
        model: Generator model name. Default ``qwen3:4b`` (matches the
            external judge size used in benchmarks).
        cache_path: Optional path to a JSON file that persists the
            ``hash(turn_text) -> bitmask`` map across runs.
        concurrency: Max concurrent classifier calls.
        timeout: Per-request timeout seconds.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen3:4b",
        cache_path: str | Path | None = None,
        concurrency: int = 4,
        timeout: float = 30.0,
    ):
        self._client = client
        self._url = ollama_url.rstrip("/")
        self._model = model
        self._cache: dict[str, int] = {}
        self._cache_path = Path(cache_path) if cache_path else None
        self._sem = asyncio.Semaphore(concurrency)
        self._timeout = timeout
        if self._cache_path and self._cache_path.exists():
            try:
                self._cache = json.loads(self._cache_path.read_text())
                logger.info(
                    "Loaded %d cached EngramRouter classifications from %s",
                    len(self._cache),
                    self._cache_path,
                )
            except Exception as e:
                logger.warning("Failed to load engram cache: %s", e)

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    async def classify(self, turn_text: str) -> int:
        """Return the 3-bit memory-type mask for this turn."""
        key = self._hash(turn_text)
        if key in self._cache:
            return self._cache[key]
        async with self._sem:
            mask = await self._llm_classify(turn_text)
        self._cache[key] = mask
        return mask

    async def classify_batch(self, turns: list[str]) -> list[int]:
        """Classify many turns concurrently."""
        return await asyncio.gather(*(self.classify(t) for t in turns))

    async def _llm_classify(self, turn_text: str) -> int:
        prompt = CLASSIFY_PROMPT.format(turn_text=turn_text[:1024])
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0},
            "think": False,
        }
        try:
            resp = await self._client.post(
                f"{self._url}/api/generate",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            text = (resp.json().get("response") or "").strip()
            mask = self._parse(text)
            if mask == 0:
                # Empty mask means classifier didn't commit; fall back to
                # episodic+semantic (the two dominant LoCoMo types).
                mask = EPISODIC | SEMANTIC
            return mask
        except Exception as e:
            logger.debug("EngramRouter classify failed: %s; defaulting to ALL_TYPES", e)
            return ALL_TYPES

    @staticmethod
    def _parse(response: str) -> int:
        m = re.search(r"\{[^}]*\}", response, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                mask = 0
                if obj.get("episodic"):
                    mask |= EPISODIC
                if obj.get("semantic"):
                    mask |= SEMANTIC
                if obj.get("procedural"):
                    mask |= PROCEDURAL
                return mask
            except json.JSONDecodeError:
                pass
        # Last-ditch keyword scan if JSON parse failed
        text = response.lower()
        mask = 0
        if re.search(r'"?episodic"?\s*:\s*true', text):
            mask |= EPISODIC
        if re.search(r'"?semantic"?\s*:\s*true', text):
            mask |= SEMANTIC
        if re.search(r'"?procedural"?\s*:\s*true', text):
            mask |= PROCEDURAL
        return mask

    def flush_cache(self) -> None:
        if not self._cache_path:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._cache))
            logger.info(
                "Flushed %d EngramRouter classifications to %s",
                len(self._cache),
                self._cache_path,
            )
        except Exception as e:
            logger.warning("Failed to flush engram cache: %s", e)


__all__ = ["EngramRouter", "EPISODIC", "SEMANTIC", "PROCEDURAL", "ALL_TYPES"]
