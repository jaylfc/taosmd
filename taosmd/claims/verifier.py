"""Claim verifiers. A verifier is shown ONLY the cited spans and judges whether
the claim is supported. Cross-family from the extractor (the F-009 method).
Fail-closed: anything unparseable maps to 'unverified', never 'supported'.
"""
from __future__ import annotations

import re
from typing import Protocol

VERDICTS = ("supported", "partial", "unsupported", "contradicted")

_PROMPT = (
    "You are checking whether a CLAIM is supported by the SOURCE text. "
    "Use only the SOURCE. Answer with exactly one word: SUPPORTED, PARTIAL, "
    "UNSUPPORTED, or CONTRADICTED.\n\nSOURCE:\n{spans}\n\nCLAIM: {claim}\n\nVerdict:"
)


def parse_verdict(text: str) -> str:
    """Map a judge reply to a status; unparseable -> 'unverified' (fail-closed)."""
    t = text.upper()
    for v in ("CONTRADICTED", "UNSUPPORTED", "PARTIAL", "SUPPORTED"):
        if re.search(rf"\b{v}\b", t):
            return v.lower()
    return "unverified"


class Verifier(Protocol):
    def verify(self, claim_text: str, span_texts: list[str]) -> tuple[str, str]:
        """Return (status, model_id). status in VERDICTS or 'unverified'."""
        ...


class FakeVerifier:
    """Scripted verifier for offline tests."""

    def __init__(self, scripted: dict[str, str], default: str = "unverified"):
        self._scripted = dict(scripted)
        self._default = default

    def verify(self, claim_text: str, span_texts: list[str]) -> tuple[str, str]:
        return (self._scripted.get(claim_text, self._default), "fake")


class LocalEntailmentVerifier:
    """Cross-family local-LLM entailment over an Ollama-compatible endpoint.

    Constructed with an httpx client, base url, and model tag. verify() is sync
    over a blocking call for simplicity inside the (already async, batched)
    verify-pass; a model or network error returns ('unverified', model) so the
    claim is never promoted on failure.
    """

    def __init__(self, client, ollama_url: str, model: str, timeout: float = 60.0):
        self._client = client
        self._url = ollama_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    def verify(self, claim_text: str, span_texts: list[str]) -> tuple[str, str]:
        prompt = _PROMPT.format(spans="\n".join(span_texts), claim=claim_text)
        try:
            r = self._client.post(
                f"{self._url}/api/generate",
                json={"model": self._model, "prompt": prompt, "stream": False},
                timeout=self._timeout,
            )
            r.raise_for_status()
            return parse_verdict(r.json().get("response", "")), self._model
        except Exception:  # noqa: BLE001 - fail closed
            return "unverified", self._model
