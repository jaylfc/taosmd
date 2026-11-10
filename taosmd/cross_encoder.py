"""Cross-Encoder Reranker (taOSmd).

Second-stage reranker that scores (query, document) pairs more accurately
than cosine similarity. Uses ms-marco-MiniLM-L-6-v2 via ONNX Runtime.

Retrieve top-N candidates with vector search, rerank to top-K with this.
~1ms per (query, doc) pair on CPU. For 15 candidates: ~15ms total.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """ONNX-based cross-encoder for reranking retrieval results."""

    def __init__(self, onnx_path: str = "models/cross-encoder-onnx", min_confidence: float | None = None):
        self._onnx_path = onnx_path
        self._session = None
        self._tokenizer = None
        self._model_file = self._find_model_file()
        self.min_confidence = min_confidence

    def _find_model_file(self) -> str | None:
        """Find the ONNX model file (handles different download layouts)."""
        candidates = [
            Path(self._onnx_path) / "model.onnx",
            Path(self._onnx_path) / "onnx" / "model.onnx",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return None

    @property
    def available(self) -> bool:
        """True if the ONNX model files exist."""
        if self._model_file is None:
            self._model_file = self._find_model_file()
        return self._model_file is not None

    def _load(self):
        """Lazy-load the ONNX model and tokenizer."""
        if self._session is not None:
            return
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover - depends on the install
            # onnxruntime is an optional extra: it publishes no musllinux wheel
            # and no sdist, so it cannot be a hard dependency without making
            # taosmd uninstallable on Alpine/postmarketOS.  Name the fix rather
            # than surfacing a bare ImportError from deep in a reranker call.
            raise ImportError(
                "the ONNX cross-encoder needs onnxruntime, which is not installed. "
                "Install it with `pip install taosmd[onnx]` (note: onnxruntime "
                "publishes no musl build, so this extra is unavailable on Alpine "
                "and postmarketOS -- use the qmd embed backend there instead)."
            ) from e
        from transformers import AutoTokenizer

        if not self._model_file:
            raise FileNotFoundError(f"No model.onnx found in {self._onnx_path}")

        self._session = ort.InferenceSession(
            self._model_file,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self._onnx_path)

    def _score(self, query: str, results: list[dict]) -> list[float]:
        """Score (query, text) pairs and return a list of cross-encoder scores.

        One float per result, in the same order as the input results.
        """
        if not self.available or not results:
            return []

        try:
            self._load()
        except Exception:
            return []

        import numpy as np

        session_input_names = [inp.name for inp in self._session.get_inputs()]
        needs_token_types = "token_type_ids" in session_input_names

        scores = []
        for r in results:
            text = r.get("text", "")[:512]
            inputs = self._tokenizer(
                query, text, return_tensors="np",
                padding=True, truncation=True, max_length=512,
                return_token_type_ids=needs_token_types or None,
            )
            feed = {name: inputs[name].astype(np.int64)
                    for name in session_input_names
                    if name in inputs}
            if needs_token_types and "token_type_ids" not in feed:
                feed["token_type_ids"] = np.zeros_like(
                    inputs["input_ids"], dtype=np.int64)
            output = self._session.run(None, feed)
            score = float(output[0][0][0]) if output[0].ndim > 1 else float(output[0][0])
            scores.append(score)

        return scores

    def rerank(self, query: str, results: list[dict], limit: int = 5) -> list[dict]:
        """Score (query, text) pairs and re-sort by cross-encoder score.

        If model not available, returns results unchanged (graceful fallback).
        If min_confidence is set and max(scores) < min_confidence, returns the
        input order truncated to limit with rerank_applied=False on each hit.
        Results are dicts with at least a "text" field.
        """
        if not self.available or not results:
            return results[:limit]

        try:
            self._load()
        except Exception:
            return results[:limit]

        scores = self._score(query, results)

        if not scores:
            return results[:limit]

        if self.min_confidence is not None and max(scores) < self.min_confidence:
            return [
                {**r, "cross_encoder_score": round(s, 4), "rerank_applied": False}
                for r, s in zip(results[:limit], scores[:limit])
            ]

        scored = list(zip(scores, results))
        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            {**r, "cross_encoder_score": round(s, 4), "rerank_applied": True}
            for s, r in scored[:limit]
        ]
