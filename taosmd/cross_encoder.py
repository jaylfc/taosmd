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

    def __init__(self, onnx_path: str = "models/cross-encoder-onnx"):
        self._onnx_path = onnx_path
        self._session = None
        self._tokenizer = None
        self._model_file = self._find_model_file()

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
        import onnxruntime as ort
        from transformers import AutoTokenizer

        if not self._model_file:
            raise FileNotFoundError(f"No model.onnx found in {self._onnx_path}")

        self._session = ort.InferenceSession(
            self._model_file,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self._onnx_path)

    def rerank(self, query: str, results: list[dict], limit: int = 5) -> list[dict]:
        """Score (query, text) pairs and re-sort by cross-encoder score.

        If model not available, returns results unchanged (graceful fallback).
        Results are dicts with at least a "text" field.
        """
        if not self.available or not results:
            return results[:limit]

        try:
            self._load()
        except Exception:
            return results[:limit]

        import numpy as np

        scores = []
        for r in results:
            text = r.get("text", "")[:512]
            inputs = self._tokenizer(
                query, text, return_tensors="np",
                padding=True, truncation=True, max_length=512,
            )
            feed = {name: inputs[name].astype(np.int64)
                    for name in [inp.name for inp in self._session.get_inputs()]
                    if name in inputs}
            output = self._session.run(None, feed)
            # Cross-encoder outputs logits; higher = more relevant
            score = float(output[0][0][0]) if output[0].ndim > 1 else float(output[0][0])
            scores.append(score)

        # Sort by cross-encoder score descending
        scored = list(zip(scores, results))
        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            {**r, "cross_encoder_score": round(s, 4)}
            for s, r in scored[:limit]
        ]
