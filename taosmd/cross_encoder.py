class CrossEncoderReranker:
    def __init__(self, onnx_path: str = "models/cross-encoder-onnx"):
        self._onnx_path = onnx_path
        self._session = None
        self._tokenizer = None

    @property
    def available(self) -> bool:
        """True if the ONNX model files exist."""
        from pathlib import Path
        model_path = Path(self._onnx_path) / "model.onnx"
        return model_path.exists()

    def _load(self):
        """Lazy-load the ONNX model and tokenizer."""
        if self._session is not None:
            return
        import onnxruntime as ort
        from transformers import AutoTokenizer
        self._session = ort.InferenceSession(
            f"{self._onnx_path}/model.onnx",
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
