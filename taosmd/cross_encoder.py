"""Cross-Encoder Reranker (taOSmd).

Second-stage reranker that scores (query, document) pairs more accurately
than cosine similarity. Uses ms-marco-MiniLM-L-6-v2 via ONNX Runtime.

Retrieve top-N candidates with vector search, rerank to top-K with this.
~1ms per (query, doc) pair on CPU. For 15 candidates: ~15ms total.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """ONNX-based cross-encoder for reranking retrieval results."""

    def __init__(
        self,
        onnx_path: str | None = None,
        min_confidence: float | None = None,
        *,
        provider: str | None = None,
        data_dir: str | Path | None = None,
    ):
        from . import generator_profiles as _gp
        self._onnx_path = onnx_path or _gp.resolve_model(
            "cross_encoder_path", fallback="models/cross-encoder-onnx", data_dir=data_dir
        )
        self._provider = provider or _gp.resolve_model(
            "cross_encoder_provider", fallback="CPUExecutionProvider", data_dir=data_dir
        )
        self._session = None
        self._tokenizer = None
        self._model_file = self._find_model_file()
        self.min_confidence = min_confidence
        # Ettin reranker head weights (loaded lazily if needed)
        self._ettin_head: dict | None = None

    def _find_model_file(self) -> str | None:
        """Find the ONNX model file (handles different download layouts).

        Git LFS pointer files (which are small text files, not real models)
        are explicitly rejected.
        """
        from .api import _is_lfs_pointer
        candidates = [
            Path(self._onnx_path) / "model.onnx",
            Path(self._onnx_path) / "onnx" / "model.onnx",
        ]
        for c in candidates:
            if c.exists() and not _is_lfs_pointer(c):
                return str(c)
        return None

    @property
    def available(self) -> bool:
        """True if the ONNX model files exist."""
        if self._model_file is None:
            self._model_file = self._find_model_file()
        return self._model_file is not None

    def _is_ettin_reranker(self) -> bool:
        """Check if the model is an ettin-reranker model by inspecting config."""
        config_path = Path(self._onnx_path) / "config.json"
        if not config_path.exists():
            return False
        try:
            with open(config_path) as f:
                config = json.load(f)
            # sentence-transformers CrossEncoder models have a "modules" list
            modules = config.get("modules", [])
            for mod in modules:
                if isinstance(mod, dict) and mod.get("type") == "transformer":
                    model_name = mod.get("path", "").lower()
                    if "ettin" in model_name or "ettin-reranker" in model_name:
                        return True
            # Also check model name in config
            model_name = config.get("_name_or_path", "").lower()
            if "ettin-reranker" in model_name:
                return True
        except Exception:
            pass
        return False

    def _load_ettin_head(self) -> dict | None:
        """Load ettin-reranker scoring head weights from safetensors.

        Returns dict with keys: dense1_weight, dense1_bias, ln_weight, ln_bias,
        dense2_weight, dense2_bias, or None if not found.
        """
        try:
            from safetensors import safe_open
        except ImportError:
            logger.debug("safetensors not installed, cannot load ettin head")
            return None

        model_dir = Path(self._onnx_path)
        # Look for model.safetensors or pytorch_model.bin
        weight_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("pytorch_model.bin"))
        if not weight_files:
            return None

        head_weights = {}
        for wf in weight_files:
            try:
                if wf.suffix == ".safetensors":
                    with safe_open(wf, framework="numpy") as f:
                        for key in f.keys():
                            if any(k in key for k in ["1.", "2.", "3.", "4."]):  # head modules
                                head_weights[key] = f.get_tensor(key)
                else:
                    import torch
                    state_dict = torch.load(wf, map_location="cpu")
                    for key, val in state_dict.items():
                        if any(k in key for k in ["1.", "2.", "3.", "4."]):
                            head_weights[key] = val.numpy()
            except Exception as e:
                logger.debug(f"Failed to load weights from {wf}: {e}")
                continue

        if not head_weights:
            return None

        # Map sentence-transformers module keys to our head structure
        # Module 1: Pooling (no weights)
        # Module 2: Dense (in_features=512, out_features=512, bias=False, GELU)
        # Module 3: LayerNorm (dimension=512)
        # Module 4: Dense (in_features=512, out_features=1, bias=True, Identity)
        result = {}
        for key, val in head_weights.items():
            if "1." in key or "module.1" in key:
                continue  # pooling has no weights
            elif "2." in key or "module.2" in key:
                if "weight" in key:
                    result["dense1_weight"] = val
                elif "bias" in key:
                    result["dense1_bias"] = val
            elif "3." in key or "module.3" in key:
                if "weight" in key:
                    result["ln_weight"] = val
                elif "bias" in key:
                    result["ln_bias"] = val
            elif "4." in key or "module.4" in key:
                if "weight" in key:
                    result["dense2_weight"] = val
                elif "bias" in key:
                    result["dense2_bias"] = val

        # Verify we have the essential weights
        required = ["dense1_weight", "ln_weight", "dense2_weight"]
        if all(k in result for k in required):
            return result
        return None

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
            providers=[self._provider],
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self._onnx_path)

        # If this is an ettin-reranker model, try to load the scoring head
        if self._is_ettin_reranker():
            self._ettin_head = self._load_ettin_head()
            if self._ettin_head:
                logger.info("Loaded ettin-reranker scoring head from safetensors")
            else:
                logger.warning(
                    "ettin-reranker model detected but scoring head not found in safetensors. "
                    "Ensure the ONNX export includes the full CrossEncoder pipeline."
                )

    def _apply_ettin_head(self, last_hidden_state: np.ndarray) -> np.ndarray:
        """Apply ettin-reranker scoring head to last_hidden_state.

        Args:
            last_hidden_state: Shape (batch, seq_len, hidden_dim)

        Returns:
            Logits shaped (batch,)
        """

        # CLS pooling: take first token
        cls_embeddings = last_hidden_state[:, 0, :]  # (batch, hidden)

        # Dense1: (hidden, hidden) + GELU
        dense1_weight = self._ettin_head["dense1_weight"]  # (hidden, hidden)
        x = cls_embeddings @ dense1_weight.T  # (batch, hidden)
        if "dense1_bias" in self._ettin_head:
            x = x + self._ettin_head["dense1_bias"]
        # GELU activation
        x = x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

        # LayerNorm
        ln_weight = self._ettin_head["ln_weight"]
        ln_bias = self._ettin_head.get("ln_bias", np.zeros_like(ln_weight))
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x = (x - mean) / np.sqrt(var + 1e-12)
        x = x * ln_weight + ln_bias

        # Dense2: (hidden, 1)
        dense2_weight = self._ettin_head["dense2_weight"]  # (1, hidden)
        dense2_bias = self._ettin_head.get("dense2_bias", np.zeros(dense2_weight.shape[0]))
        logits = x @ dense2_weight.T + dense2_bias  # (batch, 1)
        return logits.reshape(-1)  # (batch,)

    def _extract_logits(self, output: list, model_path: str) -> np.ndarray:
        """Extract and validate logits from ONNX model output.

        Accepts only logits shaped (batch,), (batch, 1), or (batch, 2).
        For (batch, 2), returns the positive-class logit (index 1).
        For ettin-reranker models with 3D output, applies the scoring head.
        Raises ValueError with a clear message for any other shape.
        """

        arr = output[0]
        ndim = arr.ndim
        shape = arr.shape

        # Handle ettin-reranker with 3D output (last_hidden_state)
        if ndim == 3 and self._ettin_head is not None:
            return self._apply_ettin_head(arr)

        if ndim == 1:
            # (batch,) - already logits
            return arr
        elif ndim == 2:
            batch, dim = shape
            if dim == 1:
                # (batch, 1) - squeeze to (batch,)
                return arr.reshape(batch)
            elif dim == 2:
                # (batch, 2) - binary classification, use positive class (index 1)
                return arr[:, 1]
            else:
                raise ValueError(
                    f"Cross-encoder model at {model_path} produced output shape {shape} "
                    f"(output name: {self._session.get_outputs()[0].name}). "
                    f"Expected logits shaped (batch,), (batch, 1), or (batch, 2). "
                    f"Got (batch, {dim}) -- this looks like pooled hidden states, not logits. "
                    f"Ensure the ONNX export includes the scoring head."
                )
        else:
            raise ValueError(
                f"Cross-encoder model at {model_path} produced {ndim}-D output shape {shape} "
                f"(output name: {self._session.get_outputs()[0].name}). "
                f"Expected logits shaped (batch,), (batch, 1), or (batch, 2). "
                f"This looks like last_hidden_state (batch, seq, hidden) -- "
                f"the ONNX export is missing the scoring head. "
                f"For ettin-reranker models, export the full CrossEncoder pipeline "
                f"or ensure safetensors with the scoring head are present."
            )

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

        session_input_names = [inp.name for inp in self._session.get_inputs()]
        # transformers >= 5 stopped emitting token_type_ids by default; the
        # ONNX export still requires them for sentence pairs, so request
        # them explicitly when the session declares the input.
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
            # Cross-encoder outputs logits; higher = more relevant
            logits = self._extract_logits(output, self._onnx_path)
            score = float(logits[0])
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
