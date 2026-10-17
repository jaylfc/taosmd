"""Qwen3-VL-Embedding-2B image embedder for E-048."""

from __future__ import annotations

import hashlib
import os

import numpy as np

from benchmarks.e048.retrieval import MediaEmbedder


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    """Row-L2 normalization copy; do not import the private name from controls."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return (arr / norms).astype(np.float32)


class Qwen3VLImageEmbedder(MediaEmbedder):
    modality = "image"
    REVISION = "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
    MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"

    def __init__(self, device: str = "auto"):
        self._device = device
        self._model = None

    @property
    def name(self) -> str:
        return "qwen3-vl-embedding-2b"

    def _lazy_init(self):
        if self._model is not None:
            return

        import sentence_transformers
        import torch
        import huggingface_hub

        self._sentence_transformers = sentence_transformers
        self._torch = torch
        self._huggingface_hub = huggingface_hub

        device = "cuda" if torch.cuda.is_available() else "cpu" if self._device == "auto" else self._device

        model = sentence_transformers.SentenceTransformer(
            self.MODEL_ID, revision=self._REVISION, device=device
        )
        model.eval()
        self._model = model
        self._device = device

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        self._lazy_init()
        with self._torch.no_grad():
            embeddings = self._model.encode(texts, normalize_embeddings=True)
            embeddings = _normalize_rows(np.asarray(embeddings, dtype=np.float32))
            return embeddings

    def embed_items(self, paths: list[str]) -> np.ndarray:
        self._lazy_init()
        # Qwen3-VL: pass each item as a dict {"image": path}
        # a bare string is treated as TEXT, so never pass the path string alone
        items = [{"image": p} for p in paths]
        with self._torch.no_grad():
            embeddings = self._model.encode(items, normalize_embeddings=True)
            embeddings = _normalize_rows(np.asarray(embeddings, dtype=np.float32))
            return embeddings

    def provenance(self) -> dict:
        self._lazy_init()

        model_files = {}
        snapshot_dir = self._huggingface_hub.snapshot_download(
            self.MODEL_ID, revision=self._REVISION
        )
        # Collect all *.safetensors files, filenames relative to snapshot dir, sorted
        safetensors_files = []
        for root, dirs, files in os.walk(snapshot_dir):
            for f in files:
                if f.endswith(".safetensors"):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, snapshot_dir)
                    safetensors_files.append((rel_path, full_path))

        # Sort by relative path
        safetensors_files.sort(key=lambda x: x[0])

        for rel_path, full_path in safetensors_files:
            h = hashlib.sha256()
            with open(full_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            model_files[rel_path] = h.hexdigest()

        return {
            "model_id": self.MODEL_ID,
            "revision": self._REVISION,
            "model_files": model_files,
            "runtime": "sentence-transformers",
            "runtime_version": self._sentence_transformers.__version__,
            "execution_provider": self._device,
            "dtype": "float32",
        }