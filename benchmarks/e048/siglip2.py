"""SigLIP2 image embedder for E-048.

Needs transformers>=4.57.0 at run time.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

from benchmarks.e048.retrieval import MediaEmbedder

MODEL_ID = "google/siglip2-base-patch16-224"
REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    """Row-L2 normalization copy; do not import the private name from controls."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return (arr / norms).astype(np.float32)


def _extract_embedding(output) -> np.ndarray:
    """Extract pooled embedding from transformers model output.

    Handles both tensor outputs and BaseModelOutputWithPooling objects.
    """
    if hasattr(output, "pooler_output"):
        pooled = output.pooler_output
        if hasattr(pooled, "detach"):
            return pooled.detach().cpu().numpy()
        return np.asarray(pooled)
    if hasattr(output, "last_hidden_state"):
        # Fallback: mean pool last hidden state
        last_hidden = output.last_hidden_state
        if hasattr(last_hidden, "detach"):
            return last_hidden.detach().cpu().numpy().mean(axis=1)
        return np.asarray(last_hidden).mean(axis=1)
    # Already a tensor
    if hasattr(output, "detach"):
        return output.detach().cpu().numpy()
    return np.asarray(output)


class Siglip2ImageEmbedder(MediaEmbedder):
    modality = "image"

    def __init__(self, device: str = "auto"):
        self._device = device
        self._model = None
        self._processor = None
        self._torch = None
        self._transformers = None
        self._huggingface_hub = None
        self._snapshot_path = None

    @property
    def name(self) -> str:
        return "siglip2-base-p16-224"

    def _lazy_init(self):
        if self._model is not None:
            return

        import huggingface_hub
        import torch
        import transformers
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        self._transformers = transformers
        self._huggingface_hub = huggingface_hub

        if self._device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self._device

        self._snapshot_path = huggingface_hub.snapshot_download(
            MODEL_ID, revision=REVISION
        )
        self._model = AutoModel.from_pretrained(
            MODEL_ID, revision=REVISION
        )
        self._processor = AutoProcessor.from_pretrained(
            MODEL_ID, revision=REVISION
        )
        self._model.eval()
        self._model = self._model.to(device)
        self._device = device

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        self._lazy_init()
        if not texts:
            # Empty input: return empty array with correct embedding dimension
            hidden_size = self._model.config.hidden_size
            return np.zeros((0, hidden_size), dtype=np.float32)

        all_embeddings = []
        for i in range(0, len(texts), 32):
            batch = texts[i : i + 32]
            inputs = self._processor(
                text=batch,
                padding="max_length",
                max_length=64,
                return_tensors="pt",
            )
            inputs = {
                k: (v.to(self._device) if hasattr(v, "to") else v)
                for k, v in inputs.items()
            }
            with self._torch.no_grad():
                outputs = self._model.get_text_features(**inputs)
                embeddings = _extract_embedding(outputs)
            all_embeddings.append(embeddings)

        embeddings = np.concatenate(all_embeddings, axis=0)
        return _normalize_rows(embeddings.astype(np.float32))

    def embed_items(self, paths: list[str]) -> np.ndarray:
        self._lazy_init()
        if not paths:
            hidden_size = self._model.config.hidden_size
            return np.zeros((0, hidden_size), dtype=np.float32)

        all_embeddings = []
        for i in range(0, len(paths), 32):
            batch_paths = paths[i : i + 32]
            images = []
            from PIL import Image

            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                images.append(img)

            inputs = self._processor(images=images, return_tensors="pt")
            inputs = {
                k: (v.to(self._device) if hasattr(v, "to") else v)
                for k, v in inputs.items()
            }
            with self._torch.no_grad():
                outputs = self._model.get_image_features(**inputs)
                embeddings = _extract_embedding(outputs)
            all_embeddings.append(embeddings)

        embeddings = np.concatenate(all_embeddings, axis=0)
        return _normalize_rows(embeddings.astype(np.float32))

    def provenance(self) -> dict:
        self._lazy_init()

        model_files = {}
        snapshot_dir = self._snapshot_path
        safetensors_files = []
        for root, dirs, files in os.walk(snapshot_dir):
            for f in files:
                if f.endswith(".safetensors"):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, snapshot_dir)
                    safetensors_files.append((rel_path, full_path))

        # Sort by relative path for deterministic ordering
        safetensors_files.sort(key=lambda x: x[0])

        if not safetensors_files:
            raise RuntimeError(
                f"No *.safetensors files found in snapshot at {snapshot_dir}"
            )

        for rel_path, full_path in safetensors_files:
            h = hashlib.sha256()
            with open(full_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            model_files[rel_path] = h.hexdigest()

        return {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "model_files": model_files,
            "runtime": "transformers",
            "runtime_version": self._transformers.__version__,
            "execution_provider": self._device,
            "dtype": "float32",
        }