"""SigLIP2 base-patch16-224 image embedder for E-048."""

from __future__ import annotations

import hashlib
import os

import numpy as np

from benchmarks.e048.retrieval import MediaEmbedder


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return (arr / norms).astype(np.float32)


def _extract_embedding(features):
    # Handle transformers output objects that wrap embeddings
    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output
    if hasattr(features, "text_embeds") and features.text_embeds is not None:
        return features.text_embeds
    if hasattr(features, "image_embeds") and features.image_embeds is not None:
        return features.image_embeds
    if hasattr(features, "proj") and features.proj is not None:
        return features.proj
    return features


MODEL_ID = "google/siglip2-base-patch16-224"
REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"


class Siglip2ImageEmbedder(MediaEmbedder):
    modality = "image"

    def __init__(self, device: str = "auto"):
        self._device = device
        self._model = None
        self._processor = None
        self._torch = None
        self._transformers = None
        self._huggingface_hub = None

    @property
    def name(self) -> str:
        return "siglip2-base-p16-224"

    def _lazy_init(self):
        if self._model is not None:
            return

        import torch
        import transformers
        import huggingface_hub
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        self._transformers = transformers
        self._huggingface_hub = huggingface_hub

        self._model = AutoModel.from_pretrained(
            MODEL_ID, revision=REVISION
        )
        self._processor = AutoProcessor.from_pretrained(
            MODEL_ID, revision=REVISION
        )
        self._model.eval()

        if self._device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self._device

        self._model = self._model.to(device)
        self._device = device

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        self._lazy_init()
        inputs = self._processor(
            text=texts,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        )
        inputs = {
            k: (v.to(self._device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }
        with self._torch.no_grad():
            features = self._model.get_text_features(**inputs)
            features = _extract_embedding(features)
            if hasattr(features, "norm"):
                features = features / features.norm(dim=-1, keepdim=True)
                return features.cpu().numpy().astype(np.float32)
            else:
                return _normalize_rows(features)

    def embed_items(self, paths: list[str]) -> np.ndarray:
        self._lazy_init()
        from PIL import Image

        if len(paths) == 0:
            # Determine embedding dimension
            config = getattr(self._model, "config", None)
            dim = getattr(config, "embed_dim", None) or getattr(config, "hidden_size", None) or getattr(config, "projection_dim", None) or 0
            return np.zeros((0, dim), dtype=np.float32)

        images = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            images.append(img)

        batches = []
        for i in range(0, len(images), 32):
            batch_images = images[i : i + 32]
            processed = self._processor(
                images=batch_images, return_tensors="pt"
            )
            processed = {
                k: (v.to(self._device) if hasattr(v, "to") else v)
                for k, v in processed.items()
            }
            with self._torch.no_grad():
                batch_features = self._model.get_image_features(**processed)
                batch_features = _extract_embedding(batch_features)
                if hasattr(batch_features, "norm"):
                    batch_features = batch_features / batch_features.norm(
                        dim=-1, keepdim=True
                    )
                    batches.append(batch_features.cpu().numpy().astype(np.float32))
                else:
                    batches.append(
                        _normalize_rows(batch_features)
                    )
        if not batches:
            config = getattr(self._model, "config", None)
            dim = getattr(config, "embed_dim", None) or getattr(config, "hidden_size", None) or getattr(config, "projection_dim", None) or 0
            return np.zeros((0, dim), dtype=np.float32)
        return np.concatenate(batches, axis=0)

    def provenance(self) -> dict:
        self._lazy_init()

        snap_dir = self._huggingface_hub.snapshot_download(
            MODEL_ID, revision=REVISION
        )
        model_files = {}
        for root, _, files in os.walk(snap_dir):
            for f in files:
                if f.endswith(".safetensors"):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, snap_dir)
                    h = hashlib.sha256()
                    with open(full_path, "rb") as fp:
                        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
                            h.update(chunk)
                    model_files[rel_path] = h.hexdigest()

        model_files = dict(sorted(model_files.items()))

        return {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "model_files": model_files,
            "runtime": "transformers",
            "runtime_version": self._transformers.__version__,
            "execution_provider": self._device,
            "dtype": "float32",
        }


def siglip2_image(**kwargs) -> Siglip2ImageEmbedder:
    return Siglip2ImageEmbedder(**kwargs)
