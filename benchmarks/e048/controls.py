"""Control embedders for E-048: OpenCLIP ViT-B/32 image and LAION CLAP audio."""

from __future__ import annotations

import hashlib
import os
import pathlib

import numpy as np

from benchmarks.e048.retrieval import MediaEmbedder


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return (arr / norms).astype(np.float32)


def _is_torch_tensor(x) -> bool:
    return hasattr(x, "norm") and callable(getattr(x, "norm", None))


class OpenClipImageEmbedder(MediaEmbedder):
    modality = "image"

    def __init__(self, device: str = "auto"):
        self._device = device
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._torch = None
        self._open_clip = None
        self._model_id = "ViT-B-32/laion2b_s34b_b79k"
        self._weight_path = None
        self._revision = "laion2b_s34b_b79k"

    @property
    def name(self) -> str:
        return "openclip-vitb32-laion2b"

    def _lazy_init(self):
        if self._model is not None:
            return

        import open_clip
        import torch

        self._open_clip = open_clip
        self._torch = torch

        cache_dir = os.environ.get("OPEN_CLIP_CACHE_DIR")
        cfg = open_clip.get_pretrained_cfg("ViT-B-32", "laion2b_s34b_b79k")
        weight_path = open_clip.pretrained.download_pretrained(cfg, cache_dir=cache_dir)
        self._weight_path = weight_path

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=weight_path
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        model.eval()

        if self._device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self._device

        model = model.to(device)
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = tokenizer
        self._device = device

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        self._lazy_init()
        with self._torch.no_grad():
            tokens = self._tokenizer(texts)
            if hasattr(tokens, "to"):
                tokens = tokens.to(self._device)
            features = self._model.encode_text(tokens)
            if _is_torch_tensor(features):
                features = features / features.norm(dim=-1, keepdim=True)
                return features.cpu().numpy().astype(np.float32)
            else:
                return _normalize_rows(features)

    def embed_items(self, paths: list[str]) -> np.ndarray:
        self._lazy_init()
        from PIL import Image

        images = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            images.append(self._preprocess(img))

        if hasattr(self._torch, "stack"):
            batch = self._torch.stack(images)
        else:
            batch = np.stack(images)
        if hasattr(batch, "to"):
            batch = batch.to(self._device)
        with self._torch.no_grad():
            features = self._model.encode_image(batch)
            if _is_torch_tensor(features):
                features = features / features.norm(dim=-1, keepdim=True)
                return features.cpu().numpy().astype(np.float32)
            else:
                return _normalize_rows(features)

    def provenance(self) -> dict:
        self._lazy_init()

        model_files = {}
        if self._weight_path and os.path.exists(self._weight_path):
            h = hashlib.sha256()
            with open(self._weight_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            model_files[os.path.basename(self._weight_path)] = h.hexdigest()
        else:
            raise RuntimeError(f"OpenCLIP weight file not found at {self._weight_path}")

        return {
            "model_id": self._model_id,
            "revision": self._revision,
            "model_files": model_files,
            "runtime": "torch",
            "runtime_version": self._torch.__version__,
            "execution_provider": self._device,
            "dtype": "fp32",
        }


class ClapAudioEmbedder(MediaEmbedder):
    modality = "audio"

    def __init__(self, device: str = "auto"):
        self._device = device
        self._model = None
        self._processor = None
        self._torch = None
        self._transformers = None
        self._model_id = "laion/clap-htsat-unfused"
        self._revision = None
        self._snapshot_path = None
        self._processor_arg_name = "audios"

    @property
    def name(self) -> str:
        return "laion-clap-htsat-unfused"

    def _lazy_init(self):
        if self._model is not None:
            return

        import inspect

        import huggingface_hub
        import torch
        import transformers
        from transformers import ClapModel, ClapProcessor

        self._torch = torch
        self._transformers = transformers

        snapshot_path = huggingface_hub.snapshot_download(self._model_id)
        self._snapshot_path = snapshot_path
        self._model = ClapModel.from_pretrained(snapshot_path)
        self._processor = ClapProcessor.from_pretrained(snapshot_path)
        self._model.eval()

        sig = inspect.signature(self._processor.__call__)
        if "audio" in sig.parameters and "audios" not in sig.parameters:
            self._processor_arg_name = "audio"
        else:
            self._processor_arg_name = "audios"

        self._revision = pathlib.Path(snapshot_path).name

        if self._device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self._device

        self._model = self._model.to(device)
        self._device = device

    def _load_audio(self, path: str) -> np.ndarray:
        import soundfile as sf
        import torch

        waveform, sr = sf.read(path, dtype="float32", always_2d=True)
        waveform = waveform.mean(axis=1)
        if sr != 48000:
            try:
                import torchaudio.functional as F
            except ImportError as e:
                raise ImportError("torchaudio is required for resampling audio to 48kHz") from e
            waveform = F.resample(
                torch.from_numpy(waveform), orig_freq=sr, new_freq=48000
            ).numpy()
        return waveform

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        self._lazy_init()
        inputs = self._processor(
            text=texts, return_tensors="pt", padding=True
        )
        inputs = {k: (v.to(self._device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with self._torch.no_grad():
            features = self._model.get_text_features(**inputs)
            if _is_torch_tensor(features):
                features = features / features.norm(dim=-1, keepdim=True)
                return features.cpu().numpy().astype(np.float32)
            else:
                return _normalize_rows(features)

    def embed_items(self, paths: list[str]) -> np.ndarray:
        self._lazy_init()
        audios = []
        for p in paths:
            waveform = self._load_audio(p)
            audios.append(waveform)

        processor_kwargs = {
            self._processor_arg_name: audios,
            "sampling_rate": 48000,
            "return_tensors": "pt",
        }
        inputs = self._processor(**processor_kwargs)
        inputs = {k: (v.to(self._device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with self._torch.no_grad():
            features = self._model.get_audio_features(**inputs)
            if _is_torch_tensor(features):
                features = features / features.norm(dim=-1, keepdim=True)
                return features.cpu().numpy().astype(np.float32)
            else:
                return _normalize_rows(features)

    def provenance(self) -> dict:
        self._lazy_init()

        model_files = {}
        if self._snapshot_path and os.path.exists(self._snapshot_path):
            snap_dir = pathlib.Path(self._snapshot_path)
            weight_file = None
            for f in snap_dir.glob("*.safetensors"):
                weight_file = f
                break
            if weight_file is None:
                for f in snap_dir.glob("*.bin"):
                    weight_file = f
                    break
            if weight_file is None:
                raise RuntimeError(f"No weight file (.safetensors or .bin) found in {snap_dir}")

            h = hashlib.sha256()
            with weight_file.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    h.update(chunk)
            model_files[weight_file.name] = h.hexdigest()
        else:
            raise RuntimeError(f"CLAP snapshot not found at {self._snapshot_path}")

        return {
            "model_id": self._model_id,
            "revision": self._revision,
            "model_files": model_files,
            "runtime": "torch",
            "runtime_version": self._torch.__version__,
            "execution_provider": self._device,
            "dtype": "fp32",
        }


def openclip_image(**kwargs) -> OpenClipImageEmbedder:
    return OpenClipImageEmbedder(**kwargs)


def clap_audio(**kwargs) -> ClapAudioEmbedder:
    return ClapAudioEmbedder(**kwargs)
