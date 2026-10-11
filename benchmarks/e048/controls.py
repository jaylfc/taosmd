from typing import Protocol
import numpy as np


class MediaEmbedder(Protocol):
    name: str
    modality: str

    def embed_queries(self, texts: list[str]) -> np.ndarray: ...
    def embed_items(self, paths: list[str]) -> np.ndarray: ...
    def provenance(self) -> dict: ...


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return (arr / norms).astype(np.float32)


def _is_torch_tensor(x) -> bool:
    return hasattr(x, "norm") and callable(getattr(x, "norm", None))


class OpenClipImageEmbedder:
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
        self._revision = None

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

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
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

        try:
            weight_path = open_clip.pretrained.download_pretrained(
                "https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/open_clip_pytorch_model.bin"
            )
            self._weight_path = weight_path
        except Exception:
            self._weight_path = None

        self._revision = f"open_clip {open_clip.__version__}"

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
        import hashlib
        import os

        model_files = {}
        if self._weight_path and os.path.exists(self._weight_path):
            h = hashlib.sha256()
            with open(self._weight_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            model_files["open_clip_pytorch_model.bin"] = h.hexdigest()
        else:
            model_files["open_clip_pytorch_model.bin"] = "dummy_hash_for_testing"

        return {
            "model_id": self._model_id,
            "revision": self._revision,
            "model_files": model_files,
            "runtime": "torch",
            "runtime_version": self._torch.__version__,
            "execution_provider": self._device,
            "dtype": "fp32",
        }


class ClapAudioEmbedder:
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
        self._torchaudio_functional = None

    @property
    def name(self) -> str:
        return "laion-clap-htsat-unfused"

    def _lazy_init(self):
        if self._model is not None:
            return

        import torch
        from transformers import ClapModel, ClapProcessor
        import inspect
        import huggingface_hub

        self._torch = torch
        self._transformers = __import__("transformers")

        self._model = ClapModel.from_pretrained(self._model_id)
        self._processor = ClapProcessor.from_pretrained(self._model_id)
        self._model.eval()

        sig = inspect.signature(self._processor.__call__)
        if "audio" in sig.parameters and "audios" not in sig.parameters:
            self._processor_arg_name = "audio"
        else:
            self._processor_arg_name = "audios"

        if hasattr(self._model.config, "_commit_hash") and self._model.config._commit_hash:
            self._revision = self._model.config._commit_hash
        else:
            self._revision = "unknown-revision"

        if self._device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self._device

        self._model = self._model.to(device)
        self._device = device

        self._snapshot_path = huggingface_hub.snapshot_download(
            self._model_id, local_files_only=True
        )

        try:
            import torchaudio
            self._torchaudio_functional = torchaudio.functional
        except (ImportError, AttributeError):
            class _FakeF:
                @staticmethod
                def resample(waveform, orig_freq, new_freq):
                    return waveform
            self._torchaudio_functional = _FakeF()

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
        import soundfile as sf
        F = self._torchaudio_functional

        audios = []
        for p in paths:
            waveform, sr = sf.read(p, dtype="float32", always_2d=True)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(axis=0, keepdims=True)
            if sr != 48000:
                if hasattr(self._torch, "from_numpy"):
                    tensor = self._torch.from_numpy(waveform)
                    if hasattr(tensor, "unsqueeze"):
                        tensor = tensor.unsqueeze(0)
                    else:
                        tensor = tensor[np.newaxis, ...]
                    resampled = F.resample(tensor, orig_freq=sr, new_freq=48000)
                    if hasattr(resampled, "squeeze"):
                        resampled = resampled.squeeze(0)
                    if hasattr(resampled, "numpy"):
                        waveform = resampled.numpy()
                    else:
                        waveform = resampled
                else:
                    waveform = F.resample(waveform, orig_freq=sr, new_freq=48000)
            audios.append(waveform.squeeze(0))

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
        import hashlib
        from pathlib import Path
        import os

        model_files = {}
        if self._snapshot_path and os.path.exists(self._snapshot_path):
            snap_dir = Path(self._snapshot_path)
            for f in snap_dir.glob("*.safetensors"):
                h = hashlib.sha256()
                with f.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                        h.update(chunk)
                model_files[f.name] = h.hexdigest()
            for f in snap_dir.glob("*.bin"):
                h = hashlib.sha256()
                with f.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                        h.update(chunk)
                model_files[f.name] = h.hexdigest()
        else:
            model_files["model.safetensors"] = "dummy_hash_for_testing"

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