from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


class FakeTensor:
    """A fake tensor that supports the operations needed by the embedders."""
    def __init__(self, arr):
        self._arr = np.asarray(arr)
    
    def numpy(self):
        return self._arr
    
    def to(self, device):
        return self
    
    def cpu(self):
        return self
    
    def norm(self, dim=-1, keepdim=False):
        result = np.linalg.norm(self._arr, axis=dim, keepdims=keepdim)
        return FakeTensor(result)
    
    def __truediv__(self, other):
        if isinstance(other, FakeTensor):
            return FakeTensor(self._arr / other._arr)
        return FakeTensor(self._arr / other)
    
    def __getattr__(self, name):
        # Delegate to underlying array for any other attributes
        return getattr(self._arr, name)
    
    @property
    def shape(self):
        return self._arr.shape
    
    @property
    def dtype(self):
        return self._arr.dtype


def _make_fake_tensor(arr):
    """Create a fake tensor object."""
    return FakeTensor(arr)


def _install_fake_torch(monkeypatch):
    fake_torch = MagicMock()
    fake_torch.__version__ = "2.0.0"
    fake_torch.cuda.is_available.return_value = False
    fake_torch.tensor = lambda *a, **k: _make_fake_tensor(np.array(a, dtype=np.float32))
    fake_torch.no_grad = lambda: MagicMock(__enter__=lambda s: None, __exit__=lambda s, *a: None)
    fake_torch.float32 = np.float32
    fake_torch.long = np.int64
    fake_torch.stack = lambda arrays, dim=0: _make_fake_tensor(np.stack([a._arr if isinstance(a, FakeTensor) else a for a in arrays], axis=dim))
    fake_torch.from_numpy = lambda x: _make_fake_tensor(x)
    fake_torch.nn = MagicMock()
    fake_torch.nn.Module = object
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.nn", fake_torch.nn)
    return fake_torch


def _install_fake_torchaudio(monkeypatch):
    fake_torchaudio = MagicMock()
    fake_functional = MagicMock()
    # Return a fake tensor that has .numpy() method
    fake_functional.resample = lambda waveform, orig_freq, new_freq: _make_fake_tensor(np.ones(12345, dtype=np.float32))
    fake_torchaudio.functional = fake_functional
    monkeypatch.setitem(sys.modules, "torchaudio", fake_torchaudio)
    monkeypatch.setitem(sys.modules, "torchaudio.functional", fake_functional)
    return fake_torchaudio


def _install_fake_soundfile(monkeypatch):
    fake_sf = MagicMock()
    fake_sf.read = lambda path, dtype="float32", always_2d=True: (
        np.column_stack([np.full(48000, 1.0, dtype=np.float32), np.full(48000, 3.0, dtype=np.float32)]),
        48000,
    )
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)
    return fake_sf


def _install_fake_pil(monkeypatch):
    fake_pil = MagicMock()
    fake_image = MagicMock()
    fake_image.convert.return_value = "rgb"
    fake_pil.Image.open = lambda p: fake_image
    fake_pil.Image.__name__ = "Image"
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_pil.Image)
    return fake_pil


def _install_fake_open_clip(monkeypatch, tmp_path):
    fake_open_clip = MagicMock()
    fake_open_clip.get_pretrained_cfg = lambda *a, **k: MagicMock()

    weight_file = tmp_path / "vitb32_laion2b.bin"
    weight_file.write_bytes(b"clip-weights")
    fake_open_clip.pretrained = MagicMock()
    fake_open_clip.pretrained.download_pretrained = lambda *a, **k: str(weight_file)

    fake_model = MagicMock()
    fake_model.encode_text = lambda tokens: _make_fake_tensor(np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32))
    fake_model.encode_image = lambda batch: _make_fake_tensor(np.array([[1.0, 2.0, 2.0]], dtype=np.float32))
    fake_model.eval = lambda: None
    fake_model.to = lambda device: fake_model

    fake_preprocess = lambda img: np.array([0.1, 0.2, 0.3], dtype=np.float32)
    fake_tokenizer = lambda texts: _make_fake_tensor(np.array([[1, 2]], dtype=np.int64))

    fake_open_clip.create_model_and_transforms = lambda *a, **k: (fake_model, None, fake_preprocess)
    fake_open_clip.get_tokenizer = lambda *a, **k: fake_tokenizer

    monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)
    monkeypatch.setitem(sys.modules, "open_clip.pretrained", fake_open_clip.pretrained)
    return fake_open_clip


def _install_fake_transformers(monkeypatch, tmp_path):
    snap_dir = tmp_path / "snapshots" / "abc123def"
    snap_dir.mkdir(parents=True, exist_ok=True)
    model_bytes = b"test-model-bytes-for-clap"
    (snap_dir / "model.safetensors").write_bytes(model_bytes)

    class FakeClapModel:
        @classmethod
        def from_pretrained(cls, path):
            return cls()
        def eval(self): return None
        def to(self, device): return self
        def get_text_features(self, **kwargs): return _make_fake_tensor(np.array([[0.6, 0.8]], dtype=np.float32))
        def get_audio_features(self, **kwargs): return _make_fake_tensor(np.array([[0.6, 0.8]], dtype=np.float32))

    class FakeClapProcessor:
        def __init__(self):
            self._call_log = []
        @classmethod
        def from_pretrained(cls, path):
            return cls()
        def __call__(self, **kwargs):
            self._call_log.append(kwargs)
            return {"input_features": _make_fake_tensor(np.array([[[0.0]]], dtype=np.float32))}

    fake_clap_model = FakeClapModel()
    fake_clap_processor = FakeClapProcessor()

    # Use a simple object instead of MagicMock to avoid attribute mocking
    class FakeTransformers:
        pass
    fake_transformers = FakeTransformers()
    fake_transformers.ClapModel = type('ClapModel', (), {
        'from_pretrained': classmethod(lambda cls, path: fake_clap_model),
        'eval': lambda self: None,
        'to': lambda self, device: self,
        'get_text_features': lambda self, **kwargs: _make_fake_tensor(np.array([[0.6, 0.8]], dtype=np.float32)),
        'get_audio_features': lambda self, **kwargs: _make_fake_tensor(np.array([[0.6, 0.8]], dtype=np.float32)),
    })
    fake_transformers.ClapProcessor = type('ClapProcessor', (), {
        'from_pretrained': classmethod(lambda cls, path: fake_clap_processor),
    })
    fake_transformers._fake_clap_processor_instance = fake_clap_processor
    fake_transformers._fake_clap_model_instance = fake_clap_model

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.models.clap", MagicMock())
    return fake_transformers


def _install_fake_huggingface_hub(monkeypatch, tmp_path):
    fake_hf_hub = MagicMock()
    snap_dir = tmp_path / "snapshots" / "abc123def"
    snap_dir.mkdir(parents=True, exist_ok=True)
    (snap_dir / "model.safetensors").write_bytes(b"test-model-bytes-for-clap")
    fake_hf_hub.snapshot_download = lambda model_id, **kwargs: str(snap_dir)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_hub)
    return fake_hf_hub


def _install_all_fakes(monkeypatch, tmp_path):
    _install_fake_torch(monkeypatch)
    _install_fake_torchaudio(monkeypatch)
    _install_fake_soundfile(monkeypatch)
    _install_fake_pil(monkeypatch)
    _install_fake_open_clip(monkeypatch, tmp_path)
    _install_fake_transformers(monkeypatch, tmp_path)
    _install_fake_huggingface_hub(monkeypatch, tmp_path)


def test_image_rows_l2_normalised(tmp_path, monkeypatch):
    _install_all_fakes(monkeypatch, tmp_path)
    from benchmarks.e048 import controls

    emb = controls.OpenClipImageEmbedder()
    emb._device = "cpu"

    q = emb.embed_queries(["a", "b"])
    assert q.shape == (2, 2)
    assert q.dtype == np.float32
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-5)

    items = emb.embed_items(["/tmp/fake.jpg"])
    assert np.allclose(np.linalg.norm(items, axis=1), 1.0, atol=1e-5)


def test_audio_48k_downmix(tmp_path, monkeypatch):
    _install_all_fakes(monkeypatch, tmp_path)
    from benchmarks.e048 import controls

    fake_sf = sys.modules["soundfile"]
    fake_sf.read = lambda path, dtype="float32", always_2d=True: (
        np.column_stack([np.full(48000, 1.0, dtype=np.float32), np.full(48000, 3.0, dtype=np.float32)]),
        48000,
    )

    emb = controls.ClapAudioEmbedder()
    emb._device = "cpu"
    arr = emb._load_audio("/fake.wav")
    assert arr.shape == (48000,)
    assert np.allclose(arr, 2.0)


def test_audio_resample_non48k(tmp_path, monkeypatch):
    _install_all_fakes(monkeypatch, tmp_path)
    from benchmarks.e048 import controls

    fake_sf = sys.modules["soundfile"]
    fake_sf.read = lambda path, dtype="float32", always_2d=True: (
        np.column_stack([np.ones(16000, dtype=np.float32), np.ones(16000, dtype=np.float32)]),
        16000,
    )

    emb = controls.ClapAudioEmbedder()
    emb._device = "cpu"
    arr = emb._load_audio("/fake.wav")
    assert arr.shape == (12345,)


def test_audio_resample_requires_torchaudio(tmp_path, monkeypatch):
    _install_all_fakes(monkeypatch, tmp_path)
    from benchmarks.e048 import controls

    fake_sf = sys.modules["soundfile"]
    fake_sf.read = lambda path, dtype="float32", always_2d=True: (
        np.column_stack([np.ones(16000, dtype=np.float32), np.ones(16000, dtype=np.float32)]),
        16000,
    )

    monkeypatch.delitem(sys.modules, "torchaudio", raising=False)
    monkeypatch.delitem(sys.modules, "torchaudio.functional", raising=False)

    emb = controls.ClapAudioEmbedder()
    with pytest.raises(ImportError, match="torchaudio is required"):
        emb._load_audio("/fake.wav")


def test_clap_provenance(tmp_path, monkeypatch):
    _install_all_fakes(monkeypatch, tmp_path)
    from benchmarks.e048 import controls

    emb = controls.ClapAudioEmbedder()
    emb._device = "cpu"

    prov = emb.provenance()
    assert prov["revision"] == "abc123def"
    expected_hash = hashlib.sha256(b"test-model-bytes-for-clap").hexdigest()
    assert prov["model_files"]["model.safetensors"] == expected_hash


def test_clap_provenance_empty_raises(tmp_path, monkeypatch):
    _install_all_fakes(monkeypatch, tmp_path)
    from benchmarks.e048 import controls
    import huggingface_hub

    empty_snap = tmp_path / "snapshots" / "deadbeef"
    empty_snap.mkdir(parents=True)
    huggingface_hub.snapshot_download = lambda model_id, **kwargs: str(empty_snap)

    emb = controls.ClapAudioEmbedder()
    emb._device = "cpu"
    with pytest.raises(RuntimeError, match="No weight file"):
        emb.provenance()


def test_openclip_provenance(tmp_path, monkeypatch):
    _install_all_fakes(monkeypatch, tmp_path)
    from benchmarks.e048 import controls

    emb = controls.OpenClipImageEmbedder()
    emb._device = "cpu"

    prov = emb.provenance()
    expected_hash = hashlib.sha256(b"clip-weights").hexdigest()
    assert prov["model_files"]["vitb32_laion2b.bin"] == expected_hash


def test_openclip_provenance_missing_raises(tmp_path, monkeypatch):
    _install_all_fakes(monkeypatch, tmp_path)
    from benchmarks.e048 import controls
    import open_clip

    open_clip.pretrained.download_pretrained = lambda *a, **k: str(tmp_path / "nope.bin")

    emb = controls.OpenClipImageEmbedder()
    emb._device = "cpu"
    with pytest.raises(RuntimeError, match="OpenCLIP weight file not found"):
        emb.provenance()


def test_clap_embed_items_processor_kwargs_48k(tmp_path, monkeypatch):
    _install_all_fakes(monkeypatch, tmp_path)
    from benchmarks.e048 import controls

    fake_transformers = sys.modules["transformers"]
    fake_processor = fake_transformers._fake_clap_processor_instance

    fake_sf = sys.modules["soundfile"]
    fake_sf.read = lambda path, dtype="float32", always_2d=True: (
        np.column_stack([np.full(48000, 1.0, dtype=np.float32), np.full(48000, 3.0, dtype=np.float32)]),
        48000,
    )

    emb = controls.ClapAudioEmbedder()
    emb._device = "cpu"
    emb.embed_items(["/fake.wav"])

    recorded_kwargs = fake_processor._call_log[-1]
    assert "audios" in recorded_kwargs or "audio" in recorded_kwargs
    audio_key = "audios" if "audios" in recorded_kwargs else "audio"
    audio_arr = recorded_kwargs[audio_key][0]
    assert audio_arr.shape == (48000,)
    assert np.allclose(audio_arr, 2.0)
    assert recorded_kwargs["sampling_rate"] == 48000


def test_clap_embed_items_processor_kwargs_resampled(tmp_path, monkeypatch):
    _install_all_fakes(monkeypatch, tmp_path)
    from benchmarks.e048 import controls

    fake_transformers = sys.modules["transformers"]
    fake_processor = fake_transformers._fake_clap_processor_instance

    fake_sf = sys.modules["soundfile"]
    fake_sf.read = lambda path, dtype="float32", always_2d=True: (
        np.column_stack([np.ones(16000, dtype=np.float32), np.ones(16000, dtype=np.float32)]),
        16000,
    )

    fake_torchaudio = sys.modules["torchaudio"]
    fake_torchaudio.functional.resample = lambda waveform, orig_freq, new_freq: _make_fake_tensor(np.ones(12345, dtype=np.float32))

    emb = controls.ClapAudioEmbedder()
    emb._device = "cpu"
    emb.embed_items(["/fake.wav"])

    recorded_kwargs = fake_processor._call_log[-1]
    assert "audios" in recorded_kwargs or "audio" in recorded_kwargs
    audio_key = "audios" if "audios" in recorded_kwargs else "audio"
    audio_arr = recorded_kwargs[audio_key][0]
    assert audio_arr.shape == (12345,)
    assert recorded_kwargs["sampling_rate"] == 48000


def test_grep_guard():
    controls_path = Path(__file__).parent.parent / "benchmarks" / "e048" / "controls.py"
    text = controls_path.read_text()
    assert "dummy_hash" not in text
    assert "hexdigest()[:12]" not in text