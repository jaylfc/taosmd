from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest


def test_image_rows_l2_normalised(tmp_path, monkeypatch):
    from benchmarks.e048 import controls
    from PIL import Image

    emb = controls.OpenClipImageEmbedder()
    emb._device = "cpu"

    class DummyModel:
        def encode_text(self, tokens):
            import torch

            return torch.tensor([[3.0, 4.0], [0.0, 5.0]], dtype=torch.float32)

        def encode_image(self, batch):
            import torch

            return torch.tensor([[1.0, 2.0, 2.0]], dtype=torch.float32)

    class DummyTokenizer:
        def __call__(self, texts):
            import torch

            return torch.tensor([[1, 2]], dtype=torch.long)

    import torch

    emb._model = DummyModel()
    emb._tokenizer = DummyTokenizer()
    emb._torch = torch

    q = emb.embed_queries(["a", "b"])
    assert q.shape == (2, 2)
    assert q.dtype == np.float32
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-5)

    emb._preprocess = lambda x: torch.tensor([0.1, 0.2, 0.3])
    monkeypatch.setattr(Image, "open", lambda p: type('I', (), {'convert': lambda self, mode: 'img'})())
    items = emb.embed_items(["/tmp/fake.jpg"])
    assert np.allclose(np.linalg.norm(items, axis=1), 1.0, atol=1e-5)


def test_audio_48k_downmix():
    from benchmarks.e048 import controls

    import soundfile as sf

    def fake_sf_read(path, dtype="float32", always_2d=True):
        data = np.column_stack([
            np.full(48000, 1.0, dtype=np.float32),
            np.full(48000, 3.0, dtype=np.float32),
        ])
        return data, 48000

    orig = sf.read
    try:
        sf.read = fake_sf_read
        emb = controls.ClapAudioEmbedder()
        arr = emb._load_audio("/fake.wav")
        assert arr.shape == (48000,)
        assert np.allclose(arr, 2.0)
    finally:
        sf.read = orig


def test_audio_resample_non48k():
    from benchmarks.e048 import controls

    import soundfile as sf
    import torchaudio.functional as F
    import torch

    def fake_sf_read(path, dtype="float32", always_2d=True):
        data = np.column_stack([np.ones(16000, dtype=np.float32), np.ones(16000, dtype=np.float32)])
        return data, 16000

    def fake_resample(waveform, orig_freq, new_freq):
        return torch.ones(12345, dtype=torch.float32)

    orig_sf = sf.read
    orig_res = F.resample
    try:
        sf.read = fake_sf_read
        F.resample = fake_resample
        emb = controls.ClapAudioEmbedder()
        arr = emb._load_audio("/fake.wav")
        assert arr.shape == (12345,)
    finally:
        sf.read = orig_sf
        F.resample = orig_res


def test_audio_resample_requires_torchaudio():
    from benchmarks.e048 import controls

    import soundfile as sf

    def fake_sf_read(path, dtype="float32", always_2d=True):
        data = np.column_stack([np.ones(16000, dtype=np.float32), np.ones(16000, dtype=np.float32)])
        return data, 16000

    orig_sf = sf.read
    try:
        sf.read = fake_sf_read
        sys.modules["torchaudio"] = None
        emb = controls.ClapAudioEmbedder()
        with pytest.raises(ImportError):
            emb._load_audio("/fake.wav")
    finally:
        sf.read = orig_sf
        sys.modules.pop("torchaudio", None)


def test_clap_provenance(tmp_path):
    from benchmarks.e048 import controls

    snap = tmp_path / "snapshots" / "abc123def"
    snap.mkdir(parents=True)
    model_bytes = b"test-model-bytes"
    (snap / "model.safetensors").write_bytes(model_bytes)

    import torch

    emb = controls.ClapAudioEmbedder.__new__(controls.ClapAudioEmbedder)
    emb._device = "cpu"
    emb._model = "fake"
    emb._model_id = "laion/clap-htsat-unfused"
    emb._snapshot_path = str(snap)
    emb._revision = snap.name
    emb._torch = torch
    prov = emb.provenance()
    assert prov["revision"] == "abc123def"
    assert prov["model_files"]["model.safetensors"] == hashlib.sha256(model_bytes).hexdigest()


def test_clap_provenance_empty_raises(tmp_path):
    from benchmarks.e048 import controls

    import torch

    snap = tmp_path / "snapshots" / "deadbeef"
    snap.mkdir(parents=True)
    emb = controls.ClapAudioEmbedder.__new__(controls.ClapAudioEmbedder)
    emb._device = "cpu"
    emb._model = "fake"
    emb._model_id = "laion/clap-htsat-unfused"
    emb._snapshot_path = str(snap)
    emb._revision = snap.name
    emb._torch = torch
    with pytest.raises(RuntimeError):
        emb.provenance()


def test_openclip_provenance(tmp_path):
    from benchmarks.e048 import controls

    w = tmp_path / "vitb32_laion2b.bin"
    w.write_bytes(b"clip-weights")
    import torch

    emb = controls.OpenClipImageEmbedder.__new__(controls.OpenClipImageEmbedder)
    emb._model = "fake"
    emb._model_id = "ViT-B-32/laion2b_s34b_b79k"
    emb._revision = "laion2b_s34b_b79k"
    emb._weight_path = str(w)
    emb._torch = torch
    emb._device = "cpu"
    prov = emb.provenance()
    assert prov["model_files"][Path(w).name] == hashlib.sha256(b"clip-weights").hexdigest()


def test_openclip_provenance_missing_raises(tmp_path):
    from benchmarks.e048 import controls

    import torch

    emb = controls.OpenClipImageEmbedder.__new__(controls.OpenClipImageEmbedder)
    emb._model = "fake"
    emb._model_id = "ViT-B-32/laion2b_s34b_b79k"
    emb._revision = "laion2b_s34b_b79k"
    emb._weight_path = str(tmp_path / "nope.bin")
    emb._torch = torch
    emb._device = "cpu"
    with pytest.raises(RuntimeError):
        emb.provenance()


def test_grep_guard():
    controls_path = Path(__file__).parent.parent / "benchmarks" / "e048" / "controls.py"
    text = controls_path.read_text()
    assert "dummy_hash" not in text
    assert "hexdigest()[:12]" not in text
