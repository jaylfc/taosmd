from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np


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
    fake_torch.stack = lambda arrays, dim=0: _make_fake_tensor(
        np.stack([a._arr if isinstance(a, FakeTensor) else a for a in arrays], axis=dim)
    )
    fake_torch.from_numpy = lambda x: _make_fake_tensor(x)
    fake_torch.nn = MagicMock()
    fake_torch.nn.Module = object
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.nn", fake_torch.nn)
    return fake_torch


def _install_fake_transformers(monkeypatch, tmp_path):
    snap_dir = tmp_path / "snapshots" / "abc123def"
    snap_dir.mkdir(parents=True, exist_ok=True)
    model_bytes = b"test-model-bytes-for-siglip2"
    (snap_dir / "model.safetensors").write_bytes(model_bytes)
    (snap_dir / "config.json").write_bytes(b'{"hidden_size": 128}')

    class FakeModel:
        @classmethod
        def from_pretrained(cls, path, revision=None):
            return cls()
        def eval(self): return None
        def to(self, device): return self
        def get_text_features(self, **kwargs): 
            return _make_fake_tensor(np.array([[0.6, 0.8], [0.3, 0.9]], dtype=np.float32))
        def get_image_features(self, **kwargs):
            return _make_fake_tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        @property
        def config(self):
            class Config:
                hidden_size = 128
            return Config()

    class FakeProcessor:
        def __init__(self):
            self._call_log = []
        @classmethod
        def from_pretrained(cls, path, revision=None):
            return cls()
        def __call__(self, **kwargs):
            self._call_log.append(kwargs)
            if "text" in kwargs:
                return {
                    "input_ids": _make_fake_tensor(np.array([[1, 2, 3]], dtype=np.int64)),
                    "attention_mask": _make_fake_tensor(np.array([[1, 1, 1]], dtype=np.int64)),
                }
            elif "images" in kwargs:
                return {"pixel_values": _make_fake_tensor(np.array([[[0.0]]], dtype=np.float32))}
            else:
                return {}

    fake_transformers = MagicMock()
    fake_transformers.__version__ = "4.30.0"
    fake_transformers.AutoModel = type('AutoModel', (), {
        'from_pretrained': classmethod(lambda cls, path, revision=None: FakeModel()),
    })
    fake_transformers.AutoProcessor = type('AutoProcessor', (), {
        'from_pretrained': classmethod(lambda cls, path, revision=None: FakeProcessor()),
    })

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    return fake_transformers


def _install_fake_huggingface_hub(monkeypatch, tmp_path):
    fake_hf_hub = MagicMock()
    snap_dir = tmp_path / "snapshots" / "abc123def"
    snap_dir.mkdir(parents=True, exist_ok=True)
    (snap_dir / "model.safetensors").write_bytes(b"test-model-bytes-for-siglip2")
    (snap_dir / "config.json").write_bytes(b'{"hidden_size": 128}')
    fake_hf_hub.snapshot_download = lambda model_id, revision=None, **kwargs: str(snap_dir)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_hub)
    return fake_hf_hub


def _install_fake_pil(monkeypatch):
    fake_pil = MagicMock()
    fake_image = MagicMock()
    fake_image.convert.return_value = "rgb"
    fake_pil.Image.open = MagicMock(return_value=fake_image)
    fake_pil.Image.__name__ = "Image"
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_pil.Image)
    return fake_pil


def _install_all_fakes(monkeypatch, tmp_path):
    _install_fake_torch(monkeypatch)
    _install_fake_transformers(monkeypatch, tmp_path)
    _install_fake_huggingface_hub(monkeypatch, tmp_path)
    _install_fake_pil(monkeypatch)


def test_image_rows_l2_normalised(tmp_path, monkeypatch):
    _install_all_fakes(monkeypatch, tmp_path)
    from benchmarks.e048 import siglip2

    emb = siglip2.Siglip2ImageEmbedder()
    emb._device = "cpu"

    q = emb.embed_queries(["a", "b"])
    assert q.shape == (2, 2)
    assert q.dtype == np.float32
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-5)

    items = emb.embed_items(["/tmp/fake.jpg", "/tmp/fake2.jpg"])
    assert np.allclose(np.linalg.norm(items, axis=1), 1.0, atol=1e-5)


def test_embed_items_records_images(tmp_path, monkeypatch):
    _install_all_fakes(monkeypatch, tmp_path)
    from benchmarks.e048 import siglip2

    fake_pil = sys.modules["PIL"]

    emb = siglip2.Siglip2ImageEmbedder()
    emb._device = "cpu"
    emb.embed_items(["/tmp/fake.jpg", "/tmp/fake2.jpg"])

    # Check that PIL.Image.open was called twice with the correct paths
    assert fake_pil.Image.open.call_count == 2
    fake_pil.Image.open.assert_any_call("/tmp/fake.jpg")
    fake_pil.Image.open.assert_any_call("/tmp/fake2.jpg")


def test_provenance(tmp_path, monkeypatch):
    _install_all_fakes(monkeypatch, tmp_path)
    from benchmarks.e048 import siglip2

    emb = siglip2.Siglip2ImageEmbedder()
    emb._device = "cpu"

    prov = emb.provenance()
    assert prov["model_id"] == "google/siglip2-base-patch16-224"
    assert prov["revision"] == "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
    expected_hash = hashlib.sha256(b"test-model-bytes-for-siglip2").hexdigest()
    assert prov["model_files"]["model.safetensors"] == expected_hash
    assert prov["runtime"] == "transformers"
    assert prov["execution_provider"] == "cpu"
    assert prov["dtype"] == "float32"


def test_provenance_model_files_sorted(tmp_path, monkeypatch):
    _install_all_fakes(monkeypatch, tmp_path)
    from benchmarks.e048 import siglip2

    # Create two weight files
    snap_dir = Path(sys.modules["huggingface_hub"].snapshot_download(
        "google/siglip2-base-patch16-224", revision="75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
    ))
    (snap_dir / "model.safetensors").write_bytes(b"first")
    (snap_dir / "another.safetensors").write_bytes(b"second")

    emb = siglip2.Siglip2ImageEmbedder()
    emb._device = "cpu"
    prov = emb.provenance()

    # Check that we have two files and they are sorted by name
    assert set(prov["model_files"].keys()) == {"another.safetensors", "model.safetensors"}
    # The keys should be in sorted order (lexicographically)
    assert list(prov["model_files"].keys()) == ["another.safetensors", "model.safetensors"]


def test_pinned_revision_passed_to_loader(tmp_path, monkeypatch):
    _install_all_fakes(monkeypatch, tmp_path)
    from benchmarks.e048 import siglip2

    fake_transformers = sys.modules["transformers"]
    # Wrap the from_pretrained methods to capture arguments
    original_auto_model = fake_transformers.AutoModel.from_pretrained
    original_auto_processor = fake_transformers.AutoProcessor.from_pretrained

    captured_args = {}

    def capture_auto_model(*args, **kwargs):
        captured_args["model"] = (args, kwargs)
        return original_auto_model(*args, **kwargs)

    def capture_auto_processor(*args, **kwargs):
        captured_args["processor"] = (args, kwargs)
        return original_auto_processor(*args, **kwargs)

    fake_transformers.AutoModel.from_pretrained = capture_auto_model
    fake_transformers.AutoProcessor.from_pretrained = capture_auto_processor

    emb = siglip2.Siglip2ImageEmbedder()
    emb._device = "cpu"
    # Trigger lazy init
    emb.embed_queries(["test"])

    # Check that revision was passed
    assert "model" in captured_args
    assert captured_args["model"][1]["revision"] == "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
    assert "processor" in captured_args
    assert captured_args["processor"][1]["revision"] == "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"


def test_constructing_embedder_imports_nothing_heavy(tmp_path, monkeypatch):
    _install_all_fakes(monkeypatch, tmp_path)
    # Now remove the heavy modules so we can detect if they are imported
    import sys
    for mod in ["torch", "transformers", "huggingface_hub", "PIL"]:
        if mod in sys.modules:
            del sys.modules[mod]

    from benchmarks.e048 import siglip2

    emb = siglip2.Siglip2ImageEmbedder()
    # After __init__, the heavy modules should not have been imported
    for mod in ["torch", "transformers", "huggingface_hub", "PIL"]:
        assert mod not in sys.modules, f"{mod} was imported during __init__"

    # Now reinstall the fakes for these modules so that _lazy_init works
    _install_fake_torch(monkeypatch)
    _install_fake_transformers(monkeypatch, tmp_path)
    _install_fake_huggingface_hub(monkeypatch, tmp_path)
    _install_fake_pil(monkeypatch)
    # Now trigger _lazy_init
    emb._lazy_init()
    # After _lazy_init, they should be imported (as fakes)
    for mod in ["torch", "transformers", "huggingface_hub", "PIL"]:
        assert mod in sys.modules, f"{mod} was not imported during _lazy_init"


def test_text_processor_call_receives_padding_and_max_length(tmp_path, monkeypatch):
    _install_all_fakes(monkeypatch, tmp_path)
    from benchmarks.e048 import siglip2

    fake_transformers = sys.modules["transformers"]
    fake_processor = fake_transformers.AutoProcessor.from_pretrained(
        "dummy", revision="dummy"
    )

    emb = siglip2.Siglip2ImageEmbedder()
    emb._device = "cpu"
    # We need to access the processor instance that will be created in _lazy_init
    # Let's monkey-patch the AutoProcessor.from_pretrained to return our fake processor
    original_from_pretrained = fake_transformers.AutoProcessor.from_pretrained
    fake_transformers.AutoProcessor.from_pretrained = lambda *a, **k: fake_processor

    emb.embed_queries(["hello", "world"])

    # Check that the processor was called with padding="max_length" and max_length=64
    assert len(fake_processor._call_log) > 0
    last_call = fake_processor._call_log[-1]
    assert last_call.get("padding") == "max_length"
    assert last_call.get("max_length") == 64
    assert last_call.get("return_tensors") == "pt"

    # Restore
    fake_transformers.AutoProcessor.from_pretrained = original_from_pretrained


def test_grep_guard():
    siglip2_path = Path(__file__).parent.parent / "benchmarks" / "e048" / "siglip2.py"
    text = siglip2_path.read_text()
    assert "dummy_hash" not in text
    assert "hexdigest()[:12]" not in text