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

    def detach(self):
        return self

    def __getattr__(self, name):
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


class FakeModelOutput:
    """Fake BaseModelOutputWithPooling-shaped object with pooler_output."""

    def __init__(self, pooler_output, last_hidden_state=None):
        self.pooler_output = pooler_output
        self.last_hidden_state = last_hidden_state


def _install_fake_torch(monkeypatch):
    fake_torch = MagicMock()
    fake_torch.__version__ = "2.0.0"
    fake_torch.cuda.is_available.return_value = False
    fake_torch.tensor = lambda *a, **k: _make_fake_tensor(np.array(a, dtype=np.float32))
    fake_torch.no_grad = lambda: MagicMock(
        __enter__=lambda s: None, __exit__=lambda s, *a: None
    )
    fake_torch.float32 = np.float32
    fake_torch.long = np.int64
    fake_torch.stack = lambda arrays, dim=0: _make_fake_tensor(
        np.stack(
            [a._arr if isinstance(a, FakeTensor) else a for a in arrays], axis=dim
        )
    )
    fake_torch.from_numpy = lambda x: _make_fake_tensor(x)
    fake_torch.nn = MagicMock()
    fake_torch.nn.Module = object
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.nn", fake_torch.nn)
    return fake_torch


def _install_fake_pil(monkeypatch):
    fake_pil = MagicMock()
    fake_image = MagicMock()
    fake_image.convert.return_value = "rgb"
    opened_paths = []

    def fake_open(p):
        opened_paths.append(p)
        return fake_image

    fake_pil.Image.open = fake_open
    fake_pil.Image.__name__ = "Image"
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_pil.Image)
    fake_pil._opened_paths = opened_paths
    return fake_pil


def _install_fake_transformers(monkeypatch, tmp_path):
    snap_dir = tmp_path / "snapshots" / REVISION
    snap_dir.mkdir(parents=True, exist_ok=True)
    # Create two fake weight files with known content for exact digest assertion
    weight_file1 = snap_dir / "model-00001-of-00002.safetensors"
    weight_file2 = snap_dir / "model-00002-of-00002.safetensors"
    weight_bytes1 = b"siglip2-weight-file-1-bytes"
    weight_bytes2 = b"siglip2-weight-file-2-bytes-different"
    weight_file1.write_bytes(weight_bytes1)
    weight_file2.write_bytes(weight_bytes2)

    recorded_model_kwargs = {}
    recorded_processor_kwargs = {}

    class FakeSiglipModel:
        config = MagicMock()
        config.hidden_size = 768

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            recorded_model_kwargs["model_id"] = model_id
            recorded_model_kwargs.update(kwargs)
            return cls()

        def eval(self):
            return None

        def to(self, device):
            return self

        def get_text_features(self, **kwargs):
            recorded_processor_kwargs["text_call"] = kwargs
            # Get batch size from input_ids if available
            batch_size = 1
            if "input_ids" in kwargs:
                input_ids = kwargs["input_ids"]
                if hasattr(input_ids, "shape"):
                    batch_size = input_ids.shape[0]
                elif isinstance(input_ids, (list, tuple)):
                    batch_size = len(input_ids)
            # Return BaseModelOutputWithPooling-shaped object with correct embedding dim
            return FakeModelOutput(
                pooler_output=_make_fake_tensor(
                    np.ones((batch_size, 768), dtype=np.float32)
                ),
                last_hidden_state=_make_fake_tensor(
                    np.ones((batch_size, 10, 768), dtype=np.float32)
                ),
            )

        def get_image_features(self, **kwargs):
            recorded_processor_kwargs["image_call"] = kwargs
            # Get batch size from pixel_values if available
            batch_size = 1
            if "pixel_values" in kwargs:
                pixel_values = kwargs["pixel_values"]
                if hasattr(pixel_values, "shape"):
                    batch_size = pixel_values.shape[0]
            return FakeModelOutput(
                pooler_output=_make_fake_tensor(
                    np.ones((batch_size, 768), dtype=np.float32)
                ),
                last_hidden_state=_make_fake_tensor(
                    np.ones((batch_size, 10, 768), dtype=np.float32)
                ),
            )

    class FakeSiglipProcessor:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            recorded_processor_kwargs["processor_model_id"] = model_id
            recorded_processor_kwargs["processor_kwargs"] = kwargs
            return cls()

        def __call__(self, **kwargs):
            recorded_processor_kwargs["processor_call"] = kwargs
            # Determine batch size
            batch_size = 1
            if "text" in kwargs:
                texts = kwargs["text"]
                batch_size = len(texts) if isinstance(texts, list) else 1
                return {
                    "input_ids": _make_fake_tensor(
                        np.ones((batch_size, 64), dtype=np.int64)
                    ),
                    "attention_mask": _make_fake_tensor(
                        np.ones((batch_size, 64), dtype=np.int64)
                    ),
                }
            else:
                images = kwargs.get("images", [])
                batch_size = len(images) if isinstance(images, list) else 1
                return {
                    "pixel_values": _make_fake_tensor(
                        np.ones((batch_size, 3, 224, 224), dtype=np.float32)
                    )
                }

    # Build fake transformers module
    class FakeTransformers:
        pass

    fake_transformers = FakeTransformers()
    fake_transformers.__version__ = "4.57.0"
    fake_transformers.AutoModel = type(
        "AutoModel",
        (),
        {
            "from_pretrained": classmethod(
                lambda cls, model_id, **kwargs: FakeSiglipModel.from_pretrained(
                    model_id, **kwargs
                )
            )
        },
    )
    fake_transformers.AutoProcessor = type(
        "AutoProcessor",
        (),
        {
            "from_pretrained": classmethod(
                lambda cls, model_id, **kwargs: FakeSiglipProcessor.from_pretrained(
                    model_id, **kwargs
                )
            )
        },
    )
    fake_transformers._fake_model_instance = FakeSiglipModel()
    fake_transformers._fake_processor_instance = FakeSiglipProcessor()
    fake_transformers._recorded_model_kwargs = recorded_model_kwargs
    fake_transformers._recorded_processor_kwargs = recorded_processor_kwargs

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(
        sys.modules, "transformers.models.siglip2", MagicMock()
    )
    return fake_transformers


def _install_fake_huggingface_hub(monkeypatch, tmp_path):
    fake_hf_hub = MagicMock()
    snap_dir = tmp_path / "snapshots" / REVISION
    snap_dir.mkdir(parents=True, exist_ok=True)
    weight_file1 = snap_dir / "model-00001-of-00002.safetensors"
    weight_file2 = snap_dir / "model-00002-of-00002.safetensors"
    weight_bytes1 = b"siglip2-weight-file-1-bytes"
    weight_bytes2 = b"siglip2-weight-file-2-bytes-different"
    weight_file1.write_bytes(weight_bytes1)
    weight_file2.write_bytes(weight_bytes2)

    recorded_snapshot_kwargs = {}

    def fake_snapshot_download(model_id, **kwargs):
        recorded_snapshot_kwargs["model_id"] = model_id
        recorded_snapshot_kwargs.update(kwargs)
        return str(snap_dir)

    fake_hf_hub.snapshot_download = fake_snapshot_download
    fake_hf_hub._recorded_kwargs = recorded_snapshot_kwargs
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_hub)
    return fake_hf_hub


def _install_all_fakes(monkeypatch, tmp_path):
    _install_fake_torch(monkeypatch)
    _install_fake_pil(monkeypatch)
    _install_fake_transformers(monkeypatch, tmp_path)
    _install_fake_huggingface_hub(monkeypatch, tmp_path)


# Import constants from the module
from benchmarks.e048.siglip2 import REVISION
from benchmarks.e048.retrieval import REQUIRED_PROVENANCE


def test_embed_queries_rows_l2_normalised(tmp_path, monkeypatch):
    _install_all_fakes(monkeypatch, tmp_path)
    from benchmarks.e048 import siglip2

    emb = siglip2.Siglip2ImageEmbedder()
    # Goes through _lazy_init, never set private attributes directly

    q = emb.embed_queries(["a", "b", "c"])
    assert q.shape == (3, 768)
    assert q.dtype == np.float32
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-5)


def test_embed_items_rows_l2_normalised_and_pil_opened(tmp_path, monkeypatch):
    _install_all_fakes(monkeypatch, tmp_path)
    from benchmarks.e048 import siglip2

    emb = siglip2.Siglip2ImageEmbedder()
    paths = ["/tmp/img1.jpg", "/tmp/img2.png", "/tmp/img3.jpeg"]

    items = emb.embed_items(paths)
    assert items.shape == (3, 768)
    assert items.dtype == np.float32
    assert np.allclose(np.linalg.norm(items, axis=1), 1.0, atol=1e-5)

    # Check PIL was called with the exact paths in order
    fake_pil = sys.modules["PIL"]
    assert fake_pil._opened_paths == paths


def test_provenance_keys_and_revision_and_exact_digests(tmp_path, monkeypatch):
    _install_all_fakes(monkeypatch, tmp_path)
    from benchmarks.e048 import siglip2

    emb = siglip2.Siglip2ImageEmbedder()
    prov = emb.provenance()

    # Check all required provenance keys present
    for key in REQUIRED_PROVENANCE:
        assert key in prov, f"Missing provenance key: {key}"

    # Revision equals pinned commit string exactly
    assert prov["revision"] == REVISION

    # Model files maps each fake weight filename to exact sha256 of its bytes
    expected_hash1 = hashlib.sha256(b"siglip2-weight-file-1-bytes").hexdigest()
    expected_hash2 = hashlib.sha256(b"siglip2-weight-file-2-bytes-different").hexdigest()
    assert "model-00001-of-00002.safetensors" in prov["model_files"]
    assert "model-00002-of-00002.safetensors" in prov["model_files"]
    assert prov["model_files"]["model-00001-of-00002.safetensors"] == expected_hash1
    assert prov["model_files"]["model-00002-of-00002.safetensors"] == expected_hash2

    # Sorted order of keys
    model_files_keys = list(prov["model_files"].keys())
    assert model_files_keys == sorted(model_files_keys)


def test_provenance_revision_passed_to_loader(tmp_path, monkeypatch):
    _install_all_fakes(monkeypatch, tmp_path)
    from benchmarks.e048 import siglip2

    emb = siglip2.Siglip2ImageEmbedder()
    emb.provenance()  # Triggers _lazy_init

    fake_transformers = sys.modules["transformers"]
    fake_hf_hub = sys.modules["huggingface_hub"]

    # Check revision was passed to model loader
    assert fake_transformers._recorded_model_kwargs.get("revision") == REVISION
    # Check revision was passed to processor loader (stored in processor_kwargs)
    assert (
        fake_transformers._recorded_processor_kwargs.get("processor_kwargs", {}).get(
            "revision"
        )
        == REVISION
    )
    # Check revision was passed to snapshot_download
    assert fake_hf_hub._recorded_kwargs.get("revision") == REVISION


def test_constructor_imports_nothing_heavy(tmp_path, monkeypatch):
    """Constructing the embedder imports nothing heavy."""
    _install_all_fakes(monkeypatch, tmp_path)
    # Capture sys.modules state before import
    heavy_modules = {"torch", "transformers", "huggingface_hub", "PIL"}
    before = set(sys.modules.keys()) & heavy_modules

    from benchmarks.e048 import siglip2

    siglip2.Siglip2ImageEmbedder()
    # After construction, before any method call
    after = set(sys.modules.keys()) & heavy_modules

    # No new heavy modules should have been imported
    assert after == before


def test_text_processor_call_padding_max_length_64(tmp_path, monkeypatch):
    _install_all_fakes(monkeypatch, tmp_path)
    from benchmarks.e048 import siglip2

    emb = siglip2.Siglip2ImageEmbedder()
    emb.embed_queries(["hello world"])

    fake_transformers = sys.modules["transformers"]
    recorded = fake_transformers._recorded_processor_kwargs.get("processor_call", {})
    assert recorded.get("padding") == "max_length"
    assert recorded.get("max_length") == 64


def test_empty_embed_queries_returns_correct_shape(tmp_path, monkeypatch):
    _install_all_fakes(monkeypatch, tmp_path)
    from benchmarks.e048 import siglip2

    emb = siglip2.Siglip2ImageEmbedder()
    q = emb.embed_queries([])
    assert q.shape == (0, 768)
    assert q.dtype == np.float32


def test_empty_embed_items_returns_correct_shape(tmp_path, monkeypatch):
    _install_all_fakes(monkeypatch, tmp_path)
    from benchmarks.e048 import siglip2

    emb = siglip2.Siglip2ImageEmbedder()
    items = emb.embed_items([])
    assert items.shape == (0, 768)
    assert items.dtype == np.float32


def test_batch_processing_multiple_batches(tmp_path, monkeypatch):
    _install_all_fakes(monkeypatch, tmp_path)
    from benchmarks.e048 import siglip2

    emb = siglip2.Siglip2ImageEmbedder()
    # Create 35 items to test batching (32 + 3)
    paths = [f"/tmp/img{i}.jpg" for i in range(35)]
    items = emb.embed_items(paths)
    assert items.shape == (35, 768)
    assert items.dtype == np.float32
    assert np.allclose(np.linalg.norm(items, axis=1), 1.0, atol=1e-5)


def test_import_blocker_meta_path(tmp_path, monkeypatch):
    """Test that real torch, transformers, PIL, huggingface_hub, sentence_transformers
    are never imported by installing a sys.meta_path finder that raises ImportError."""
    # First install our fakes
    _install_all_fakes(monkeypatch, tmp_path)

    # Now install a meta_path finder that blocks the real modules
    blocked_modules = {
        "torch",
        "torch.nn",
        "transformers",
        "transformers.models.siglip2",
        "huggingface_hub",
        "PIL",
        "PIL.Image",
        "sentence_transformers",
    }

    class BlockingFinder:
        def find_spec(self, fullname, path, target=None):
            if fullname in blocked_modules or any(
                fullname.startswith(b + ".") for b in blocked_modules
            ):
                raise ImportError(f"Blocked import of {fullname}")
            return None

    blocker = BlockingFinder()
    sys.meta_path.insert(0, blocker)
    try:
        from benchmarks.e048 import siglip2

        emb = siglip2.Siglip2ImageEmbedder()

        # These should all work through _lazy_init using our fakes
        q = emb.embed_queries(["test query"])
        assert q.shape == (1, 768)
        assert q.dtype == np.float32
        assert np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-5)

        items = emb.embed_items(["/tmp/test.jpg"])
        assert items.shape == (1, 768)
        assert items.dtype == np.float32
        assert np.allclose(np.linalg.norm(items, axis=1), 1.0, atol=1e-5)

        prov = emb.provenance()
        assert prov["revision"] == REVISION
        assert set(REQUIRED_PROVENANCE).issubset(set(prov.keys()))
    finally:
        sys.meta_path.remove(blocker)


def test_grep_guard():
    siglip2_path = Path(__file__).parent.parent / "benchmarks" / "e048" / "siglip2.py"
    text = siglip2_path.read_text()
    assert "dummy_hash" not in text
    assert "hexdigest()[:12]" not in text
