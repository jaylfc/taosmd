from __future__ import annotations

import hashlib
import importlib.abc
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

_PINNED_REVISION = "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
_PINNED_MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"

HEAVY_MODULES = (
    "torch",
    "transformers",
    "sentence_transformers",
    "PIL",
    "huggingface_hub",
)


class _BlockHeavyImports(importlib.abc.MetaPathFinder):
    """Meta path finder that blocks real imports of heavy modules."""

    _HEAVY = frozenset(HEAVY_MODULES)

    def find_spec(self, fullname, path=None, target=None):
        top = fullname.split(".")[0]
        if top in self._HEAVY:
            raise ImportError(f"blocked real import of {fullname}")
        return None


def _install_fakes(monkeypatch, tmp_path, cuda_available=False, snapshot_dir=None):
    """Install fake torch, sentence_transformers, and huggingface_hub modules.

    Mirrors the approach of tests/test_e048_controls.py: fake modules are
    injected via monkeypatch.setitem(sys.modules, ...) before any call that
    triggers _lazy_init, so the real packages are never imported.
    """
    fake_torch = MagicMock()
    fake_torch.__version__ = "2.0.0"
    fake_torch.cuda.is_available.return_value = cuda_available
    fake_torch.no_grad = lambda: MagicMock(
        __enter__=lambda s: None, __exit__=lambda s, *a: None
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    last_instance = [None]

    class FakeSentenceTransformer:
        def __init__(self, model_id, revision=None, device=None):
            self.init_kwargs = {
                "model_id": model_id,
                "revision": revision,
                "device": device,
            }
            self.encode_calls = []
            last_instance[0] = self

        def eval(self):
            return self

        def encode(self, inputs, normalize_embeddings=True):
            self.encode_calls.append(inputs)
            n = len(inputs)
            return np.array([[3.0, 4.0]] * n, dtype=np.float32)

    fake_st = MagicMock()
    fake_st.__version__ = "0.1.0"
    fake_st.SentenceTransformer = FakeSentenceTransformer
    fake_st._last_instance = last_instance
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

    fake_hf = MagicMock()
    snap = snapshot_dir if snapshot_dir is not None else tmp_path
    fake_hf.snapshot_download = lambda model_id, **kwargs: str(snap)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    return fake_torch, fake_st, fake_hf


def test_embed_queries_normalised(tmp_path, monkeypatch):
    _install_fakes(monkeypatch, tmp_path)
    from benchmarks.e048.qwen3vl import Qwen3VLImageEmbedder

    emb = Qwen3VLImageEmbedder()
    result = emb.embed_queries(["a", "b", "c"])

    assert result.dtype == np.float32
    assert result.shape == (3, 2)
    norms = np.linalg.norm(result, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_embed_items_normalised_and_image_dicts(tmp_path, monkeypatch):
    _, fake_st, _ = _install_fakes(monkeypatch, tmp_path)
    from benchmarks.e048.qwen3vl import Qwen3VLImageEmbedder

    paths = ["/fake/img1.jpg", "/fake/img2.jpg", "/fake/img3.jpg"]
    emb = Qwen3VLImageEmbedder()
    result = emb.embed_items(paths)

    assert result.dtype == np.float32
    assert result.shape == (3, 2)
    norms = np.linalg.norm(result, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

    fake_instance = fake_st._last_instance[0]
    assert len(fake_instance.encode_calls) == 1
    received = fake_instance.encode_calls[0]
    assert received == [{"image": p} for p in paths]


def test_provenance_required_keys_and_hashes(tmp_path, monkeypatch):
    _install_fakes(monkeypatch, tmp_path)
    from benchmarks.e048.qwen3vl import Qwen3VLImageEmbedder
    from benchmarks.e048.retrieval import REQUIRED_PROVENANCE

    file_b = b"weight-bytes-aaa"
    file_c = b"weight-bytes-zzz"
    (tmp_path / "model.safetensors").write_bytes(file_b)
    (tmp_path / "config.safetensors").write_bytes(file_c)

    emb = Qwen3VLImageEmbedder()
    prov = emb.provenance()

    for key in REQUIRED_PROVENANCE:
        assert key in prov, f"missing {key}"

    assert prov["model_id"] == _PINNED_MODEL_ID
    assert prov["revision"] == _PINNED_REVISION

    expected = {
        "config.safetensors": hashlib.sha256(file_c).hexdigest(),
        "model.safetensors": hashlib.sha256(file_b).hexdigest(),
    }
    assert prov["model_files"] == expected

    keys = list(prov["model_files"].keys())
    assert keys == sorted(keys)


def test_pinned_revision_passed_to_loader(tmp_path, monkeypatch):
    _, fake_st, _ = _install_fakes(monkeypatch, tmp_path)
    from benchmarks.e048.qwen3vl import Qwen3VLImageEmbedder
    import benchmarks.e048.qwen3vl as qwen3vl

    emb = Qwen3VLImageEmbedder()
    emb.embed_queries(["hello"])

    fake_instance = fake_st._last_instance[0]
    assert hasattr(qwen3vl, "REVISION")
    assert hasattr(qwen3vl, "MODEL_ID")
    assert fake_instance.init_kwargs["revision"] == qwen3vl.REVISION
    assert fake_instance.init_kwargs["revision"] == _PINNED_REVISION
    assert fake_instance.init_kwargs["model_id"] == _PINNED_MODEL_ID


def test_construct_imports_nothing_heavy(monkeypatch):
    for name in HEAVY_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)

    from benchmarks.e048.qwen3vl import Qwen3VLImageEmbedder

    Qwen3VLImageEmbedder(device="cpu")

    for name in HEAVY_MODULES:
        assert name not in sys.modules, f"{name} imported during construction"


def test_meta_path_blocks_real_imports(tmp_path, monkeypatch):
    _install_fakes(monkeypatch, tmp_path)
    from benchmarks.e048.qwen3vl import Qwen3VLImageEmbedder

    (tmp_path / "weights.safetensors").write_bytes(b"fake-weights")

    finder = _BlockHeavyImports()
    monkeypatch.setattr(sys, "meta_path", [finder] + sys.meta_path)

    emb = Qwen3VLImageEmbedder()

    q = emb.embed_queries(["a", "b", "c"])
    assert q.dtype == np.float32
    assert q.shape == (3, 2)

    paths = ["/fake/1.jpg", "/fake/2.jpg"]
    items = emb.embed_items(paths)
    assert items.dtype == np.float32
    assert items.shape == (2, 2)

    prov = emb.provenance()
    assert prov["model_id"] == _PINNED_MODEL_ID
    assert prov["revision"] == _PINNED_REVISION


@pytest.mark.parametrize(
    ("device", "cuda_available", "expected"),
    [
        ("auto", True, "cuda"),
        ("auto", False, "cpu"),
        ("cpu", True, "cpu"),
        ("cuda", False, "cuda"),
    ],
)
def test_device_precedence(monkeypatch, tmp_path, device, cuda_available, expected):
    _, fake_st, _ = _install_fakes(monkeypatch, tmp_path, cuda_available=cuda_available)
    from benchmarks.e048.qwen3vl import Qwen3VLImageEmbedder

    (tmp_path / "model.safetensors").write_bytes(b"model")
    emb = Qwen3VLImageEmbedder(device=device)
    emb.embed_queries(["hello"])

    fake_instance = fake_st._last_instance[0]
    assert fake_instance.init_kwargs["device"] == expected
    assert emb.provenance()["execution_provider"] == expected


def test_provenance_raises_when_no_safetensors(monkeypatch, tmp_path):
    _install_fakes(monkeypatch, tmp_path)
    from benchmarks.e048.qwen3vl import Qwen3VLImageEmbedder

    emb = Qwen3VLImageEmbedder()
    with pytest.raises(RuntimeError, match="safetensors"):
        emb.provenance()


def test_provenance_model_files_sorted_regardless_of_walk_order(monkeypatch, tmp_path):
    _install_fakes(monkeypatch, tmp_path)
    from benchmarks.e048.qwen3vl import Qwen3VLImageEmbedder
    import benchmarks.e048.qwen3vl as qwen3vl

    (tmp_path / "a.safetensors").write_bytes(b"a")
    (tmp_path / "z.safetensors").write_bytes(b"z")

    def fake_walk(path):
        return [(str(tmp_path), [], ["z.safetensors", "a.safetensors"])]

    monkeypatch.setattr(qwen3vl.os, "walk", fake_walk)

    emb = Qwen3VLImageEmbedder()
    prov = emb.provenance()
    assert list(prov["model_files"].keys()) == ["a.safetensors", "z.safetensors"]
