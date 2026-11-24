"""ONNX embedder load-time guards for sentence_transformers models whose export
drops the Dense (projection) head (tsk-lfuqlq).

A sentence-transformers model such as MongoDB/mdbr-leaf-ir stacks
Transformer -> Pooling -> Dense -> Normalize. Its official ONNX export omits
the Dense layer, so the graph exposes no ``sentence_embedding`` output and the
loader would silently mean-pool the 384-d backbone output instead of the
768-d projected embedding -- a different model, and no error. These tests pin
that the loader refuses such exports and that a Dense-inclusive export works.

Hermetic: fake model dirs and fake ONNX sessions, no downloads, no real
onnxruntime or transformers required.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from taosmd.vector_memory import (
    VectorMemory,
    _onnx_apply_prefix,
    _onnx_pooling_mode,
)


class FakeVarOutput:
    """Minimal stand-in for an onnxruntime graph output meta-object."""

    def __init__(self, name):
        self.name = name


class FakeTokenizer:
    """Callable tokenizer returning fixed int64 input ids / attention mask."""

    def __call__(self, text, return_tensors="np", padding=True, truncation=True):
        return {
            "input_ids": np.array([[1, 2, 3]], dtype=np.int64),
            "attention_mask": np.array([[1, 1, 1]], dtype=np.int64),
        }


def _clear_onnx_env(monkeypatch):
    """Drop env prefix/pooling overrides so logic is deterministic."""
    for var in ("TAOSMD_ONNX_QUERY_PREFIX", "TAOSMD_ONNX_DOC_PREFIX",
                "TAOSMD_ONNX_POOLING"):
        monkeypatch.delenv(var, raising=False)


def _make_model_dir(tmp_path, name="mdbr-leaf-ir", *, modules=None,
                    dense_out=768, sentence_embedding=True,
                    st_config=None):
    """Build a fake ONNX model dir mirroring a sentence-transformers export.

    ``modules`` is the modules.json content. When it lists a
    ``sentence_transformers.models.Dense`` module, a matching
    ``<path>/config.json`` with ``out_features`` is written under ``2_Dense/``.
    """
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(b"\x00" * 64)  # non-LFS placeholder
    (model_dir / "tokenizer_config.json").write_text('{"model_type": "bert"}')
    (model_dir / "config.json").write_text('{"hidden_size": 384}')
    if modules is not None:
        (model_dir / "modules.json").write_text(json.dumps(modules))
    if dense_out is not None:
        dense_dir = model_dir / "2_Dense"
        dense_dir.mkdir()
        (dense_dir / "config.json").write_text(
            json.dumps({"in_features": 384, "out_features": dense_out, "bias": True,
                        "activation_function": "torch.nn.modules.linear.Identity"})
        )
    if st_config is not None:
        (model_dir / "config_sentence_transformers.json").write_text(
            json.dumps(st_config)
        )
    return model_dir


def _install_fake_onnx(monkeypatch, session):
    """Inject fake onnxruntime + transformers so VectorMemory.init loads fakes."""
    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.InferenceSession = lambda model_file, **kwargs: session
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *a, **k: FakeTokenizer()
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)


# ---------------------------------------------------------------------------
# Fake sessions
# ---------------------------------------------------------------------------
class _DenseMissingSession:
    """Graph with NO sentence_embedding output: the Dense head was stripped."""

    def __init__(self, output_names, dim=384):
        self._outputs = [FakeVarOutput(n) for n in output_names]
        self.dim = dim

    def get_inputs(self):
        return []

    def get_outputs(self):
        return self._outputs

    def run(self, *args, **kwargs):
        # Mean-pooled 384-d backbone output: the wrong embedding.
        return [np.full((1, 3, self.dim), 0.5, dtype=np.float32)]


class _SentenceEmbeddingSession:
    """Graph exposing sentence_embedding: the Dense head is present."""

    def __init__(self, dim=768):
        self.dim = dim

    def get_inputs(self):
        return []

    def get_outputs(self):
        return [FakeVarOutput("sentence_embedding")]

    def run(self, *args, **kwargs):
        return [np.full((1, self.dim), 0.5, dtype=np.float32)]


# ---------------------------------------------------------------------------
# (a) modules.json lists Dense, graph has no sentence_embedding: init raises.
# ---------------------------------------------------------------------------
_DENSE_MODULES = [
    {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
    {"idx": 1, "name": "1_Pooling", "path": "1_Pooling",
     "type": "sentence_transformers.models.Pooling"},
    {"idx": 2, "name": "2_Dense", "path": "2_Dense",
     "type": "sentence_transformers.models.Dense"},
    {"idx": 3, "name": "3_Normalize", "path": "3_Normalize",
     "type": "sentence_transformers.models.Normalize"},
]


def test_dense_head_stripped_export_raises_at_init(tmp_path, monkeypatch):
    _clear_onnx_env(monkeypatch)
    model_dir = _make_model_dir(
        tmp_path, modules=_DENSE_MODULES, dense_out=768,
        sentence_embedding=False,
    )
    session = _DenseMissingSession(["last_hidden_state"], dim=384)
    _install_fake_onnx(monkeypatch, session)
    vm = VectorMemory(
        db_path=str(tmp_path / "vec.db"),
        embed_mode="onnx", onnx_path=str(model_dir),
    )
    with pytest.raises(RuntimeError, match="sentence_embedding|Dense|export"):
        asyncio.run(vm.init())


# ---------------------------------------------------------------------------
# (b) same dir but sentence_embedding present: works, returns that output.
# ---------------------------------------------------------------------------
def test_dense_inclusive_export_returns_sentence_embedding(tmp_path, monkeypatch):
    _clear_onnx_env(monkeypatch)
    model_dir = _make_model_dir(
        tmp_path, modules=_DENSE_MODULES, dense_out=768,
        sentence_embedding=True,
    )
    session = _SentenceEmbeddingSession(dim=768)
    _install_fake_onnx(monkeypatch, session)
    vm = VectorMemory(
        db_path=str(tmp_path / "vec.db"),
        embed_mode="onnx", onnx_path=str(model_dir),
    )
    asyncio.run(vm.init())
    assert vm._embed_mode == "onnx"
    assert vm._onnx_session is session
    vec = vm._embed_onnx("any text", "search_document")
    assert vec == pytest.approx([1.0 / math.sqrt(768)] * 768)
    asyncio.run(vm.close())


# ---------------------------------------------------------------------------
# (c) dimension mismatch against the declared output dim: raises.
# ---------------------------------------------------------------------------
def test_declared_output_dim_mismatch_raises(tmp_path, monkeypatch):
    _clear_onnx_env(monkeypatch)
    model_dir = _make_model_dir(
        tmp_path, modules=_DENSE_MODULES, dense_out=768,
        sentence_embedding=True,
    )
    # Graph exposes sentence_embedding but only 384-d: Dense says 768.
    session = _SentenceEmbeddingSession(dim=384)
    _install_fake_onnx(monkeypatch, session)
    vm = VectorMemory(
        db_path=str(tmp_path / "vec.db"),
        embed_mode="onnx", onnx_path=str(model_dir),
    )
    with pytest.raises(RuntimeError, match="768|384|output|dim"):
        asyncio.run(vm.init())


# ---------------------------------------------------------------------------
# (d) MiniLM-style dir with no Dense in modules.json: mean pooling unchanged.
# ---------------------------------------------------------------------------
def test_no_dense_mean_pooling_unchanged(tmp_path, monkeypatch):
    _clear_onnx_env(monkeypatch)
    # No modules.json -> no Dense head, nothing to refuse.
    model_dir = _make_model_dir(tmp_path, name="minilm-onnx", modules=None,
                                dense_out=None, sentence_embedding=False)
    session = _DenseMissingSession(["last_hidden_state"], dim=384)
    # Non-zero token output so mean pooling is verifiable.
    session.run = lambda *a, **k: [np.full((1, 3, 384), 1.0, dtype=np.float32)]
    _install_fake_onnx(monkeypatch, session)
    vm = VectorMemory(
        db_path=str(tmp_path / "vec.db"),
        embed_mode="onnx", onnx_path=str(model_dir),
    )
    asyncio.run(vm.init())
    assert vm._embed_mode == "onnx"
    vec = vm._embed_onnx("probe", "search_document")
    assert vec == pytest.approx([1.0 / math.sqrt(384)] * 384)
    asyncio.run(vm.close())


# ---------------------------------------------------------------------------
# (e) leaf prefixes are applied for query vs document.
# ---------------------------------------------------------------------------
def test_leaf_prefix_query_and_document():
    _LEAF_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    query = _onnx_apply_prefix("models/mdbr-leaf-ir", "what is ml", "search_query")
    doc = _onnx_apply_prefix("models/mdbr-leaf-ir", "ml is math", "search_document")
    assert query == _LEAF_QUERY_PREFIX + "what is ml"
    assert doc == "ml is math"


def test_leaf_pooling_mode_is_mean():
    assert _onnx_pooling_mode("models/mdbr-leaf-ir") == "mean"


def test_leaf_prefix_env_override_wins(monkeypatch):
    monkeypatch.setenv("TAOSMD_ONNX_QUERY_PREFIX", "q: ")
    monkeypatch.setenv("TAOSMD_ONNX_DOC_PREFIX", "d: ")
    assert _onnx_apply_prefix("models/mdbr-leaf-ir", "x", "search_query") == "q: x"
    assert _onnx_apply_prefix("models/mdbr-leaf-ir", "x", "search_document") == "d: x"
