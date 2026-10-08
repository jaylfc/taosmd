"""Tests for the ONNX embedder's generic feed of extra graph inputs.

These cover tsk-vve2xl: models such as EmbeddingGemma-2 export ONNX graphs
with extra media inputs (image_features, video_features, audio_features)
alongside input_ids/attention_mask. The embedder must empty-fill every
declared input it can drive (leading symbolic dim, concrete trailing dims)
so the graph runs instead of raising on a missing feed key, and it must
fall back to an empty vector only when an input genuinely cannot be driven.

The tests are hermetic: no real ONNX model is loaded and the onnxruntime /
transformers packages are never imported. A fake tokenizer and a fake
session (with simple ``.name`` / ``.shape`` / ``.type`` input descriptors)
stand in for the real objects.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from taosmd.vector_memory import VectorMemory


class FakeVarInput:
    """Minimal stand-in for an onnxruntime graph Input meta-object."""

    def __init__(self, name, shape, type):
        self.name = name
        self.shape = list(shape)
        self.type = type


class FakeVarOutput:
    def __init__(self, name):
        self.name = name


class FakeTokenizer:
    """Callable tokenizer returning fixed int64 input ids / attention mask."""

    def __call__(self, text, return_tensors="np", padding=True, truncation=True):
        return {
            "input_ids": np.array([[1, 2, 3]], dtype=np.int64),
            "attention_mask": np.array([[1, 1, 1]], dtype=np.int64),
        }


def _media_inputs():
    """input_ids, attention_mask, and a media input the model declares."""
    return [
        FakeVarInput("input_ids", [1, "seq"], "tensor(int64)"),
        FakeVarInput("attention_mask", [1, "seq"], "tensor(int64)"),
        FakeVarInput("image_features", ["n", 512], "tensor(float)"),
    ]


def _make_vm(session, tokenizer):
    vm = VectorMemory(embed_mode="onnx")
    vm._onnx_session = session
    vm._onnx_tokenizer = tokenizer
    return vm


def _clear_prefix_env(monkeypatch):
    """Drop env prefix overrides so _onnx_apply_prefix is deterministic."""
    for var in ("TAOSMD_ONNX_QUERY_PREFIX", "TAOSMD_ONNX_DOC_PREFIX",
                "TAOSMD_ONNX_POOLING"):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# (a) A media input the model requires is empty-filled so the graph runs.
# ---------------------------------------------------------------------------
class _RequiresImageFeatures:
    """Fake session demanding a correctly empty-filled image_features input."""

    def __init__(self, inputs):
        self._inputs = inputs

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return [FakeVarOutput("sentence_embedding")]

    def run(self, *args, **kwargs):
        feed = args[-1] if args else kwargs["feed"]
        assert "image_features" in feed, "image_features must be fed"
        assert feed["image_features"].shape == (0, 512)
        assert feed["image_features"].dtype == np.float32
        return [np.array([[3.0, 4.0]], dtype=np.float32)]


def test_embed_onnx_feeds_empty_media_input(monkeypatch):
    _clear_prefix_env(monkeypatch)
    vm = _make_vm(_RequiresImageFeatures(_media_inputs()), FakeTokenizer())
    vec = vm._embed_onnx("taosmd embedder probe", "search_document")
    assert vec == pytest.approx([0.6, 0.8])


# ---------------------------------------------------------------------------
# (b) MiniLM regression pin: no extra inputs -> nothing spurious is fed.
# ---------------------------------------------------------------------------
class _MiniLM:
    """Fake session with only the core inputs; forbids a media input."""

    def __init__(self):
        self._inputs = [
            FakeVarInput("input_ids", [1, "seq"], "tensor(int64)"),
            FakeVarInput("attention_mask", [1, "seq"], "tensor(int64)"),
        ]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return [FakeVarOutput("sentence_embedding")]

    def run(self, *args, **kwargs):
        feed = args[-1] if args else kwargs["feed"]
        assert "image_features" not in feed, "MiniLM must not see a media input"
        return [np.array([[3.0, 4.0]], dtype=np.float32)]


def test_embed_onnx_minilm_does_not_feed_spurious_inputs(monkeypatch):
    _clear_prefix_env(monkeypatch)
    vm = _make_vm(_MiniLM(), FakeTokenizer())
    vec = vm._embed_onnx("probe", "search_document")
    assert vec == pytest.approx([0.6, 0.8])


# ---------------------------------------------------------------------------
# (c) An input whose non-leading dim is symbolic cannot be empty-filled, so it
# is left unfed; the embed fails and returns [] with a WARNING.
# ---------------------------------------------------------------------------
class _RejectsSymbolicNonLeading:
    """Fake session that raises because an unfed symbolic input is required."""

    def __init__(self, inputs):
        self._inputs = inputs

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return [FakeVarOutput("sentence_embedding")]

    def run(self, *args, **kwargs):
        feed = args[-1] if args else kwargs["feed"]
        assert "image_features" in feed, "image_features should be filled"
        assert "bad" not in feed, "symbolic non-leading-dim input must not be filled"
        raise ValueError("model requires unfed input 'bad'")


def test_embed_onnx_symbolic_nonleading_dim_is_not_fed(monkeypatch, caplog):
    _clear_prefix_env(monkeypatch)
    inputs = _media_inputs() + [
        FakeVarInput("bad", ["n", "m"], "tensor(float)"),
    ]
    vm = _make_vm(_RejectsSymbolicNonLeading(inputs), FakeTokenizer())
    with caplog.at_level(logging.DEBUG, logger="taosmd.vector_memory"):
        vec = vm._embed_onnx("probe", "search_document")
    assert vec == []
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "expected a WARNING, not a silent DEBUG, on embed failure"


# ---------------------------------------------------------------------------
# (d) Load-time probe: a graph we cannot drive fails loudly to QMD fallback.
# ---------------------------------------------------------------------------
class _AlwaysRaisesSession:
    """Fake session whose run always raises, declaring an unfed 'bad' input."""

    def __init__(self, inputs):
        self._inputs = inputs

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return [FakeVarOutput("sentence_embedding")]

    def run(self, *args, **kwargs):
        raise ValueError("probe: model run always fails")


def test_load_time_probe_falls_back_to_qmd(tmp_path, monkeypatch, caplog):
    _clear_prefix_env(monkeypatch)

    # Fake onnxruntime + transformers so init's onnx branch runs without the
    # real packages (onnxruntime may be absent in CI).
    inputs = _media_inputs() + [
        FakeVarInput("bad", ["n", "m"], "tensor(float)"),
    ]
    session = _AlwaysRaisesSession(inputs)

    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.InferenceSession = lambda model_file, **kwargs: session

    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *a, **k: FakeTokenizer()
    )

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)

    db = tmp_path / "vec.db"
    vm = VectorMemory(
        db_path=str(db),
        embed_mode="onnx",
        onnx_path="models/fake-probe-model",
    )
    with caplog.at_level(logging.DEBUG, logger="taosmd.vector_memory"):
        asyncio.run(vm.init())

    # A graph we cannot drive must fail loud (WARNING) and fall back to qmd,
    # surfacing the unfed input name rather than silently staying onnx.
    assert vm._embed_mode == "qmd"
    assert vm._onnx_session is session  # still set, but mode fell back
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("bad" in r.getMessage() for r in warnings), (
        f"expected a WARNING naming the unfed input 'bad'; got: "
        f"{[r.getMessage() for r in warnings]}"
    )
