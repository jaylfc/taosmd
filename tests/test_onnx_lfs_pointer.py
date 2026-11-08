"""Tests for Git LFS pointer detection in ONNX model loading.

These tests are hermetic: no network, no model downloads, tmp_path only.
"""

from __future__ import annotations

import logging
import pytest

from taosmd.api import _has_onnx_model, _is_lfs_pointer


LFS_POINTER_CONTENT = b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 133093890\n"


def test_has_onnx_model_false_for_lfs_pointer_at_root(tmp_path):
    """A model.onnx that is an LFS pointer should return False."""
    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(LFS_POINTER_CONTENT)
    assert _has_onnx_model(tmp_path) is False


def test_has_onnx_model_false_for_lfs_pointer_in_onnx_subdir(tmp_path):
    """An onnx/model.onnx that is an LFS pointer should return False."""
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    model_file = onnx_dir / "model.onnx"
    model_file.write_bytes(LFS_POINTER_CONTENT)
    assert _has_onnx_model(tmp_path) is False


def test_has_onnx_model_true_for_real_model_at_root(tmp_path):
    """A model.onnx with real (non-pointer) content should return True."""
    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"\x00" * 2048)  # 2KB of arbitrary non-pointer bytes
    assert _has_onnx_model(tmp_path) is True


def test_has_onnx_model_true_for_real_model_in_onnx_subdir(tmp_path):
    """An onnx/model.onnx with real content should return True."""
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    model_file = onnx_dir / "model.onnx"
    model_file.write_bytes(b"\x00" * 2048)
    assert _has_onnx_model(tmp_path) is True


def test_is_lfs_pointer_detects_pointer(tmp_path):
    """_is_lfs_pointer returns True for a Git LFS pointer file."""
    ptr = tmp_path / "model.onnx"
    ptr.write_bytes(LFS_POINTER_CONTENT)
    assert _is_lfs_pointer(ptr) is True


def test_is_lfs_pointer_false_for_real_model(tmp_path):
    """_is_lfs_pointer returns False for a real model file."""
    model = tmp_path / "model.onnx"
    model.write_bytes(b"\x00" * 2048)
    assert _is_lfs_pointer(model) is False


def test_is_lfs_pointer_false_for_missing_file(tmp_path):
    """_is_lfs_pointer returns False for a non-existent file."""
    missing = tmp_path / "missing.onnx"
    assert _is_lfs_pointer(missing) is False


def test_is_lfs_pointer_false_for_large_file(tmp_path):
    """_is_lfs_pointer returns False for a file >= 1024 bytes even if it starts with 'version'."""
    large = tmp_path / "model.onnx"
    # 1024 bytes starting with "version" but not an LFS pointer
    large.write_bytes(b"version " + b"x" * 1016)
    assert _is_lfs_pointer(large) is False


# VectorMemory integration test
@pytest.mark.asyncio
async def test_vector_memory_onnx_lfs_pointer_logs_warning(tmp_path, caplog):
    """VectorMemory init with an LFS pointer logs a warning naming the pointer."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("transformers")

    # Create a fake ONNX model dir with an LFS pointer
    model_dir = tmp_path / "fake-model"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(LFS_POINTER_CONTENT)
    # Also need a tokenizer config for transformers to load
    (model_dir / "tokenizer_config.json").write_text('{"model_type": "bert"}')
    (model_dir / "config.json").write_text('{"hidden_size": 384}')

    from taosmd.vector_memory import VectorMemory

    db_path = tmp_path / "vec.db"
    vm = VectorMemory(
        db_path=str(db_path),
        embed_mode="onnx",
        onnx_path=str(model_dir),
    )

    with caplog.at_level(logging.WARNING, logger="taosmd.vector_memory"):
        await vm.init()

    # The warning should mention "Git LFS pointer"
    warning_messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("Git LFS pointer" in msg for msg in warning_messages), (
        f"Expected a WARNING containing 'Git LFS pointer'; got: {warning_messages}"
    )
    # Should have fallen back to qmd
    assert vm._embed_mode == "qmd"
