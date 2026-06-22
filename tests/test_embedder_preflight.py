# tests/test_embedder_preflight.py
"""Setup-time preflight for the dense ONNX embedder.

The recipe writes ``embed_model`` (e.g. arctic-embed-s) into config, but the
model files are fetched separately by scripts/setup.sh. If a provisioning path
skips that script the embedder is absent, and at serve time the store silently
degrades to a different embedder, which corrupts the vector space. The preflight
catches that gap loudly at setup, mirroring the enricher-model preflight.
"""

from taosmd import auto_setup


def _make_onnx(root, name, layout):
    """Create a fake model dir; layout is 'root' or 'onnx_subdir'."""
    d = root / "models" / name
    if layout == "root":
        d.mkdir(parents=True)
        (d / "model.onnx").write_bytes(b"\x00")
    elif layout == "onnx_subdir":
        (d / "onnx").mkdir(parents=True)
        (d / "onnx" / "model.onnx").write_bytes(b"\x00")
    return d


def test_present_at_root_returns_true(tmp_path):
    _make_onnx(tmp_path, "minilm-onnx", "root")
    ok = auto_setup._preflight_embedder_model(
        "minilm-onnx", models_root=str(tmp_path / "models"), interactive=False
    )
    assert ok is True


def test_present_under_onnx_subdir_returns_true(tmp_path):
    # arctic-embed-s ships model.onnx under an onnx/ subdir
    _make_onnx(tmp_path, "arctic-embed-s", "onnx_subdir")
    ok = auto_setup._preflight_embedder_model(
        "arctic-embed-s", models_root=str(tmp_path / "models"), interactive=False
    )
    assert ok is True


def test_missing_returns_false_without_raising(tmp_path):
    ok = auto_setup._preflight_embedder_model(
        "arctic-embed-s", models_root=str(tmp_path / "models"), interactive=False
    )
    assert ok is False


def test_missing_prints_download_command(tmp_path, capsys):
    auto_setup._preflight_embedder_model(
        "arctic-embed-s", models_root=str(tmp_path / "models"), interactive=False
    )
    out = capsys.readouterr().out
    # names the missing model and the exact fetch command for it
    assert "arctic-embed-s" in out
    assert "snowflake-arctic-embed-s" in out
    # warns that a silent embedder swap is the failure mode
    assert "vector" in out.lower()


def test_empty_embed_model_does_not_crash(tmp_path):
    # defensive: a blank embed_model should be treated as nothing to check
    ok = auto_setup._preflight_embedder_model(
        "", models_root=str(tmp_path / "models"), interactive=False
    )
    assert ok is False
