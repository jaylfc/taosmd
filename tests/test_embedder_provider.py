"""Tests for ONNX embedder provider configuration (tsk-awwytg).

Validates that the embedder ONNX provider is configurable via the
config models section, with GPU memory limit support and graceful
fallback when the configured provider is unavailable.
"""

import asyncio
import json
import logging
import sys
from unittest.mock import MagicMock, patch

# onnxruntime is an optional extra; import lazily in mock helper
ort = None
try:
    import onnxruntime as ort  # type: ignore  # noqa: F401
except ImportError:
    pass


def _run(coro):
    return asyncio.run(coro)


class _ORTMock:
    """Mock for onnxruntime that captures InferenceSession calls."""
    def __init__(self):
        self._available_providers = ["CPUExecutionProvider", "CUDAExecutionProvider"]
        self._session_instance = MagicMock()
        self.last_providers = None
        self.InferenceSession = MagicMock(side_effect=self._create_session)
        self.get_available_providers = MagicMock(side_effect=lambda: self._available_providers)
    
    def _create_session(self, model_path, providers=None, **kwargs):
        self.last_providers = providers
        return self._session_instance
    
    def set_available_providers(self, providers):
        self._available_providers = list(providers)


def _patch_onnxruntime(mock_ort):
    """Patch onnxruntime module in sys.modules and taosmd.vector_memory."""
    import taosmd.vector_memory as vm
    original_ort = sys.modules.get("onnxruntime")
    sys.modules["onnxruntime"] = mock_ort
    vm.ort = mock_ort
    return original_ort


def _unpatch_onnxruntime(original_ort):
    import taosmd.vector_memory as vm
    if original_ort is not None:
        sys.modules["onnxruntime"] = original_ort
    else:
        sys.modules.pop("onnxruntime", None)
    if hasattr(vm, "ort"):
        delattr(vm, "ort")


def _patch_tokenizer():
    """Patch transformers.AutoTokenizer in sys.modules."""
    import transformers
    original = getattr(transformers, "AutoTokenizer", None)
    mock_tokenizer = MagicMock()
    mock_tokenizer.from_pretrained.return_value = MagicMock()
    transformers.AutoTokenizer = mock_tokenizer
    return original, mock_tokenizer


def _unpatch_tokenizer(original):
    import transformers
    if original is not None:
        transformers.AutoTokenizer = original
    else:
        delattr(transformers, "AutoTokenizer")


def test_default_is_cpu_without_config(tmp_path):
    """No config -> providers == ['CPUExecutionProvider'] exactly as today."""
    from taosmd.vector_memory import VectorMemory

    data_dir = tmp_path / "taosmd"
    data_dir.mkdir()

    mock_ort = _ORTMock()
    mock_ort.set_available_providers(["CPUExecutionProvider"])
    original_ort = _patch_onnxruntime(mock_ort)
    original_tokenizer, _ = _patch_tokenizer()
    
    try:
        vm = VectorMemory(
            db_path=str(data_dir / "vm.db"),
            embed_mode="onnx",
            onnx_path="models/minilm-onnx",
        )
        _run(vm.init())

        # The ONNX session should be created with CPUExecutionProvider only
        assert vm._onnx_session is not None
        assert mock_ort.last_providers == ["CPUExecutionProvider"]
        _run(vm.close())
    finally:
        _unpatch_onnxruntime(original_ort)
        _unpatch_tokenizer(original_tokenizer)


def test_configured_cuda_used_when_available(tmp_path, caplog):
    """Config embedder_provider='CUDAExecutionProvider', embedder_gpu_mem_mb=2048,
    available providers include CUDA -> the session gets
    [('CUDAExecutionProvider', {'gpu_mem_limit': 2147483648}), 'CPUExecutionProvider'].
    """
    from taosmd.vector_memory import VectorMemory
    from taosmd import config as cfg

    data_dir = tmp_path / "taosmd"
    data_dir.mkdir()

    # Set config roles
    cfg.set_model_role("embedder_provider", "CUDAExecutionProvider", data_dir=str(data_dir))
    cfg.set_model_role("embedder_gpu_mem_mb", "2048", data_dir=str(data_dir))

    # Mock onnxruntime to capture the providers argument and simulate CUDA available
    mock_ort = _ORTMock()
    mock_ort.set_available_providers(["CPUExecutionProvider", "CUDAExecutionProvider"])
    original_ort = _patch_onnxruntime(mock_ort)
    original_tokenizer, _ = _patch_tokenizer()
    
    try:
        vm = VectorMemory(
            db_path=str(data_dir / "vm.db"),
            embed_mode="onnx",
            onnx_path="models/minilm-onnx",
            data_dir=str(data_dir),
        )
        _run(vm.init())

        # Check that InferenceSession was called with the expected providers
        providers_arg = mock_ort.last_providers
        assert providers_arg is not None, "InferenceSession not called"
        expected = [
            ("CUDAExecutionProvider", {"gpu_mem_limit": 2147483648}),
            "CPUExecutionProvider",
        ]
        assert providers_arg == expected, f"Expected {expected}, got {providers_arg}"

        _run(vm.close())
    finally:
        _unpatch_onnxruntime(original_ort)
        _unpatch_tokenizer(original_tokenizer)


def test_unavailable_provider_falls_back_to_cpu(tmp_path, caplog):
    """CUDA configured but not available -> CPU only, one warning logged (caplog), no exception."""
    from taosmd.vector_memory import VectorMemory
    from taosmd import config as cfg

    data_dir = tmp_path / "taosmd"
    data_dir.mkdir()

    # Set config roles
    cfg.set_model_role("embedder_provider", "CUDAExecutionProvider", data_dir=str(data_dir))
    cfg.set_model_role("embedder_gpu_mem_mb", "2048", data_dir=str(data_dir))

    # Mock onnxruntime to simulate CUDA NOT available
    mock_ort = _ORTMock()
    mock_ort.set_available_providers(["CPUExecutionProvider"])
    original_ort = _patch_onnxruntime(mock_ort)
    original_tokenizer, _ = _patch_tokenizer()
    
    try:
        caplog.set_level(logging.WARNING)
        vm = VectorMemory(
            db_path=str(data_dir / "vm.db"),
            embed_mode="onnx",
            onnx_path="models/minilm-onnx",
            data_dir=str(data_dir),
        )
        _run(vm.init())

        # Check that InferenceSession was called with CPU only
        providers_arg = mock_ort.last_providers
        assert providers_arg is not None, "InferenceSession not called"
        assert providers_arg == ["CPUExecutionProvider"], f"Expected CPU only, got {providers_arg}"

        # Check that exactly one warning was logged
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1, f"Expected 1 warning, got {len(warnings)}: {[w.message for w in warnings]}"
        assert "CUDAExecutionProvider" in warnings[0].message
        assert "CPUExecutionProvider" in warnings[0].message

        _run(vm.close())
    finally:
        _unpatch_onnxruntime(original_ort)
        _unpatch_tokenizer(original_tokenizer)


def test_provider_not_in_store_marker(tmp_path):
    """The store_meta mode string is identical with and without the provider configured."""
    from taosmd.vector_memory import VectorMemory
    from taosmd import config as cfg

    data_dir = tmp_path / "taosmd"
    data_dir.mkdir()

    # Test without provider config
    mock_ort1 = _ORTMock()
    mock_ort1.set_available_providers(["CPUExecutionProvider"])
    original_ort1 = _patch_onnxruntime(mock_ort1)
    original_tokenizer1, _ = _patch_tokenizer()
    
    try:
        vm1 = VectorMemory(
            db_path=str(data_dir / "vm1.db"),
            embed_mode="onnx",
            onnx_path="models/minilm-onnx",
        )
        _run(vm1.init())
        mode1 = vm1._conn.execute("SELECT value FROM store_meta WHERE key='mode'").fetchone()["value"]
        _run(vm1.close())
    finally:
        _unpatch_onnxruntime(original_ort1)
        _unpatch_tokenizer(original_tokenizer1)

    # Test with provider config
    cfg.set_model_role("embedder_provider", "CUDAExecutionProvider", data_dir=str(data_dir))
    cfg.set_model_role("embedder_gpu_mem_mb", "2048", data_dir=str(data_dir))

    mock_ort2 = _ORTMock()
    mock_ort2.set_available_providers(["CPUExecutionProvider", "CUDAExecutionProvider"])
    original_ort2 = _patch_onnxruntime(mock_ort2)
    original_tokenizer2, _ = _patch_tokenizer()
    
    try:
        vm2 = VectorMemory(
            db_path=str(data_dir / "vm2.db"),
            embed_mode="onnx",
            onnx_path="models/minilm-onnx",
            data_dir=str(data_dir),
        )
        _run(vm2.init())
        mode2 = vm2._conn.execute("SELECT value FROM store_meta WHERE key='mode'").fetchone()["value"]
        _run(vm2.close())
    finally:
        _unpatch_onnxruntime(original_ort2)
        _unpatch_tokenizer(original_tokenizer2)

    # The mode strings must be identical (provider is a transport, not part of vector space)
    assert mode1 == mode2, f"Mode mismatch: {mode1!r} != {mode2!r}"


def test_ensure_stores_passes_data_dir(tmp_path, caplog):
    """Ensure _ensure_stores passes data_dir to VectorMemory so provider
    config from a non-default data dir is respected.

    After the fix, data_dir=str(path) is passed to VectorMemory, so
    VectorMemory._data_dir is set to the served data dir. Without the fix,
    _data_dir would be None and provider config would be read from the
    default location.
    """
    import taosmd.api as api_mod

    data_dir = tmp_path / "taosmd"
    data_dir.mkdir()

    # Write config.json with embedder_provider set into the served data dir
    (data_dir / "config.json").write_text(
        json.dumps({"models": {"embedder_provider": "CUDAExecutionProvider"}})
    )

    # Call _ensure_stores - data_dir should be passed to VectorMemory
    stores = _run(api_mod._ensure_stores(data_dir=str(data_dir)))

    # Verify data_dir was stored in the result
    assert stores["data_dir"] == str(data_dir), (
        f"Expected data_dir={str(data_dir)}, got {stores['data_dir']}"
    )

    # Verify VectorMemory received data_dir by checking _data_dir
    vm = stores["vector"]
    assert vm._data_dir == str(data_dir), (
        f"Expected VectorMemory._data_dir={str(data_dir)}, got {vm._data_dir!r}"
    )


def test_gpu_mem_limit_json_int(tmp_path, caplog):
    """JSON int embedder_gpu_mem_mb=2048 should produce gpu_mem_limit 2048*1024*1024.

    Finding 2: embedder_gpu_mem_mb as a JSON integer is silently ignored
    because config.get_model_role returns only non-empty strings. The fix
    adds int acceptance; without it, JSON int 2048 produces no gpu_mem_limit.
    """
    from taosmd.vector_memory import VectorMemory

    data_dir = tmp_path / "taosmd"
    data_dir.mkdir()

    # Write config.json with embedder_gpu_mem_mb as JSON integer (not string),
    # WITHOUT calling set_model_role so the raw JSON int is what get_model_role sees.
    config_path = data_dir / "config.json"
    config_path.write_text(
        json.dumps({"models": {"embedder_provider": "CUDAExecutionProvider", "embedder_gpu_mem_mb": 2048}})
    )

    # Mock onnxruntime to capture providers
    mock_ort = _ORTMock()
    with patch("sys.modules", {"onnxruntime": mock_ort}, create=True):
        original_ort = _patch_onnxruntime(mock_ort)
        original_tokenizer, _ = _patch_tokenizer()
        try:
            vm = VectorMemory(
                db_path=str(data_dir / "vm.db"),
                embed_mode="onnx",
                onnx_path="models/minilm-onnx",
                data_dir=str(data_dir),
            )
            _run(vm.init())

            # Check the provider was set with gpu_mem_limit
            providers_arg = mock_ort.last_providers
            assert providers_arg is not None, "InferenceSession not called"
            # gpu_mem_limit should be 2048 * 1024 * 1024 = 2147483648
            # Without the fix, get_model_role only accepts str, so JSON int 2048
            # returns None → no gpu_mem_limit → this assertion FAILS.
            assert any(
                isinstance(p, tuple) and p[0] == "CUDAExecutionProvider"
                and p[1]["gpu_mem_limit"] == 2048 * 1024 * 1024
                for p in providers_arg
            ), f"Expected CUDAExecutionProvider with gpu_mem_limit={2048*1024*1024}, got {providers_arg}"

            _run(vm.close())
        finally:
            _unpatch_onnxruntime(original_ort)
            _unpatch_tokenizer(original_tokenizer)


def test_gpu_mem_limit_rejects_nonpositive(tmp_path, caplog):
    """Parameterized: 0 and negative embedder_gpu_mem_mb values are rejected (no gpu_mem_limit).

    Finding 3: '0' and '-5' are accepted and give gpu_mem_limit 0 / -5242880.
    The fix makes <= 0 values log a warning and be ignored (same as unparseable).
    Without the fix, non-positive values pass through and set gpu_mem_limit.
    """
    from taosmd.vector_memory import VectorMemory
    from taosmd import config as cfg

    for val in ["0", "-5", 0, -5]:
        data_dir = tmp_path / "taosmd"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Write config.json with embedder_gpu_mem_mb as the test value
        # Also set embedder_provider so the gpu_mem_limit code path is exercised
        config_path = data_dir / "config.json"
        config_path.write_text(
            json.dumps({"models": {"embedder_provider": "CUDAExecutionProvider", "embedder_gpu_mem_mb": val}})
        )

        # Set config role
        cfg.set_model_role("embedder_provider", "CUDAExecutionProvider", data_dir=str(data_dir))
        cfg.set_model_role("embedder_gpu_mem_mb", str(val) if isinstance(val, int) else val, data_dir=str(data_dir))

        # Mock onnxruntime
        mock_ort = _ORTMock()
        with patch("sys.modules", {"onnxruntime": mock_ort}, create=True):
            original_ort = _patch_onnxruntime(mock_ort)
            original_tokenizer, _ = _patch_tokenizer()
            try:
                vm = VectorMemory(
                    db_path=str(data_dir / "vm.db"),
                    embed_mode="onnx",
                    onnx_path="models/minilm-onnx",
                    data_dir=str(data_dir),
                )
                _run(vm.init())

                providers_arg = mock_ort.last_providers
                assert providers_arg is not None, "InferenceSession not called"

                # No gpu_mem_limit should be present for non-positive values
                has_gpu_limit = any(
                    isinstance(p, tuple) and p[0] == "CUDAExecutionProvider"
                    and "gpu_mem_limit" in p[1]
                    for p in providers_arg
                )
                assert not has_gpu_limit, (
                    f"Expected no gpu_mem_limit for val={val!r}, "
                    f"but got providers: {providers_arg}"
                )

                _run(vm.close())
            finally:
                _unpatch_onnxruntime(original_ort)
                _unpatch_tokenizer(original_tokenizer)
