"""Tests for ONNX embedder provider configuration (tsk-awwytg).

Validates that the embedder ONNX provider is configurable via the
config models section, with GPU memory limit support and graceful
fallback when the configured provider is unavailable.
"""

import asyncio
import logging
import sys
from unittest.mock import MagicMock

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
