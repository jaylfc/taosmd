"""Tests for model role resolution (tsk-x6kvdp).

Validates that:
- Default model strings are unchanged without config
- Config roles override defaults
- Explicit arguments beat config
- Unknown roles are rejected
- Memory model resolution is untouched
"""

import asyncio

import pytest


def _run(coro):
    return asyncio.run(coro)


def test_defaults_unchanged_without_config(tmp_path):
    """No config.json → crystallize/reflect/expand_query_llm use 'qwen3:4b',
    reranker uses 'models/cross-encoder-onnx' and CPU provider."""
    from taosmd.crystallize import CrystalStore
    from taosmd.reflect import InsightStore
    from taosmd.query_expansion import expand_query_llm
    from taosmd.cross_encoder import CrossEncoderReranker

    data_dir = tmp_path / "taosmd"
    data_dir.mkdir()

    # No config.json exists
    assert not (data_dir / "config.json").exists()

    # Test crystallize
    cs = CrystalStore(db_path=data_dir / "crystals.db")
    _run(cs.init())
    try:
        # Stub llm_complete to capture model
        import taosmd.llm_complete as lc
        original = lc.llm_complete
        captured_model = {}

        async def capture_model(client, url, model, *args, **kwargs):
            captured_model["model"] = model
            return "NARRATIVE: test\nOUTCOMES:\n- ok\nLESSONS:\n- none\nFILES:\n- none"

        lc.llm_complete = capture_model
        try:
            _run(cs.crystallize(
                session_id="test-session",
                turns=[{"role": "user", "content": "hello", "timestamp": 1.0}],
                agent_name=None,
                llm_url="http://localhost:11434",
                data_dir=str(data_dir),
            ))
        finally:
            lc.llm_complete = original

        assert captured_model.get("model") == "qwen3:4b", f"crystallize model: {captured_model.get('model')}"
    finally:
        _run(cs.close())

    # Test reflect
    # Create a minimal KG stub
    class KGStub:
        async def query_entity(self, entity, track_access=False):
            return [{"subject": "a", "predicate": "p", "object": "b"},
                    {"subject": "c", "predicate": "p", "object": "d"}]
        async def timeline(self, limit=100):
            return [{"subject": "a", "predicate": "p", "object": "b"},
                    {"subject": "c", "predicate": "p", "object": "d"}]

    kg = KGStub()
    is_store = InsightStore(db_path=data_dir / "insights.db")
    _run(is_store.init())
    try:
        import taosmd.llm_complete as lc
        original = lc.llm_complete
        captured_model = {}

        async def capture_model(client, url, model, *args, **kwargs):
            captured_model["model"] = model
            return "insight text"

        lc.llm_complete = capture_model
        try:
            _run(is_store.reflect(kg=kg, llm_url="http://localhost:11434", data_dir=str(data_dir)))
        finally:
            lc.llm_complete = original

        assert captured_model.get("model") == "qwen3:4b", f"reflect model: {captured_model.get('model')}"
    finally:
        _run(is_store.close())

    # Test expand_query_llm
    captured_model = {}

    async def capture_model(client, url, model, *args, **kwargs):
        captured_model["model"] = model
        return "REFORMULATIONS:\n1. test\nENTITIES:\n- test"

    import taosmd.llm_complete as lc
    original = lc.llm_complete
    lc.llm_complete = capture_model
    try:
        _run(expand_query_llm("test query", llm_url="http://localhost:11434", data_dir=str(data_dir)))
    finally:
        lc.llm_complete = original

    assert captured_model.get("model") == "qwen3:4b", f"expand_query_llm model: {captured_model.get('model')}"

    # Test CrossEncoderReranker defaults
    # We don't have a model, so available=False, but we can check the constructor defaults
    reranker = CrossEncoderReranker(data_dir=str(data_dir))
    assert reranker._onnx_path == "models/cross-encoder-onnx", f"reranker path: {reranker._onnx_path}"
    # Provider is hardcoded in _load to CPUExecutionProvider


def test_config_role_overrides_default(tmp_path):
    """set_model_role() → functions use configured model."""
    from taosmd.crystallize import CrystalStore
    from taosmd.reflect import InsightStore
    from taosmd.query_expansion import expand_query_llm
    from taosmd.cross_encoder import CrossEncoderReranker
    from taosmd import config as cfg

    data_dir = tmp_path / "taosmd"
    data_dir.mkdir()

    # Set config roles
    cfg.set_model_role("crystallize", "ollama:gemma4:31b", data_dir=str(data_dir))
    cfg.set_model_role("reflect", "ollama:granite4:8b", data_dir=str(data_dir))
    cfg.set_model_role("query_expansion", "ollama:llama3.1:8b", data_dir=str(data_dir))
    cfg.set_model_role("cross_encoder_path", "/custom/path", data_dir=str(data_dir))
    cfg.set_model_role("cross_encoder_provider", "CUDAExecutionProvider", data_dir=str(data_dir))

    # Test crystallize
    cs = CrystalStore(db_path=data_dir / "crystals.db")
    _run(cs.init())
    try:
        import taosmd.llm_complete as lc
        original = lc.llm_complete
        captured_model = {}

        async def capture_model(client, url, model, *args, **kwargs):
            captured_model["model"] = model
            return "NARRATIVE: test\nOUTCOMES:\n- ok\nLESSONS:\n- none\nFILES:\n- none"

        lc.llm_complete = capture_model
        try:
            _run(cs.crystallize(
                session_id="test-session",
                turns=[{"role": "user", "content": "hello", "timestamp": 1.0}],
                agent_name=None,
                llm_url="http://localhost:11434",
                data_dir=str(data_dir),
            ))
        finally:
            lc.llm_complete = original

        assert captured_model.get("model") == "ollama:gemma4:31b", f"crystallize model: {captured_model.get('model')}"
    finally:
        _run(cs.close())

    # Test reflect
    class KGStub:
        async def query_entity(self, entity, track_access=False):
            return [{"subject": "a", "predicate": "p", "object": "b"},
                    {"subject": "c", "predicate": "p", "object": "d"}]
        async def timeline(self, limit=100):
            return [{"subject": "a", "predicate": "p", "object": "b"},
                    {"subject": "c", "predicate": "p", "object": "d"}]

    kg = KGStub()
    is_store = InsightStore(db_path=data_dir / "insights.db")
    _run(is_store.init())
    try:
        import taosmd.llm_complete as lc
        original = lc.llm_complete
        captured_model = {}

        async def capture_model(client, url, model, *args, **kwargs):
            captured_model["model"] = model
            return "insight text"

        lc.llm_complete = capture_model
        try:
            _run(is_store.reflect(kg=kg, llm_url="http://localhost:11434", data_dir=str(data_dir)))
        finally:
            lc.llm_complete = original

        assert captured_model.get("model") == "ollama:granite4:8b", f"reflect model: {captured_model.get('model')}"
    finally:
        _run(is_store.close())

    # Test expand_query_llm
    captured_model = {}
    async def capture_model(client, url, model, *args, **kwargs):
        captured_model["model"] = model
        return "REFORMULATIONS:\n1. test\nENTITIES:\n- test"

    import taosmd.llm_complete as lc
    original = lc.llm_complete
    lc.llm_complete = capture_model
    try:
        _run(expand_query_llm("test query", llm_url="http://localhost:11434", data_dir=str(data_dir)))
    finally:
        lc.llm_complete = original

    assert captured_model.get("model") == "ollama:llama3.1:8b", f"expand_query_llm model: {captured_model.get('model')}"

    # Test CrossEncoderReranker config
    reranker = CrossEncoderReranker(data_dir=str(data_dir))
    assert reranker._onnx_path == "/custom/path", f"reranker path: {reranker._onnx_path}"
    # Note: provider is used in _load, we can't easily test without patching


def test_explicit_argument_beats_config(tmp_path):
    """Config sets a role, explicit argument wins."""
    from taosmd.crystallize import CrystalStore
    from taosmd.reflect import InsightStore
    from taosmd.query_expansion import expand_query_llm
    from taosmd.cross_encoder import CrossEncoderReranker
    from taosmd import config as cfg

    data_dir = tmp_path / "taosmd"
    data_dir.mkdir()

    cfg.set_model_role("crystallize", "ollama:gemma4:31b", data_dir=str(data_dir))
    cfg.set_model_role("reflect", "ollama:granite4:8b", data_dir=str(data_dir))
    cfg.set_model_role("query_expansion", "ollama:llama3.1:8b", data_dir=str(data_dir))

    # Test crystallize with explicit model
    cs = CrystalStore(db_path=data_dir / "crystals.db")
    _run(cs.init())
    try:
        import taosmd.llm_complete as lc
        original = lc.llm_complete
        captured_model = {}

        async def capture_model(client, url, model, *args, **kwargs):
            captured_model["model"] = model
            return "NARRATIVE: test\nOUTCOMES:\n- ok\nLESSONS:\n- none\nFILES:\n- none"

        lc.llm_complete = capture_model
        try:
            _run(cs.crystallize(
                session_id="test-session",
                turns=[{"role": "user", "content": "hello", "timestamp": 1.0}],
                agent_name=None,
                llm_url="http://localhost:11434",
                model="explicit-model",
                data_dir=str(data_dir),
            ))
        finally:
            lc.llm_complete = original

        assert captured_model.get("model") == "explicit-model", f"crystallize model: {captured_model.get('model')}"
    finally:
        _run(cs.close())

    # Test reflect with explicit model
    class KGStub:
        async def query_entity(self, entity, track_access=False):
            return [{"subject": "a", "predicate": "p", "object": "b"},
                    {"subject": "c", "predicate": "p", "object": "d"}]
        async def timeline(self, limit=100):
            return [{"subject": "a", "predicate": "p", "object": "b"},
                    {"subject": "c", "predicate": "p", "object": "d"}]

    kg = KGStub()
    is_store = InsightStore(db_path=data_dir / "insights.db")
    _run(is_store.init())
    try:
        import taosmd.llm_complete as lc
        original = lc.llm_complete
        captured_model = {}

        async def capture_model(client, url, model, *args, **kwargs):
            captured_model["model"] = model
            return "insight text"

        lc.llm_complete = capture_model
        try:
            _run(is_store.reflect(kg=kg, llm_url="http://localhost:11434", model="explicit-reflect", data_dir=str(data_dir)))
        finally:
            lc.llm_complete = original

        assert captured_model.get("model") == "explicit-reflect", f"reflect model: {captured_model.get('model')}"
    finally:
        _run(is_store.close())

    # Test expand_query_llm with explicit model
    captured_model = {}
    async def capture_model(client, url, model, *args, **kwargs):
        captured_model["model"] = model
        return "REFORMULATIONS:\n1. test\nENTITIES:\n- test"

    import taosmd.llm_complete as lc
    original = lc.llm_complete
    lc.llm_complete = capture_model
    try:
        _run(expand_query_llm("test query", llm_url="http://localhost:11434", model="explicit-expand", data_dir=str(data_dir)))
    finally:
        lc.llm_complete = original

    assert captured_model.get("model") == "explicit-expand", f"expand_query_llm model: {captured_model.get('model')}"

    # Test CrossEncoderReranker with explicit args
    reranker = CrossEncoderReranker(onnx_path="/explicit/path", provider="CUDAExecutionProvider", data_dir=str(data_dir))
    assert reranker._onnx_path == "/explicit/path"
    assert reranker._provider == "CUDAExecutionProvider"


def test_unknown_role_rejected(tmp_path):
    """set_model_role('nope', 'x') raises ValueError naming valid roles."""
    from taosmd import config as cfg

    data_dir = tmp_path / "taosmd"
    data_dir.mkdir()

    with pytest.raises(ValueError) as exc:
        cfg.set_model_role("nope", "x", data_dir=str(data_dir))
    assert "nope" in str(exc.value)
    # Should list valid roles
    valid_roles = {"crystallize", "reflect", "query_expansion", "cross_encoder_path", "cross_encoder_provider"}
    for role in valid_roles:
        assert role in str(exc.value), f"Valid role {role} not in error message"


def test_memory_model_resolution_untouched(tmp_path):
    """resolve_generator returns same value with/without models section."""
    from taosmd import generator_profiles as gp
    from taosmd import config as cfg

    data_dir = tmp_path / "taosmd"
    data_dir.mkdir()

    # Without models section
    result1 = gp.resolve_generator(agent=None, fallback="fallback-model", data_dir=str(data_dir))

    # With models section (but no memory_model pin)
    cfg.set_model_role("crystallize", "some-model", data_dir=str(data_dir))
    result2 = gp.resolve_generator(agent=None, fallback="fallback-model", data_dir=str(data_dir))

    # Should be the same (since resolve_generator doesn't use models section)
    assert result1 == result2, f"resolve_generator changed: {result1} vs {result2}"


def test_resolve_model_function(tmp_path):
    """Test the new resolve_model function directly."""
    from taosmd import generator_profiles as gp
    from taosmd import config as cfg

    data_dir = tmp_path / "taosmd"
    data_dir.mkdir()

    # No config → fallback
    assert gp.resolve_model("crystallize", fallback="qwen3:4b", data_dir=str(data_dir)) == "qwen3:4b"

    # With config
    cfg.set_model_role("crystallize", "custom-model", data_dir=str(data_dir))
    assert gp.resolve_model("crystallize", fallback="qwen3:4b", data_dir=str(data_dir)) == "custom-model"

    # Clear config value → fallback
    cfg.set_model_role("crystallize", "", clear=True, data_dir=str(data_dir))
    assert gp.resolve_model("crystallize", fallback="qwen3:4b", data_dir=str(data_dir)) == "qwen3:4b"

    # Explicit None fallback
    assert gp.resolve_model("crystallize", fallback=None, data_dir=str(data_dir)) is None