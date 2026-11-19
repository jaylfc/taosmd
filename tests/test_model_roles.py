"""Tests for model role resolution (tsk-x6kvdp).

Validates that:
- Default model strings are unchanged without config
- Config roles override defaults
- Explicit arguments beat config
- Unknown roles are rejected
- Memory model resolution is untouched
"""

import asyncio
import json

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
    assert reranker._provider == "CPUExecutionProvider", f"reranker provider: {reranker._provider}"


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
    assert reranker._provider == "CUDAExecutionProvider", f"reranker provider: {reranker._provider}"


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


def test_reranker_uses_served_data_dir_path(tmp_path, monkeypatch):
    """Through the REAL api.search with a recipe whose reranker is on,
    config cross_encoder_path set in the SERVED data dir (not HOME) ->
    the reranker is constructed with that path (record via monkeypatch of CrossEncoderReranker.__init__ or the session creation)."""
    from taosmd import api as taosmd_api
    from taosmd.cross_encoder import CrossEncoderReranker
    from taosmd import recipes as recipes_module

    # Create served data dir with config
    served_data_dir = tmp_path / "served-data"
    served_data_dir.mkdir()
    home_data_dir = tmp_path / "home-data"
    home_data_dir.mkdir()

    # Put cross-encoder model in served data dir
    served_model_dir = served_data_dir / "models" / "cross-encoder-onnx"
    served_model_dir.mkdir(parents=True)
    (served_model_dir / "model.onnx").write_bytes(b"placeholder")

    # Put a DIFFERENT model in home data dir (should NOT be used)
    home_model_dir = home_data_dir / "models" / "cross-encoder-onnx"
    home_model_dir.mkdir(parents=True)
    (home_model_dir / "model.onnx").write_bytes(b"home-placeholder")

    # Config in served data dir points to served model
    config_path = served_data_dir / "config.json"
    config_path.write_text(json.dumps({
        "models": {
            "cross_encoder_path": "models/cross-encoder-onnx",
            "cross_encoder_provider": "CPUExecutionProvider"
        }
    }))

    # No config in home data dir

    # Monkeypatch CrossEncoderReranker.__init__ to capture the resolved _onnx_path
    captured_path = {}
    original_init = CrossEncoderReranker.__init__

    def capturing_init(self, onnx_path=None, min_confidence=None, *, provider=None, data_dir=None):
        # Call original to let it resolve the path
        original_init(self, onnx_path=onnx_path, min_confidence=min_confidence, provider=provider, data_dir=data_dir)
        # Capture the resolved path
        captured_path["onnx_path"] = self._onnx_path
        captured_path["data_dir"] = data_dir

    monkeypatch.setattr(CrossEncoderReranker, "__init__", capturing_init)

    # Also patch available to True and _load to no-op for the dummy instance
    monkeypatch.setattr(CrossEncoderReranker, "available", property(lambda self: True))
    monkeypatch.setattr(CrossEncoderReranker, "_load", lambda self: None)
    monkeypatch.setattr(CrossEncoderReranker, "_score", lambda self, q, r: [0.5] * len(r))

    # Clear stores cache
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})

    # Create a recipe with reranker enabled (using bge-v2-m3 which triggers CrossEncoderReranker)
    from taosmd.recipes import Recipe
    test_recipe = Recipe(
        id="test-rerank",
        name="test-rerank",
        retrieval={"reranker": "bge-v2-m3", "strategy": "thorough", "limit": 5, "candidate_top_k": 20, "fusion": "boost", "adjacent_neighbors": 0, "graph_expansion": 0, "llm_reranker": False, "late_interaction": False, "colbert_model": "", "embed_model": "minilm-onnx"},
        ingest={},
        generator={},
        librarian={},
    )

    # Mock resolve_recipe to return our test recipe
    def mock_resolve_recipe(agent, data_dir=None):
        return test_recipe
    monkeypatch.setattr(recipes_module, "resolve_recipe", mock_resolve_recipe)

    # Run search through the real API with served data dir
    import asyncio
    asyncio.run(taosmd_api.search(
        query="test",
        agent="test-agent",
        data_dir=str(served_data_dir),
        limit=5
    ))

    # The reranker should have been constructed with the served data dir's model path
    # The config value is used as-is (relative path)
    assert captured_path["onnx_path"] == "models/cross-encoder-onnx", \
        f"Expected onnx_path='models/cross-encoder-onnx', got {captured_path['onnx_path']}"
    assert captured_path["data_dir"] == str(served_data_dir), \
        f"Expected data_dir={served_data_dir}, got {captured_path['data_dir']}"
