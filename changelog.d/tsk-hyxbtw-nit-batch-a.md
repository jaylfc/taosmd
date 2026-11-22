### Fixed
- Import no longer raises on bad pyproject.toml: `_resolve_version()` now wraps the file read in a try/except so any OSError or UnicodeDecodeError falls through to metadata or "0+unknown" instead of propagating.
- CrossEncoderReranker signature restored: `__init__(self, onnx_path=None, min_confidence=None, *, provider=None)` makes `provider` keyword-only, so `CrossEncoderReranker('path', 0.5)` correctly sets `min_confidence=0.5`.
- Cross-encoder provider config now respected: the `provider` argument is passed to `onnxruntime.InferenceSession` (verified by monkeypatch test).
- Reranker uses served data_dir config: `api.search` constructs the reranker with the `cross_encoder_path` from the served data dir's config, not HOME.
