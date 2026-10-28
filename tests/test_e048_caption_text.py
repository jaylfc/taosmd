from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.e048 import retrieval


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_minilm_files(onnx_dir: Path) -> tuple[str, str]:
    onnx_dir.mkdir(parents=True, exist_ok=True)
    model_path = onnx_dir / "model.onnx"
    tok_path = onnx_dir / "tokenizer.json"
    model_bytes = b"minilm-model-bytes-for-test"
    tok_bytes = b"minilm-tokenizer-bytes-for-test"
    model_path.write_bytes(model_bytes)
    tok_path.write_bytes(tok_bytes)
    return _sha256_bytes(model_bytes), _sha256_bytes(tok_bytes)


class FakeResponse:
    def __init__(self, json_data=None, status_code=200):
        self._json_data = json_data or {}
        self.status_code = status_code

    def json(self):
        return self._json_data


class FakeHttpxClient:
    def __init__(self, get_responses=None, post_responses=None):
        self.get_responses = get_responses or []
        self.post_responses = post_responses or []
        self.get_calls = []
        self.post_calls = []
        self._get_idx = 0
        self._post_idx = 0

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        if self._get_idx < len(self.get_responses):
            resp = self.get_responses[self._get_idx]
            self._get_idx += 1
            return resp
        return FakeResponse({})

    def post(self, url, json=None, **kwargs):
        self.post_calls.append((url, json, kwargs))
        if self._post_idx < len(self.post_responses):
            resp = self.post_responses[self._post_idx]
            self._post_idx += 1
            return resp
        return FakeResponse({})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeVectorMemory:
    def __init__(self, db_path=None, embed_mode=None, onnx_path=None):
        self.db_path = db_path
        self.embed_mode = embed_mode
        self.onnx_path = onnx_path
        self.init_calls = 0
        self.embed_calls = []

    async def init(self):
        self.init_calls += 1
        return None

    async def embed(self, text, task="search_document"):
        self.embed_calls.append((text, task))
        # Deterministic vector from text hash
        h = hashlib.sha256((text + ":" + task).encode("utf-8")).digest()
        # Build a float vector of length 384 from hash bytes
        arr = np.frombuffer(h * 12, dtype=np.uint8)[:384].astype(np.float32)
        arr = arr / 255.0
        # Normalize to unit norm
        norm = np.linalg.norm(arr)
        if norm == 0:
            arr = np.ones(384, dtype=np.float32)
            norm = np.linalg.norm(arr)
        arr = (arr / norm).astype(np.float32)
        return arr.tolist()  # return as list like the real API might give, but module will convert

    @classmethod
    def _from_module(cls):
        return cls


def _det_vec_from_text(text, task):
    h = hashlib.sha256((text + ":" + task).encode("utf-8")).digest()
    arr = np.frombuffer(h * 12, dtype=np.uint8)[:384].astype(np.float32)
    arr = arr / 255.0
    norm = np.linalg.norm(arr)
    if norm == 0:
        arr = np.ones(384, dtype=np.float32)
        norm = np.linalg.norm(arr)
    arr = (arr / norm).astype(np.float32)
    return arr


def test_captions_requested_once_and_cached(monkeypatch, tmp_path):
    from benchmarks.e048 import caption_text

    onnx_dir = tmp_path / "models" / "minilm-onnx"
    model_sha, tok_sha = _write_minilm_files(onnx_dir)
    cache_path = tmp_path / "cache.jsonl"

    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.png"
    img1.write_bytes(b"image1bytes")
    img2.write_bytes(b"image2bytes")

    # Fake httpx client
    fake_client = FakeHttpxClient(
        get_responses=[
            FakeResponse({"models": [{"name": "llava:7b", "digest": "abc123digest"}]}),
        ],
        post_responses=[
            FakeResponse({"response": "A cat on a mat."}),
            FakeResponse({"response": "A dog in the park."}),
        ],
    )

    import httpx

    def fake_client_ctor(*args, **kwargs):
        return fake_client

    monkeypatch.setattr(httpx, "Client", fake_client_ctor)

    # Fake VectorMemory
    class StubVM:
        instances = []

        def __init__(self, db_path=None, embed_mode=None, onnx_path=None):
            self.db_path = db_path
            self.embed_mode = embed_mode
            self.onnx_path = onnx_path
            StubVM.instances.append(self)

        async def init(self):
            return None

        async def embed(self, text, task="search_document"):
            v = _det_vec_from_text(text, task)
            return v

    monkeypatch.setattr(caption_text, "VectorMemory", StubVM)

    embedder = caption_text.CaptionTextImageEmbedder(
        captioner_url="http://127.0.0.1:11434",
        captioner_model="llava:7b",
        onnx_dir=str(onnx_dir),
        cache_path=str(cache_path),
    )

    paths = [str(img1), str(img2)]
    out = embedder.embed_items(paths)
    assert out.shape == (2, 384)
    assert out.dtype == np.float32
    # Check unit norm
    for i in range(out.shape[0]):
        assert abs(np.linalg.norm(out[i]) - 1.0) < 1e-5
    # Posts happened twice (once per image)
    assert len(fake_client.post_calls) == 2

    # Second call should use cache - zero new posts
    out2 = embedder.embed_items(paths)
    assert len(fake_client.post_calls) == 2  # no new posts
    assert np.allclose(out, out2)


def test_cache_line_with_different_captioner_digest_not_reused(monkeypatch, tmp_path):
    from benchmarks.e048 import caption_text

    onnx_dir = tmp_path / "models" / "minilm-onnx"
    _write_minilm_files(onnx_dir)
    cache_path = tmp_path / "cache.jsonl"

    img1 = tmp_path / "img1.jpg"
    img1.write_bytes(b"image1bytes")

    fake_client = FakeHttpxClient(
        get_responses=[
            FakeResponse({"models": [{"name": "llava:7b", "digest": "digest-v2"}]}),
        ],
        post_responses=[
            FakeResponse({"response": "New caption."}),
        ],
    )

    import httpx

    monkeypatch.setattr(httpx, "Client", lambda *args, **kwargs: fake_client)

    class StubVM:
        def __init__(self, *args, **kwargs):
            pass

        async def init(self):
            return None

        async def embed(self, text, task="search_document"):
            v = _det_vec_from_text(text, task)
            return v

    monkeypatch.setattr(caption_text, "VectorMemory", StubVM)

    # Write old cache entry with different digest
    img_sha = _sha256_bytes(img1.read_bytes())
    prompt_sha = hashlib.sha256(caption_text.CAPTION_PROMPT.encode("utf-8")).hexdigest()
    old_line = {
        "sha256": img_sha,
        "captioner_model": "llava:7b",
        "captioner_digest": "digest-v1",
        "prompt_sha256": prompt_sha,
        "caption": "Old caption.",
    }
    cache_path.write_text(json.dumps(old_line) + "\n")

    embedder = caption_text.CaptionTextImageEmbedder(
        captioner_url="http://127.0.0.1:11434",
        captioner_model="llava:7b",
        onnx_dir=str(onnx_dir),
        cache_path=str(cache_path),
    )

    paths = [str(img1)]
    out = embedder.embed_items(paths)
    assert out.shape == (1, 384)
    # Should have made a POST because digest differs
    assert len(fake_client.post_calls) == 1
    # Cache should now have new entry with new digest
    lines = cache_path.read_text().strip().splitlines()
    assert len(lines) == 2  # old + new


def test_empty_caption_raises(monkeypatch, tmp_path):
    from benchmarks.e048 import caption_text

    onnx_dir = tmp_path / "models" / "minilm-onnx"
    _write_minilm_files(onnx_dir)
    cache_path = tmp_path / "cache.jsonl"

    img1 = tmp_path / "img1.jpg"
    img1.write_bytes(b"image1bytes")

    fake_client = FakeHttpxClient(
        get_responses=[
            FakeResponse({"models": [{"name": "llava:7b", "digest": "abc123"}]}),
        ],
        post_responses=[
            FakeResponse({"response": "   "}),
        ],
    )

    import httpx

    monkeypatch.setattr(httpx, "Client", lambda *args, **kwargs: fake_client)

    class StubVM:
        def __init__(self, *args, **kwargs):
            pass

        async def init(self):
            return None

        async def embed(self, text, task="search_document"):
            return _det_vec_from_text(text, task)

    monkeypatch.setattr(caption_text, "VectorMemory", StubVM)

    embedder = caption_text.CaptionTextImageEmbedder(
        captioner_url="http://127.0.0.1:11434",
        captioner_model="llava:7b",
        onnx_dir=str(onnx_dir),
        cache_path=str(cache_path),
    )

    try:
        embedder.embed_items([str(img1)])
        assert False, "Should have raised"
    except RuntimeError as e:
        assert str(img1) in str(e) or "img1" in str(e)


def test_model_missing_from_tags_raises(monkeypatch, tmp_path):
    from benchmarks.e048 import caption_text

    onnx_dir = tmp_path / "models" / "minilm-onnx"
    _write_minilm_files(onnx_dir)
    cache_path = tmp_path / "cache.jsonl"

    fake_client = FakeHttpxClient(
        get_responses=[
            FakeResponse({"models": [{"name": "other:7b", "digest": "abc123"}]}),
        ],
        post_responses=[],
    )

    import httpx

    monkeypatch.setattr(httpx, "Client", lambda *args, **kwargs: fake_client)

    class StubVM:
        def __init__(self, *args, **kwargs):
            pass

        async def init(self):
            return None

        async def embed(self, text, task="search_document"):
            return _det_vec_from_text(text, task)

    monkeypatch.setattr(caption_text, "VectorMemory", StubVM)

    try:
        caption_text.CaptionTextImageEmbedder(
            captioner_url="http://127.0.0.1:11434",
            captioner_model="llava:7b",
            onnx_dir=str(onnx_dir),
            cache_path=str(cache_path),
        )
        assert False, "Should have raised"
    except RuntimeError:
        pass


def test_embed_items_rows_unit_norm_float32_and_provenance(monkeypatch, tmp_path):
    from benchmarks.e048 import caption_text

    onnx_dir = tmp_path / "models" / "minilm-onnx"
    model_sha, tok_sha = _write_minilm_files(onnx_dir)
    cache_path = tmp_path / "cache.jsonl"

    img1 = tmp_path / "img1.jpg"
    img1.write_bytes(b"image1bytes")

    fake_client = FakeHttpxClient(
        get_responses=[
            FakeResponse({"models": [{"name": "llava:7b", "digest": "dig123"}]}),
        ],
        post_responses=[
            FakeResponse({"response": "A photo."}),
        ],
    )

    import httpx

    monkeypatch.setattr(httpx, "Client", lambda *args, **kwargs: fake_client)

    class StubVM:
        def __init__(self, *args, **kwargs):
            pass

        async def init(self):
            return None

        async def embed(self, text, task="search_document"):
            return _det_vec_from_text(text, task)

    monkeypatch.setattr(caption_text, "VectorMemory", StubVM)

    embedder = caption_text.CaptionTextImageEmbedder(
        captioner_url="http://127.0.0.1:11434",
        captioner_model="llava:7b",
        onnx_dir=str(onnx_dir),
        cache_path=str(cache_path),
    )

    paths = [str(img1)]
    out = embedder.embed_items(paths)
    assert out.shape == (1, 384)
    assert out.dtype == np.float32
    assert abs(np.linalg.norm(out[0]) - 1.0) < 1e-5

    prov = embedder.provenance()
    # Check required provenance keys
    for k in retrieval.REQUIRED_PROVENANCE:
        assert k in prov
    # Check extra keys
    assert prov["model_id"] == "caption-then-text"
    assert prov["captioner_model"] == "llava:7b"
    assert prov["captioner_digest"] == "dig123"
    assert "caption_prompt_sha256" in prov
    assert prov["captions_cache"] == str(cache_path)
    assert "captions_cache_sha256" in prov
    assert prov["runtime"] == "onnxruntime+ollama"
    assert prov["execution_provider"] == "CPUExecutionProvider"
    assert prov["dtype"] == "float32"
    # model_files includes captioner entry
    assert "captioner:llava:7b" in prov["model_files"]
    assert prov["model_files"]["captioner:llava:7b"] == "dig123"
    # includes model.onnx and tokenizer.json
    model_files = prov["model_files"]
    # Find entries
    assert any(k.endswith("model.onnx") for k in model_files.keys()) or "model.onnx" in model_files or any("model.onnx" in k for k in model_files)
