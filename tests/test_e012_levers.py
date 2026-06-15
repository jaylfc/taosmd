import importlib.util
import pathlib
import asyncio

_RUNNER = pathlib.Path(__file__).resolve().parent.parent / "benchmarks" / "longmemeval_runner.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("lme_runner", _RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeResp:
    def __init__(self, content, status=200):
        self.status_code = status
        self._content = content

    def json(self):
        return {"message": {"content": self._content}}


class _FakeClient:
    """Returns a fixed Ollama chat response for any .post()."""
    def __init__(self, content, status=200):
        self._content = content
        self._status = status

    async def post(self, *args, **kwargs):
        return _FakeResp(self._content, self._status)


class _BoomClient:
    async def post(self, *args, **kwargs):
        raise RuntimeError("boom")


def test_decompose_returns_lines_when_two_or_more():
    mod = _load_runner()
    out = asyncio.run(mod.decompose_query(_FakeClient("who did she marry\nwhen did they meet"), "Q"))
    assert out == ["who did she marry", "when did they meet"]


def test_decompose_falls_back_when_fewer_than_two_lines():
    mod = _load_runner()
    out = asyncio.run(mod.decompose_query(_FakeClient("only one line"), "original question"))
    assert out == ["original question"]


def test_decompose_falls_back_on_error():
    mod = _load_runner()
    out = asyncio.run(mod.decompose_query(_BoomClient(), "original question"))
    assert out == ["original question"]


def test_self_verify_returns_revised_when_nonempty():
    mod = _load_runner()
    out = asyncio.run(mod.self_verify_answer(_FakeClient("corrected answer"), "ctx", "Q", "draft answer"))
    assert out == "corrected answer"


def test_self_verify_keeps_draft_on_error():
    mod = _load_runner()
    out = asyncio.run(mod.self_verify_answer(_BoomClient(), "ctx", "Q", "draft answer"))
    assert out == "draft answer"


def test_self_verify_keeps_draft_when_revision_empty():
    mod = _load_runner()
    out = asyncio.run(mod.self_verify_answer(_FakeClient("   "), "ctx", "Q", "draft answer"))
    assert out == "draft answer"


def test_self_verify_noops_on_empty_draft():
    mod = _load_runner()
    out = asyncio.run(mod.self_verify_answer(_FakeClient("anything"), "ctx", "Q", ""))
    assert out == ""


def test_sample_dataset_no_seed_is_head_slice():
    mod = _load_runner()
    data = [{"id": i} for i in range(100)]
    assert mod.sample_dataset(data, 5, seed=None) == data[:5]


def test_sample_dataset_seed_is_deterministic_and_representative():
    mod = _load_runner()
    # Type-ordered like the oracle set: first 50 one type, next 50 another.
    data = [{"id": i, "t": "temporal" if i < 50 else "multi"} for i in range(100)]
    a = mod.sample_dataset(data, 20, seed=42)
    b = mod.sample_dataset(data, 20, seed=42)
    assert a == b  # same seed -> same sample, comparable across arms
    assert {d["t"] for d in a} == {"temporal", "multi"}  # not a single-type head slice
    assert mod.sample_dataset(data, 20, seed=7) != a  # different seed -> different sample
