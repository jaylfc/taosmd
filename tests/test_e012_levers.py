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
