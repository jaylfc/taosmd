# JOB-002: Make CrossEncoderReranker's default model path cwd-independent

**Status: ON HOLD (verified 2026-07-21). DO NOT START THIS JOB.**

The bug this job describes is still real: `taosmd/cross_encoder.py` line 21
still defaults `onnx_path` to the relative `"models/cross-encoder-onnx"`.
The fix written below is what has gone stale, and applying it as written
would make things worse. Issue #199 covers the same ground and supersedes
this job.

Two reasons this is no longer a self-contained job:

1. The read path and the download path have SEPARATE defaults.
   `taosmd/api.py` line 635 builds `CrossEncoderReranker()` with no argument,
   and line 641 calls `_recipes.ensure_reranker_model(block=False)`, which
   has its own default of `"models/cross-encoder-onnx"` at
   `taosmd/recipes.py` line 457. The steps below change only the first one.
   That would leave the downloader writing to `$CWD/models/` while the
   reranker looks in the repo root, so `available` would never become true
   and a fresh deployment would re-fire a roughly 600MB download on every
   single search. That is a regression, not a fix.
2. Resolving against the repo root is itself the wrong answer for a wheel
   install, where the repo root is site-packages and is not writable. Issue
   #199 asks for the reranker path to go through the same mechanism as the
   embedder (`_resolve_onnx_path(data_dir, ...)` in `taosmd/api.py`) or an
   env override, and separately asks whether a live search should be pulling
   600MB at all. Choosing between those is a design decision.

Leave this to the primary session. The file is kept for history.

Read docs/agent-jobs/README.md first and follow its absolute rules.

- Branch: `fix/cross-encoder-cwd-path` (from `origin/master`)
- Commit message: `fix(rerank): resolve the default cross-encoder path relative to the repo, not the cwd`
- PR title: `fix(rerank): resolve the default cross-encoder path relative to the repo, not the cwd`
- Allowed files: `taosmd/cross_encoder.py`, `tests/test_cross_encoder_path.py` (new) ONLY.

## The bug

`taosmd/cross_encoder.py` line 21: the constructor default is the RELATIVE
path `"models/cross-encoder-onnx"`. Whether the reranker finds its model
then depends on the caller's current working directory: started from the
repo root it works, started from anywhere else it silently reports the model
missing. Explicit `onnx_path` arguments must keep working exactly as today.

## Steps

1. `git fetch origin && git checkout -b fix/cross-encoder-cwd-path origin/master`
2. Read `taosmd/cross_encoder.py` fully (about 100 lines) before editing.
3. Write the failing test FIRST in a new file `tests/test_cross_encoder_path.py`:

```python
"""The default cross-encoder model path must not depend on the cwd."""

import os

from taosmd.cross_encoder import CrossEncoderReranker, _default_onnx_path


def test_default_path_is_absolute():
    assert os.path.isabs(_default_onnx_path())


def test_default_path_independent_of_cwd(tmp_path, monkeypatch):
    before = _default_onnx_path()
    monkeypatch.chdir(tmp_path)
    assert _default_onnx_path() == before


def test_explicit_path_is_untouched(tmp_path):
    r = CrossEncoderReranker(onnx_path=str(tmp_path / "nowhere"))
    assert r._onnx_path == str(tmp_path / "nowhere")
    assert r.available is False  # no model there, and that is fine
```

4. Run it, expect ImportError on `_default_onnx_path`:
   `python3 -m pytest tests/test_cross_encoder_path.py -q`
5. Implement in `taosmd/cross_encoder.py`. Add near the top of the file
   (after the imports, before the class):

```python
def _default_onnx_path() -> str:
    """Default model dir, resolved against the repo root, not the cwd.

    The historical default was the relative "models/cross-encoder-onnx",
    which only worked when the process started in the repo root. Resolve
    the same directory against this file's repo checkout instead. An
    explicit onnx_path argument bypasses this entirely.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "models", "cross-encoder-onnx")
```

   Add `import os` to the imports ONLY if it is not already imported.
   Change the constructor signature from
   `def __init__(self, onnx_path: str = "models/cross-encoder-onnx"):` to
   `def __init__(self, onnx_path: str | None = None):` and make the first
   line of the body:
   `self._onnx_path = onnx_path if onnx_path is not None else _default_onnx_path()`
6. Check `available`: read the class for an `available` property; if the
   attribute is named differently, adapt the third test to whatever the real
   "model not loaded" signal is, and say so in the PR body.
7. `python3 -m pytest tests/test_cross_encoder_path.py -q` (3 passed), then
   the FULL suite `python3 -m pytest -q -m "not slow"` (all green; the
   existing reranker tests prove explicit-path behavior is unchanged), then
   `python3 -m ruff check taosmd/cross_encoder.py tests/test_cross_encoder_path.py`.
8. One commit, push, open the PR. PR body: the bug (cwd-dependent default),
   the fix (repo-root-resolved default, explicit arg untouched), test counts.

## STOP conditions

If the constructor already resolves the path some other way, or a config
lookup exists for it, STOP and describe what you found in a PR comment
instead of changing the approach.
