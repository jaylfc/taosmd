# JOB-003: Remove the dead importlib block in _webui_dir

Read docs/agent-jobs/README.md first and follow its absolute rules.

- Branch: `chore/http-server-dead-import` (from `origin/master`)
- Commit message: `chore(http): drop the dead importlib block in _webui_dir`
- PR title: `chore(http): drop the dead importlib block in _webui_dir`
- Allowed files: `taosmd/http_server.py` ONLY.

## The dead code

`taosmd/http_server.py`, function `_webui_dir()` (around line 112): the
try block imports `importlib.resources as _ir`, assigns `_ = ref` to silence
an unused-variable warning, and then unconditionally falls through to the
`__file__`-relative path that does all the real work. The whole try block is
dead, and ruff flags the `_ir` import (F401). The function's BEHAVIOR must
not change: it returns the `__file__`-relative webui path when
`index.html` exists there, else None.

## Steps

1. `git fetch origin && git checkout -b chore/http-server-dead-import origin/master`
2. Read the current function. It looks like this (comments may differ
   slightly; if the STRUCTURE differs from this, STOP per rule 9):

```python
def _webui_dir() -> Path | None:
    try:
        ref = _pkg_files("taosmd").joinpath("webui")
        import importlib.resources as _ir  # noqa: PLC0415
        _ = ref
    except Exception:
        pass
    candidate = Path(__file__).parent / "webui"
    if (candidate / "index.html").exists():
        return candidate
    return None
```

3. Replace the body so only the working path remains, and update the
   docstring to tell the truth:

```python
def _webui_dir() -> Path | None:
    """Return the path to the built webui, or None if absent.

    Resolved relative to this file, which works in both source checkouts
    and wheel installs (the webui ships inside the package directory).
    """
    candidate = Path(__file__).parent / "webui"
    if (candidate / "index.html").exists():
        return candidate
    return None
```

4. Check whether `_pkg_files` (the `from importlib.resources import files
   as _pkg_files` import at the top of the file) is still used anywhere
   else: `grep -n "_pkg_files" taosmd/http_server.py`. If this function was
   its only user, remove that import line too. If it has other users, leave
   it.
5. `python3 -m pytest -q -m "not slow"` (all green; the webui-serving tests
   cover this function), and
   `python3 -m ruff check taosmd/http_server.py` (the F401 finding for
   `_ir` must be gone; do not fix any OTHER findings).
6. One commit, push, open the PR.
