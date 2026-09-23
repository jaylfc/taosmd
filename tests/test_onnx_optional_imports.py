"""Regression test: the taosmd package must stay importable with onnxruntime
genuinely absent.

onnxruntime is the `onnx` optional extra (pyproject `[project.optional-
dependencies]`): it publishes no musllinux wheel and no sdist, so it is
genuinely absent on musl hosts (Alpine, postmarketOS). The real regression
this guards against is a *top-level* ``import onnxruntime`` landing in any
taosmd module -- as opposed to the lazy, guarded imports inside
``taosmd/cross_encoder.py`` and ``taosmd/vector_memory.py`` that the
`onnx = ["onnxruntime"]` extra (see pyproject.toml) exists to keep optional.
A top-level import would make ``import taosmd...`` fail outright on any host
without onnxruntime, which is exactly the Alpine/postmarketOS breakage the
`onnx` extra was created to fix (see the ``onnxruntime-optional-musl``
changelog fragment).

This has to run in a **fresh subprocess**. CI now always installs
onnxruntime (`uv sync --extra onnx`), and once a module has been imported
in-process its entry is cached in ``sys.modules`` -- a later, in-process
"block onnxruntime and re-import" attempt would silently reuse the cached
module and miss a top-level import regression entirely. Blocking
onnxruntime *before anything else is imported*, in a brand-new
interpreter, is the only way to actually exercise the import-time guard.
"""
from __future__ import annotations

import subprocess
import sys

# Runs in a fresh `python -c` subprocess. Blocks onnxruntime before importing
# anything else, then imports every module under the taosmd package
# (pkgutil.walk_packages), and reports which module(s), if any, fail to
# import specifically because of onnxruntime. A module that fails to import
# for an unrelated reason (a different missing optional dependency) is not
# this test's concern and is not counted as a failure here -- only
# onnxruntime-caused failures are.
_PROBE = r"""
import sys
sys.modules["onnxruntime"] = None

import pkgutil
import traceback

import taosmd

onnx_failures = []
other_failures = []
for info in pkgutil.walk_packages(taosmd.__path__, prefix="taosmd."):
    try:
        __import__(info.name)
    except Exception as exc:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        if "onnxruntime" in tb:
            onnx_failures.append((info.name, tb))
        else:
            other_failures.append((info.name, tb))

for name, tb in other_failures:
    print(f"IGNORED (not onnxruntime-related): {name}\n{tb}", file=sys.stderr)

if onnx_failures:
    for name, tb in onnx_failures:
        print(f"ONNXRUNTIME-CAUSED IMPORT FAILURE: {name}", file=sys.stderr)
        print(tb, file=sys.stderr)
    sys.exit(1)

sys.exit(0)
"""


def test_taosmd_package_importable_without_onnxruntime():
    """Every taosmd.* submodule must import cleanly with onnxruntime blocked.

    A regression here means some module does ``import onnxruntime`` (or
    ``from onnxruntime import ...``) at module scope instead of lazily
    inside the function/method that actually needs it, which breaks
    `import taosmd` outright on any host where onnxruntime is not
    installed -- the musl breakage the `onnx` extra exists to prevent.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        "one or more taosmd modules require onnxruntime at import time "
        "(it must be a lazy, guarded import instead -- see "
        "taosmd/cross_encoder.py and taosmd/vector_memory.py for the "
        "existing pattern):\n"
        f"--- subprocess stdout ---\n{proc.stdout}\n"
        f"--- subprocess stderr ---\n{proc.stderr}"
    )
