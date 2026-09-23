"""onnxruntime must stay OUT of taosmd's core dependencies.

Measured on a Nothing Phone (1) running postmarketOS edge (Alpine, musl,
aarch64) on 2026-09-15: `pip install -e ".[proxy]"` for taOS died with

    ERROR: Could not find a version that satisfies the requirement onnxruntime
           (from taosmd) (from versions: none)

"from versions: none" is the signature of a package that publishes neither a
musllinux wheel nor an sdist -- there is nothing pip can install on musl at any
version. While `onnxruntime` sat in `dependencies`, taosmd and therefore every
downstream project (taOS pins `taosmd==0.4.0` as a core dependency) could not be
installed on any musl host at all.

Every `import onnxruntime` in this package is lazy and guarded, so the core
install works without it; only the "onnx" embed mode and the ONNX cross-encoder
are unavailable, and both say so.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _pyproject() -> dict:
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)


def _names(reqs) -> set[str]:
    return {Requirement(r).name.lower() for r in reqs}


def test_onnxruntime_is_not_a_core_dependency() -> None:
    core = _names(_pyproject()["project"]["dependencies"])
    assert "onnxruntime" not in core, (
        "onnxruntime is back in taosmd's core dependencies. It ships no "
        "musllinux wheel and no sdist, so this makes taosmd -- and every "
        "project that depends on it -- uninstallable on Alpine/postmarketOS. "
        "Keep it in the optional 'onnx' extra."
    )


def test_onnxruntime_is_still_reachable_as_an_extra() -> None:
    """Removing it from core must not lose the ability to install it at all."""
    extras = _pyproject()["project"]["optional-dependencies"]
    assert "onnx" in extras, "the 'onnx' extra is gone; onnxruntime is now uninstallable"
    assert "onnxruntime" in _names(extras["onnx"])
