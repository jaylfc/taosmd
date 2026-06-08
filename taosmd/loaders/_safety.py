"""Opt-in safety guards shared by every loader.

Two small, stdlib-only helpers that loaders call from their ``load()``
entry point before touching a file:

  * ``check_size``: refuse files larger than ``max_bytes`` so a stray
    multi-gigabyte file can't blow up memory on a ``f.read()``. The cap
    is generous by default (100 MB) and configurable per call.

  * ``resolve_within``: when a caller pins a ``base_dir``, refuse paths
    that resolve outside it (``../../etc/passwd``, an absolute escape, a
    symlink pointing out of the tree). When ``base_dir`` is ``None``
    (the default) nothing is restricted and the resolved path is
    returned unchanged, so standalone use against an arbitrary path is
    never broken.

Both are opt-in: a loader called the old way (no ``base_dir``, default
cap) behaves exactly as before for any real file.
"""

from __future__ import annotations

import os
from pathlib import Path

# Generous default: large enough that no normal chat / transcript /
# email / doc file trips it, small enough to stop a runaway read.
DEFAULT_MAX_BYTES = 100 * 1024 * 1024  # 100 MB


def check_size(path: str | Path, max_bytes: int | None = DEFAULT_MAX_BYTES) -> None:
    """Raise ``ValueError`` if ``path`` is larger than ``max_bytes``.

    ``max_bytes=None`` disables the check entirely (explicit opt-out).
    Anything else is compared against ``os.path.getsize``. The error
    names the path and both sizes so the caller can see the overage.
    """
    if max_bytes is None:
        return
    size = os.path.getsize(path)
    if size > max_bytes:
        raise ValueError(
            f"{path} is {size} bytes, which exceeds the loader size "
            f"limit of {max_bytes} bytes. Pass a larger max_bytes "
            f"(or max_bytes=None) to override."
        )


def resolve_within(path: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve ``path`` and, when ``base_dir`` is set, confine it there.

    ``base_dir`` is expected to be a directory that contains (directly or
    transitively) the file at ``path``; passing a file as ``base_dir``
    only ever matches that exact file.

    With ``base_dir=None`` (the default) this is just ``Path(path)``
    resolved, no restriction, so direct standalone use of any path is
    unaffected.

    With ``base_dir`` given, the resolved path must sit inside the
    resolved ``base_dir`` or a ``ValueError`` is raised. Resolving first
    means traversal (``../``), absolute escapes, and symlinks that point
    out of the tree are all caught.

    This is a containment check, not an atomic open: there is an inherent
    TOCTOU window between resolving the path here and the caller opening
    it, so a symlink swapped in after this returns is not caught. For the
    local-first, single-user ingest path this guards against accidental
    escapes, not a concurrent adversary on the same machine.
    """
    resolved = Path(path).resolve()
    if base_dir is None:
        return resolved

    base = Path(base_dir).resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError(
            f"{path} resolves to {resolved}, which is outside the "
            f"allowed base path {base}."
        )
    return resolved
