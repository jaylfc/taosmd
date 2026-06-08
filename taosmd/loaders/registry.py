"""Loader registry: picks the right loader for a path.

Iteration order matters: more-specific loaders go first so that
``ChatLoader`` (which claims ``*.chat.json``) wins over a hypothetical
generic JSON loader. ``DocLoader`` is last as the catch-all for plain
text-like content.

To add a new loader, append it to ``REGISTRY`` via ``register_loader``
or insert at a specific index when ordering matters.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

from .chat_loader import ChatLoader
from .doc_loader import DocLoader
from .email_loader import EmailLoader
from .interface import LoaderInterface
from .transcript_loader import TranscriptLoader


# Order: specific → generic. The first loader whose can_handle returns
# True wins. DocLoader stays last; it's the catch-all.
REGISTRY: list[type[LoaderInterface]] = [
    ChatLoader,
    TranscriptLoader,
    EmailLoader,
    DocLoader,
]


def register_loader(
    loader_cls: type[LoaderInterface],
    *,
    index: int | None = None,
) -> None:
    """Add a loader class to the registry.

    ``index=None`` (default) appends BEFORE the catch-all ``DocLoader``
    so generic loaders don't shadow specific ones. Pass an explicit
    index to override ordering.
    """
    if index is None:
        # Insert before DocLoader (which is last as the catch-all).
        if REGISTRY and REGISTRY[-1] is DocLoader:
            REGISTRY.insert(len(REGISTRY) - 1, loader_cls)
        else:
            REGISTRY.append(loader_cls)
    else:
        REGISTRY.insert(index, loader_cls)


def _path_to_extension(path: str | Path) -> str:
    """Return the lowercased extension, including multi-suffix forms.

    ``meeting.transcript.json`` → ``transcript.json`` (not just ``json``)
    so loaders that claim a multi-suffix shape get a chance.
    """
    p = Path(path)
    name = p.name.lower()
    if "." not in name:
        return ""
    # Try two-suffix form first ("transcript.json"), then one-suffix.
    parts = name.split(".")
    if len(parts) >= 3:
        return ".".join(parts[-2:])
    return parts[-1]


def pick_loader(file_path: str | Path) -> LoaderInterface:
    """Return an instance of the first registered loader that claims
    the given path. Falls back to ``DocLoader`` (the catch-all).
    """
    p = Path(file_path)
    ext = _path_to_extension(p)
    mime_type, _ = mimetypes.guess_type(str(p))
    mime_type = mime_type or ""

    for loader_cls in REGISTRY:
        if loader_cls.can_handle(extension=ext, mime_type=mime_type):
            return loader_cls()
    # Belt-and-braces: REGISTRY ends with DocLoader so this is
    # unreachable, but defend against someone removing the catch-all.
    return DocLoader()
