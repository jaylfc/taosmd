"""LoaderInterface: the ABC every concrete loader implements.

Lifted from cognee's LoaderInterface (cognee/infrastructure/loaders/
LoaderInterface.py): each loader declares which extensions and MIME
types it can handle, the registry picks one for a given path, and the
loader's ``load()`` returns a typed ``Blob``.

The signature differs slightly from cognee's: cognee returns ``str``,
we return a ``Blob`` subclass so downstream consumers get the typed
fields. The cost is each loader picks its blob type; the win is no
second parsing pass downstream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from ._safety import DEFAULT_MAX_BYTES
from .blob import Blob


class LoaderInterface(ABC):
    """Cognee-style ABC for typed file ingestors.

    Subclasses set the three ClassVars and implement ``load``. The
    registry uses ``can_handle`` to pick a loader for a given path.
    """

    loader_name: ClassVar[str] = ""
    supported_extensions: ClassVar[tuple[str, ...]] = ()
    supported_mime_types: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def can_handle(cls, extension: str = "", mime_type: str = "") -> bool:
        """Return True when this loader claims a file with the given
        extension and / or MIME type. Extension comparison is case-
        insensitive and the leading dot is optional. Either argument can
        be empty; the registry passes whichever it could determine.
        """
        ext = extension.lower().lstrip(".") if extension else ""
        mt = mime_type.lower() if mime_type else ""
        if ext and any(ext == s.lower().lstrip(".") for s in cls.supported_extensions):
            return True
        if mt and any(mt == s.lower() for s in cls.supported_mime_types):
            return True
        return False

    @abstractmethod
    async def load(
        self,
        file_path: str | Path,
        *,
        max_bytes: int | None = DEFAULT_MAX_BYTES,
        base_dir: str | Path | None = None,
        **kwargs,
    ) -> Blob:
        """Read the file at ``file_path`` and return a typed ``Blob``.

        Implementations should set ``Blob.source_path`` to a string of
        ``file_path`` and populate ``raw_text`` opportunistically; when
        a string view is cheap to derive, it lets legacy ingest paths
        keep working. When it isn't cheap, leave it empty and rely on
        the typed fields.

        Two opt-in safety guards, both with standalone-safe defaults:

          ``max_bytes``: the file is rejected (``ValueError``) if it is
          larger than this. Defaults to a generous 100 MB; pass ``None``
          to disable the check.

          ``base_dir``: when given, the resolved ``file_path`` must sit
          inside it or a ``ValueError`` is raised (path-traversal /
          symlink containment). Defaults to ``None`` (no restriction),
          so direct use against any path keeps working.

        Implementations enforce both via ``taosmd.loaders._safety``
        before reading the file.
        """
        raise NotImplementedError
