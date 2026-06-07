"""DocLoader — catch-all for plain text + markdown into a ``DocBlob``.

This is the loader the registry picks when nothing more specific
claims a file. It does no parsing beyond extracting an optional title
from the first markdown ``#`` heading.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from ._safety import DEFAULT_MAX_BYTES, check_size, resolve_within
from .blob import DocBlob
from .interface import LoaderInterface


class DocLoader(LoaderInterface):
    loader_name: ClassVar[str] = "doc"
    supported_extensions: ClassVar[tuple[str, ...]] = (
        "txt", "md", "markdown", "rst",
    )
    supported_mime_types: ClassVar[tuple[str, ...]] = (
        "text/plain", "text/markdown",
    )

    async def load(
        self,
        file_path: str | Path,
        *,
        max_bytes: int | None = DEFAULT_MAX_BYTES,
        base_dir: str | Path | None = None,
        **kwargs,
    ) -> DocBlob:
        path = Path(file_path)
        safe_path = resolve_within(file_path, base_dir)
        check_size(safe_path, max_bytes)
        with open(safe_path, encoding="utf-8", errors="replace") as f:
            content = f.read()

        title = ""
        for line in content.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("# "):
                title = stripped[2:].strip()
                break

        return DocBlob(
            source_path=str(path),
            raw_text=content,
            title=title,
            content=content,
        )
