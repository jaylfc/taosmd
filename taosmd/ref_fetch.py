"""Resolve and fetch taOS Files-backed refs with hash verification.

A ref uri of the form ``taos://<project-slug>/files/<path>`` is resolved to
``GET /api/projects/{slug}/files/{path}`` on the configured controller. The
fetch helper verifies the returned bytes against the ref's ``sha256`` and
returns the verified bytes or a typed error.

No new server storage is introduced: this is purely a client-side helper.
"""

from __future__ import annotations

import hashlib
import os
import urllib.parse


class RefFetchError(Exception):
    """Base error for ref fetch failures."""


class HashMismatchError(RefFetchError):
    """Raised when the fetched bytes do not match the ref's sha256."""


class NotFoundError(RefFetchError):
    """Raised when the Files API reports the resource is missing."""


class UnauthorizedError(RefFetchError):
    """Raised when the Files API reports an auth failure."""


_SCHEME_PREFIX = "taos://"
_FILES_SEGMENT = "files/"


def resolve_ref_uri(ref: dict, files_url: str) -> str:
    """Map a taos:// ref uri to the concrete Files API fetch endpoint.

    Args:
        ref: A ref dict with at least a ``uri`` field.
        files_url: Base URL of the taOS controller (same config family as
            ``registry_url``).

    Returns:
        The full URL for ``GET /api/projects/{slug}/files/{path}``.

    Raises:
        ValueError: If the uri scheme is not ``taos://`` or the shape is invalid.
    """
    uri = ref.get("uri", "")
    if not isinstance(uri, str) or not uri.startswith(_SCHEME_PREFIX):
        raise ValueError(
            f"unsupported uri scheme: {uri!r} (only taos:// is accepted)"
        )
    rest = uri[len(_SCHEME_PREFIX):]
    parts = rest.split("/", 1)
    if len(parts) != 2 or not parts[1].startswith(_FILES_SEGMENT):
        raise ValueError(
            f"invalid taos ref uri: {uri!r} (expected taos://<slug>/files/<path>)"
        )
    slug = urllib.parse.quote(parts[0], safe="")
    path = parts[1][len(_FILES_SEGMENT):]
    if not path:
        raise ValueError(
            f"invalid taos ref uri: {uri!r} (path is empty)"
        )
    encoded_path = urllib.parse.quote(path, safe="/")
    base = files_url.rstrip("/")
    return f"{base}/api/projects/{slug}/files/{encoded_path}"


async def fetch_by_ref(ref: dict, fetcher, agent: str) -> bytes:
    """Fetch bytes for a ref using an injected fetcher and verify the hash.

    Args:
        ref: A ref dict with ``uri`` and ``sha256`` fields.
        fetcher: A callable ``fetcher(url: str, agent: str) -> bytes`` that
            performs the HTTP GET and returns the raw response body. It should
            raise :class:`NotFoundError` or :class:`UnauthorizedError` for
            those HTTP status codes.
        agent: The agent identity (passed to ``fetcher`` for auth context).

    Returns:
        The verified raw bytes.

    Raises:
        ValueError: If the uri cannot be resolved.
        HashMismatchError: If the fetched bytes' sha256 does not match ref.sha256.
        NotFoundError: If the fetcher reports the resource is missing.
        UnauthorizedError: If the fetcher reports an auth failure.
        RefFetchError: For other fetch failures.
    """
    import asyncio

    files_url = _get_files_url()
    url = resolve_ref_uri(ref, files_url)
    loop = asyncio.get_running_loop()
    raw = await loop.run_in_executor(None, fetcher, url, agent)
    expected = ref.get("sha256")
    if not expected:
        raise RefFetchError("ref has no sha256")
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise HashMismatchError(
            f"sha256 mismatch: expected {expected}, got {actual}"
        )
    return raw


def _get_files_url() -> str:
    """Resolve the files base URL from env or config.

    Falls back to ``registry_url`` when ``files_url`` is unset, so a
    single-controller install needs only one setting.
    """
    env = os.environ.get("TAOSMD_FILES_URL")
    if env and env.strip():
        return env.strip()
    try:
        from .config import get_files_url
        url = get_files_url()
        if url:
            return url
    except Exception:
        pass
    try:
        from .config import get_registry_url
        url = get_registry_url()
        if url:
            return url
    except Exception:
        pass
    raise RefFetchError(
        "files_url is not configured: set TAOSMD_FILES_URL or files_url in config.json"
    )


__all__ = [
    "RefFetchError",
    "HashMismatchError",
    "NotFoundError",
    "UnauthorizedError",
    "resolve_ref_uri",
    "fetch_by_ref",
]
