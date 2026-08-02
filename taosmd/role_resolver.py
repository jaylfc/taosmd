"""Role-to-holder resolution for A2A recipient addressing (taOS#2155).

taosmd owns the message-side concepts; taOS owns the role *binding*. A
role handle (``@taOS-<name>``, e.g. ``@taOS-PA``) is stored verbatim on the
message envelope and resolved at delivery time, so rotating the holder never
requires reconfiguring senders. The resolver is an injected seam: this module
provides the :class:`RoleResolver` that talks to the (taOS-provided)
resolution endpoint, but the HTTP server accepts any object exposing a
compatible ``resolve(role) -> canonical_id | None`` method.

Fail-loud contract
------------------
* reachable, exactly one holder -> returns the canonical id
* reachable, no single holder   -> returns ``None`` (callers map this to a
  400 at send time, or omit ``resolved_to`` at read time)
* configured but unreachable     -> raises :class:`RoleResolveError`
  (callers map this to 503: never a silent drop, never a guess)

Resolution is NEVER cached at send time -- the stored envelope keeps only the
role handle, so rotation is reflected on the next read. A short TTL cache
bounds repeated lookups within a single resolver instance.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

# A recipient handle is a *role* when it is prefixed with ``@taOS-`` (the taOS
# PA/role namespace), e.g. ``@taOS-PA``. Bare ``@handle`` strings are agent
# handles and need no resolution. This convention is the message-side shim that
# lets the bus recognise a role recipient without consulting the resolver.
ROLE_PREFIX = "@taOS-"

# Path (relative to the resolver base URL) that resolves a role to its holder.
# ``{role}`` is the bare role name with the leading ``@`` stripped.
_RESOLVE_PATH = "/api/roles/{role}/resolve"

# Default cache TTL (seconds) for resolved role -> holder mappings.
_DEFAULT_TTL = 30.0


def is_role_handle(recipient: str | None) -> bool:
    """Return True when ``recipient`` is a role handle (``@taOS-...``)."""
    return isinstance(recipient, str) and recipient.startswith(ROLE_PREFIX)


class RoleResolveError(Exception):
    """The role resolver is configured but could not be reached.

    Raised only for *transport* failures (connection refused, DNS failure,
    non-2xx other than 404). A reachable resolver that finds no holder returns
    ``None`` instead -- that is a valid negative answer, not a failure.
    """


class RoleResolver:
    """Resolve taOS role handles to the canonical id of their current holder.

    Args:
        url: Base URL of the taOS resolution endpoint (e.g. the taOS API).
        token: Optional bearer token sent as ``Authorization``.
        timeout: Per-request timeout in seconds.
        ttl: Cache lifetime in seconds; a resolved role is re-fetched once this
            elapses so holder rotation propagates without a restart.
    """

    def __init__(self, url: str, *, token: str | None = None,
                 timeout: int = 10, ttl: float = _DEFAULT_TTL) -> None:
        self._base = url.rstrip("/")
        self._token = token
        self._timeout = timeout
        self._ttl = ttl
        # role -> (expiry_monotonic, canonical_id | None)
        self._cache: dict[str, tuple[float, str | None]] = {}
        self._lock = threading.Lock()

    def resolve(self, role: str) -> str | None:
        """Return the canonical id of the single holder of ``role``, else None.

        Raises :class:`RoleResolveError` when the endpoint is unreachable.
        A negative answer (no holder / not exactly one) is returned as None.
        """
        if not is_role_handle(role):
            return None
        with self._lock:
            cached = self._cache.get(role)
            if cached is not None and cached[0] > time.monotonic():
                return cached[1]
        result = self._fetch(role)
        with self._lock:
            self._cache[role] = (time.monotonic() + self._ttl, result)
        return result

    def _fetch(self, role: str) -> str | None:
        bare = role.lstrip("@")
        url = self._base + _RESOLVE_PATH.format(role=bare)
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise RoleResolveError(
                f"role resolver {url} returned HTTP {exc.code}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RoleResolveError(
                f"role resolver {url} unreachable: {exc}"
            ) from exc
        # The endpoint reports the single current holder; absent/empty means
        # the role has no (single) holder right now.
        holder = body.get("holder") if isinstance(body, dict) else None
        if not isinstance(holder, str) or not holder:
            return None
        return holder

    def bust(self, role: str | None = None) -> None:
        """Drop cached entries, forcing a re-fetch on the next ``resolve()``.

        ``None`` clears the whole cache; otherwise just ``role`` is evicted.
        """
        with self._lock:
            if role is None:
                self._cache.clear()
            else:
                self._cache.pop(role, None)


__all__ = [
    "RoleResolver",
    "RoleResolveError",
    "is_role_handle",
    "ROLE_PREFIX",
]
