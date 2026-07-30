Role resolver configuration for taOSmd A2A.

Configurable resolver for role-to-identity mapping with short TTL cache.
Exposed via a minimal seam so the taOS binding/rotation/Observatory can inject
or rotate the implementation without modifying core taosmd code.

from __future__ import annotations

import logging
import time
from typing import Protocol

logger = logging.getLogger(__name__)


class RoleResolver(Protocol):
    """Protocol for resolving a role to a canonical identity or None.

    Protocols are duck-typed: objects implementing resolve() are accepted.
    """

    def resolve(self, role: str) -> str | None:
        """Return the current holder of the role, or None if the role does not exist.

        Usage: called for both send-time validation and delivery-time reads.
        Must be fast (O(1)) with a small TTL cache to avoid thundering herd from
        a constantly rotating taOS Observatory.
        """
        ...


class NoopRoleResolver:
    """Resolver that rejects all role sends.

    Exposed as the default when no resolver is configured.
    """

    def resolve(self, role: str) -> str | None:
        raise RuntimeError("no resolver configured: cannot resolve role %r", role)


class ConfigRoleResolver:
    """Role resolver backed by static config with short TTL cache.

    Format: a dict mapping role strings to identity strings (e.g.,
    {"@taOS-PA": "bob_123"}).

    TTL ensures rapid queries can read the latest mapping while avoiding
    excessive registry calls for role resolution at send time (which is rare).
    """

    def __init__(self, role_map: dict[str, str] | None = None, ttl_seconds: float = 30.0):
        self._role_map = role_map or {}
        self._cache: dict[str, tuple[str | None, float]] = {}
        self._ttl = ttl_seconds

    def resolve(self, role: str) -> str | None:
        cached = self._cache.get(role)
        if cached is not None:
            value, expires = cached
            if time.time() < expires:
                return value
        value = self._role_map.get(role)
        self._cache[role] = (value, time.time() + self._ttl)
        if value is None and role in self._role_map:
            # Explicitly mapped to None: role exists but has no holder.
            logger.debug("role %r exists in config but has no holder", role)
        return value


class NullSafeRoleResolver:
    """Wrapper that converts exceptions from the resolver into 503s.

    FAIL-LOUD: configured but unreachable -> 503 on role sends and role-filtered reads.
    No resolver configured -> role sends are 400 (enforced by the caller, not here).
    Prevents silent drop or fallback.

    This is meant to wrap a configured resolver that might crash or raise due to
    network issues (e.g., a registry verifier). If the underlying resolver raises
    any exception, we convert that to a ResolutionError for the caller.
    """

    def __init__(self, resolver: RoleResolver):
        self._resolver = resolver

    def resolve(self, role: str) -> str | None:
        try:
            return self._resolver.resolve(role)
        except Exception as exc:
            logger.exception("role resolution failed for %r", role)
            raise ResolutionError(f"role resolver error: {exc}") from exc


class ResolutionError(RuntimeError):
    """Catch-all for resolver failures (used as error type on 503 responses)."""


def make_resolver(configured: dict[str, str] | None) -> RoleResolver | None:
    """Factory: return a RoleResolver when configured, or None.

    When configured (not None), returns a NullSafeRoleResolver wrapping a
    ConfigRoleResolver backed by the provided map.

    When unconfigured (None), returns None. Callers differentiate via ``if resolver is not None``:
    * role sends -> 400 (bad request)
    * role-filtered reads -> 400 (bad request)
    """
    if not configured:
        return None
    core = ConfigRoleResolver(configured)
    return NullSafeRoleResolver(core)


def get_resolver_from_config(data_dir: str | Path) -> RoleResolver | None:
    """Load the configured role map from data_dir/config.json.

    Key: ``a2a_resolvers.role_map`` (dict).

    If the key is absent or empty, return None (disables resolution).
    Otherwise, return a resolver instance.
    """
    import json
    from pathlib import Path

    cfg_path = Path(data_dir) / "config.json"
    if not cfg_path.exists():
        return None
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    configured = cfg.get("a2a_resolvers", {}).get("role_map")
    if not isinstance(configured, dict) or not configured:
        return None
    return make_resolver(configured)


__all__ = [
    "RoleResolver",
    "ResolutionError",
    "NoopRoleResolver",
    "ConfigRoleResolver",
    "NullSafeRoleResolver",
    "make_resolver",
    "get_resolver_from_config",
]