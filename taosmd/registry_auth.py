"""Registry-based authentication for the A2A bus (opt-in).

When a taOS agent registry is configured, a sender on the bus must present an
EdDSA-JWT minted by that registry. The token is verified against the registry's
Ed25519 public key, and the policy layer additionally requires that the token's
``sub`` (the agent's canonical_id) matches the message ``from`` and is not in the
registry's revocation list.

With no registry configured the bus keeps its free-handle behaviour, so a
standalone install is unaffected. The crypto stack (``pyjwt`` + ``cryptography``)
is an optional extra; if a registry is configured but the libraries are missing,
verification raises :class:`AuthError` with an actionable message rather than
silently trusting the handle.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request

logger = logging.getLogger(__name__)

# Registry endpoints, relative to the configured base URL (taOS contract).
PUBKEY_PATH = "/api/agents/registry/pubkey"
REVOKED_PATH = "/api/agents/registry/revoked"
GRANTS_PATH = "/api/agents/registry/grants"

# The literal ``iss`` the taOS registry mints into every token (#159). The bus
# pins this so a token from any other issuer is rejected.
REGISTRY_ISS = "taos-registry"


class AuthError(Exception):
    """Raised when a token fails verification or the auth policy."""


def _require_jwt():
    try:
        import jwt  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised via degrade test
        raise AuthError(
            "registry auth requires the crypto extra: pip install taosmd[registry]"
        ) from exc
    return jwt


def decode_and_verify(token: str, public_key: str) -> dict:
    """Verify an EdDSA JWT against an Ed25519 public key and return its claims.

    Raises :class:`AuthError` on a bad signature, malformed token, or expiry.
    """
    jwt = _require_jwt()
    try:
        return jwt.decode(token, public_key, algorithms=["EdDSA"])
    except Exception as exc:  # noqa: BLE001 - normalise to AuthError
        raise AuthError(f"token verification failed: {exc}") from exc


def authorize_sender(token: str, claimed_from: str, *, public_key: str,
                     revoked: set[str], expected_iss: str | None = None) -> dict:
    """Authorise a bus sender. Returns the verified claims or raises AuthError.

    Policy (after the EdDSA signature check):
      * the token must carry a ``sub`` (the agent canonical_id);
      * ``sub`` must equal the message ``from`` (no impersonation);
      * ``sub`` must not be in the registry revocation set;
      * when ``expected_iss`` is set, ``iss`` must match it (issuer pinning).
    """
    claims = decode_and_verify(token, public_key)
    sub = claims.get("sub")
    if not sub:
        raise AuthError("token has no 'sub' (canonical_id) claim")
    if sub != claimed_from:
        raise AuthError(f"token sub {sub!r} does not match from {claimed_from!r}")
    if sub in revoked:
        raise AuthError(f"canonical_id {sub!r} is revoked")
    if expected_iss is not None and claims.get("iss") != expected_iss:
        raise AuthError(f"token iss {claims.get('iss')!r} != expected {expected_iss!r}")
    return claims


class RegistryVerifier:
    """Caching wrapper that authorises bus senders against a taOS registry.

    The Ed25519 public key is fetched once and cached (it rotates rarely). The
    revocation set is fetched and re-fetched on ``refresh_interval`` so revokes
    propagate without a restart, matching the registry poll-and-cache model.

    Revocation failures are handled fail-closed where it matters: if the feed
    has NEVER loaded, :meth:`authorize` raises (we cannot prove an agent is
    unrevoked, so we refuse). Once a known-good set is cached, a later refresh
    failure keeps that set (a transient outage must not silently un-revoke an
    agent, nor take the whole bus down).

    ``pubkey_loader`` and ``revoked_loader`` are injected so the network layer
    can be supplied by the caller (and stubbed in tests). ``clock`` defaults to
    wall-clock ``time.time``; an injected clock makes refresh timing testable.
    """

    def __init__(self, *, pubkey_loader, revoked_loader,
                 refresh_interval: float = 300.0, clock=time.time,
                 expected_iss: str | None = None):
        self._pubkey_loader = pubkey_loader
        self._revoked_loader = revoked_loader
        self._refresh_interval = refresh_interval
        self._clock = clock
        self._expected_iss = expected_iss
        self._pubkey: str | None = None
        self._revoked: set[str] = set()
        self._revoked_fetched_at: float | None = None

    def _get_pubkey(self) -> str:
        if self._pubkey is None:
            self._pubkey = self._pubkey_loader()
        return self._pubkey

    def _get_revoked(self) -> set[str]:
        now = self._clock()
        stale = (self._revoked_fetched_at is None
                 or now - self._revoked_fetched_at >= self._refresh_interval)
        if stale:
            try:
                self._revoked = set(self._revoked_loader())
                self._revoked_fetched_at = now
            except Exception as exc:  # noqa: BLE001
                if self._revoked_fetched_at is None:
                    # Never loaded: we cannot prove an agent is unrevoked, so
                    # fail CLOSED rather than fall through to an empty allowlist.
                    raise AuthError(
                        f"revocation feed unavailable, refusing to authorise: {exc}"
                    ) from exc
                # Already have a known-good set: keep it across a transient
                # refresh failure (fail-safe, never silently un-revokes).
                logger.warning("registry revocation refresh failed, "
                               "using last-good set: %s", exc)
        return self._revoked

    def authorize(self, token: str, claimed_from: str) -> dict:
        """Authorise a sender; return verified claims or raise AuthError."""
        return authorize_sender(
            token, claimed_from,
            public_key=self._get_pubkey(), revoked=self._get_revoked(),
            expected_iss=self._expected_iss,
        )


# --- response parsers --------------------------------------------------------
# The registry's exact JSON shape is not pinned in the contract yet, so accept
# the common ones: a raw PEM body, or a JSON object carrying the key under a
# conventional field. The revoked feed is the agreed [{canonical_id,...}] list
# (also tolerate a {"revoked": [...]} wrapper).

def parse_pubkey_response(body: str) -> str:
    """Extract an Ed25519 public key (PEM) from a registry response body."""
    text = body.strip()
    if text.startswith("-----BEGIN"):
        return body
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AuthError(f"unrecognised pubkey response: {exc}") from exc
    if isinstance(obj, dict):
        for field in ("pubkey", "public_key", "publicKey", "key"):
            val = obj.get(field)
            if isinstance(val, str) and val.strip():
                return val
    raise AuthError("pubkey response had no recognised key field")


def parse_revoked_response(body: str) -> set[str]:
    """Extract the set of revoked canonical_ids from a revoked-feed body."""
    obj = json.loads(body)
    if isinstance(obj, dict):
        obj = obj.get("revoked", [])
    revoked: set[str] = set()
    for rec in obj or []:
        if isinstance(rec, str):
            revoked.add(rec)
        elif isinstance(rec, dict):
            cid = rec.get("canonical_id") or rec.get("sub") or rec.get("id")
            if isinstance(cid, str) and cid:
                revoked.add(cid)
    return revoked


def _http_get(url: str, timeout: float = 5.0, token: str | None = None) -> str:
    """GET ``url`` and return the body text.

    When ``token`` is given it is sent as ``Authorization: Bearer <token>``.
    The pubkey endpoint is public (no token); only the auth-gated revoked feed
    passes one (the #710 contract).
    """
    req = urllib.request.Request(url)  # noqa: S310
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read().decode("utf-8")


def parse_grants_response(body: str) -> list[dict]:
    """Extract grant records from a registry grants-feed response body.

    Returns the raw list of grant dicts; callers filter by expires_at.
    Each record is expected to carry at least ``canonical_id``.
    """
    obj = json.loads(body)
    if isinstance(obj, dict):
        obj = obj.get("grants", [])
    return [r for r in (obj or []) if isinstance(r, dict) and r.get("canonical_id")]


class GrantsVerifier:
    """Caching wrapper that checks whether an agent has an active permission grant.

    Polls ``GET /api/agents/registry/grants`` (admin-only) on interval and
    caches the result. The bus enforcement gate requires *both* a valid JWT
    (RegistryVerifier) *and* an active grant (this class).

    Fail semantics mirror RegistryVerifier: if the feed has NEVER loaded,
    :meth:`has_grant` raises AuthError (fail-closed — cannot prove a grant
    exists). Once a known-good list is cached, a transient refresh failure
    keeps the last-good set rather than silently removing all grants.

    Phase 1 (initial deploy): ``expires_at`` is always null, so every record
    in the feed counts as active. Phase 2 adds real expiries.
    """

    def __init__(self, *, grants_loader, refresh_interval: float = 300.0,
                 clock=time.time):
        self._grants_loader = grants_loader
        self._refresh_interval = refresh_interval
        self._clock = clock
        self._grants: list[dict] = []
        self._fetched_at: float | None = None

    def _get_grants(self) -> list[dict]:
        now = self._clock()
        stale = (self._fetched_at is None
                 or now - self._fetched_at >= self._refresh_interval)
        if stale:
            try:
                self._grants = self._grants_loader()
                self._fetched_at = now
            except Exception as exc:  # noqa: BLE001
                if self._fetched_at is None:
                    raise AuthError(
                        f"grants feed unavailable, refusing to authorise: {exc}"
                    ) from exc
                logger.warning("grants feed refresh failed, using last-good set: %s", exc)
        return self._grants

    def has_grant(self, canonical_id: str, scope: str | None = None) -> bool:
        """Return True if the agent has at least one active (unexpired) grant.

        When ``scope`` is given, at least one grant must match that scope.
        When ``scope`` is ``None``, any active grant for the agent suffices.
        Raises :class:`AuthError` if the feed has never been loaded.
        """
        now = self._clock()
        grants = self._get_grants()
        for g in grants:
            if g.get("canonical_id") != canonical_id:
                continue
            exp = g.get("expires_at")
            if exp is not None and exp < now:
                continue
            if scope is not None and g.get("scope") != scope:
                continue
            return True
        return False


def grants_verifier_from_url(base_url: str, *, refresh_interval: float = 300.0,
                              opener=_http_get, clock=time.time,
                              grants_token: str | None = None) -> "GrantsVerifier":
    """Build a :class:`GrantsVerifier` that fetches from a registry base URL."""
    base = base_url.rstrip("/")
    return GrantsVerifier(
        grants_loader=lambda: parse_grants_response(
            opener(base + GRANTS_PATH, token=grants_token)),
        refresh_interval=refresh_interval, clock=clock,
    )


def verifier_from_url(base_url: str, *, refresh_interval: float = 300.0,
                      opener=_http_get, clock=time.time,
                      expected_iss: str | None = None,
                      revoked_token: str | None = None) -> "RegistryVerifier":
    """Build a :class:`RegistryVerifier` that fetches from a registry base URL.

    The HTTP getter is injectable (``opener``) so callers/tests can supply
    their own transport. Standalone code never calls this (no registry URL).

    ``revoked_token`` is the taOS local/admin token sent as a Bearer header on
    the revoked-feed poll (the #710 contract moved it behind admin auth). The
    pubkey endpoint stays public and is fetched without a token.
    """
    base = base_url.rstrip("/")
    return RegistryVerifier(
        pubkey_loader=lambda: parse_pubkey_response(opener(base + PUBKEY_PATH)),
        revoked_loader=lambda: parse_revoked_response(
            opener(base + REVOKED_PATH, token=revoked_token)),
        refresh_interval=refresh_interval, clock=clock, expected_iss=expected_iss,
    )
