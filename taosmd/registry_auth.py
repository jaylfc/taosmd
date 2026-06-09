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
    propagate without a restart, matching the registry poll-and-cache model. If
    a revocation refresh fails, the last known-good set is kept (fail-safe: a
    transient registry outage must not silently un-revoke an agent).

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
            except Exception as exc:  # noqa: BLE001 - keep last-good set
                if self._revoked_fetched_at is None:
                    logger.warning("registry revocation fetch failed, "
                                   "no cached set yet: %s", exc)
                else:
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


def _http_get(url: str, timeout: float = 5.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        return resp.read().decode("utf-8")


def verifier_from_url(base_url: str, *, refresh_interval: float = 300.0,
                      opener=_http_get, clock=time.time,
                      expected_iss: str | None = None) -> "RegistryVerifier":
    """Build a :class:`RegistryVerifier` that fetches from a registry base URL.

    The HTTP getter is injectable (``opener``) so callers/tests can supply
    their own transport. Standalone code never calls this (no registry URL).
    """
    base = base_url.rstrip("/")
    return RegistryVerifier(
        pubkey_loader=lambda: parse_pubkey_response(opener(base + PUBKEY_PATH)),
        revoked_loader=lambda: parse_revoked_response(opener(base + REVOKED_PATH)),
        refresh_interval=refresh_interval, clock=clock, expected_iss=expected_iss,
    )
