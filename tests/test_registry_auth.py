"""Tests for registry-based A2A bus authentication (taOS agent registry).

The bus is opt-in authenticated: when a registry public key is configured,
a sender must present an EdDSA-JWT (minted by the taOS registry) whose ``sub``
(canonical_id) matches the message ``from`` and is not revoked. With no
registry configured, the bus keeps its free-handle behaviour (standalone).

These tests inject keys and revoked-sets directly so they never touch the
network. Ed25519 keypairs + JWT signing use ``cryptography`` + ``pyjwt``,
which are the same optional libraries the verifier requires at runtime.
"""
from __future__ import annotations

import pytest

# The verifier requires pyjwt[crypto]; skip the whole module if the optional
# crypto stack is not installed (the runtime degrades with a clear error, but
# these tests need it present to exercise real signatures).
pytest.importorskip("jwt")
pytest.importorskip("cryptography")

import jwt as pyjwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

from taosmd import registry_auth


def _keypair():
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()
    priv_pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    pub_pem = pub.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return priv_pem, pub_pem


def _sign(priv_pem: str, claims: dict) -> str:
    return pyjwt.encode(claims, priv_pem, algorithm="EdDSA")


def test_decode_and_verify_returns_claims_for_valid_signature():
    priv_pem, pub_pem = _keypair()
    token = _sign(priv_pem, {"sub": "hermes-20260608-153000", "iss": "taos"})

    claims = registry_auth.decode_and_verify(token, pub_pem)

    assert claims["sub"] == "hermes-20260608-153000"
    assert claims["iss"] == "taos"


def test_decode_and_verify_rejects_token_signed_by_a_different_key():
    priv_pem, _ = _keypair()
    _, other_pub_pem = _keypair()
    token = _sign(priv_pem, {"sub": "imposter"})

    with pytest.raises(registry_auth.AuthError):
        registry_auth.decode_and_verify(token, other_pub_pem)


def test_decode_and_verify_rejects_garbage_token():
    _, pub_pem = _keypair()
    with pytest.raises(registry_auth.AuthError):
        registry_auth.decode_and_verify("not-a-jwt", pub_pem)


def test_authorize_sender_accepts_matching_sub():
    priv_pem, pub_pem = _keypair()
    token = _sign(priv_pem, {"sub": "agent-20260101-000000"})

    claims = registry_auth.authorize_sender(
        token, "agent-20260101-000000", public_key=pub_pem, revoked=set()
    )

    assert claims["sub"] == "agent-20260101-000000"


def test_authorize_sender_rejects_when_sub_does_not_match_from():
    priv_pem, pub_pem = _keypair()
    token = _sign(priv_pem, {"sub": "agent-A"})

    with pytest.raises(registry_auth.AuthError):
        registry_auth.authorize_sender(
            token, "agent-B", public_key=pub_pem, revoked=set()
        )


def test_authorize_sender_rejects_revoked_canonical_id():
    priv_pem, pub_pem = _keypair()
    token = _sign(priv_pem, {"sub": "agent-revoked"})

    with pytest.raises(registry_auth.AuthError):
        registry_auth.authorize_sender(
            token, "agent-revoked", public_key=pub_pem, revoked={"agent-revoked"}
        )


def test_authorize_sender_enforces_expected_issuer_when_configured():
    priv_pem, pub_pem = _keypair()
    good = _sign(priv_pem, {"sub": "a", "iss": "taos"})
    bad = _sign(priv_pem, {"sub": "a", "iss": "someone-else"})

    assert registry_auth.authorize_sender(
        good, "a", public_key=pub_pem, revoked=set(), expected_iss="taos")["iss"] == "taos"
    with pytest.raises(registry_auth.AuthError):
        registry_auth.authorize_sender(
            bad, "a", public_key=pub_pem, revoked=set(), expected_iss="taos")


def test_authorize_sender_skips_issuer_check_when_not_configured():
    priv_pem, pub_pem = _keypair()
    token = _sign(priv_pem, {"sub": "a", "iss": "whatever"})
    # expected_iss defaults to None -> issuer is not enforced
    assert registry_auth.authorize_sender(
        token, "a", public_key=pub_pem, revoked=set())["sub"] == "a"


def test_authorize_sender_requires_a_sub_claim():
    priv_pem, pub_pem = _keypair()
    token = _sign(priv_pem, {"iss": "taos"})  # no sub

    with pytest.raises(registry_auth.AuthError):
        registry_auth.authorize_sender(
            token, "anything", public_key=pub_pem, revoked=set()
        )


# --- RegistryVerifier: caching + polling layer over the pure functions -------


class _FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def test_verifier_authorizes_with_fetched_pubkey_and_revoked():
    priv_pem, pub_pem = _keypair()
    token = _sign(priv_pem, {"sub": "agent-1"})
    v = registry_auth.RegistryVerifier(
        pubkey_loader=lambda: pub_pem,
        revoked_loader=lambda: set(),
    )

    claims = v.authorize(token, "agent-1")

    assert claims["sub"] == "agent-1"


def test_verifier_enforces_fetched_revocation_set():
    priv_pem, pub_pem = _keypair()
    token = _sign(priv_pem, {"sub": "agent-bad"})
    v = registry_auth.RegistryVerifier(
        pubkey_loader=lambda: pub_pem,
        revoked_loader=lambda: {"agent-bad"},
    )

    with pytest.raises(registry_auth.AuthError):
        v.authorize(token, "agent-bad")


def test_verifier_caches_loaders_within_refresh_window():
    priv_pem, pub_pem = _keypair()
    calls = {"pubkey": 0, "revoked": 0}

    def pubkey_loader():
        calls["pubkey"] += 1
        return pub_pem

    def revoked_loader():
        calls["revoked"] += 1
        return set()

    clock = _FakeClock()
    v = registry_auth.RegistryVerifier(
        pubkey_loader=pubkey_loader, revoked_loader=revoked_loader,
        refresh_interval=300, clock=clock,
    )
    v.authorize(_sign(priv_pem, {"sub": "a"}), "a")
    clock.advance(100)  # still within the 300s window
    v.authorize(_sign(priv_pem, {"sub": "a"}), "a")

    # pubkey fetched once (rotates rarely), revoked not re-fetched yet
    assert calls["pubkey"] == 1
    assert calls["revoked"] == 1


def test_verifier_refreshes_revocation_after_interval():
    priv_pem, pub_pem = _keypair()
    revoked_state = {"set": set()}
    calls = {"revoked": 0}

    def revoked_loader():
        calls["revoked"] += 1
        return set(revoked_state["set"])

    clock = _FakeClock()
    v = registry_auth.RegistryVerifier(
        pubkey_loader=lambda: pub_pem, revoked_loader=revoked_loader,
        refresh_interval=300, clock=clock,
    )
    token = _sign(priv_pem, {"sub": "agent-x"})
    v.authorize(token, "agent-x")  # ok, not revoked

    # agent gets revoked; advance past the refresh window
    revoked_state["set"] = {"agent-x"}
    clock.advance(301)

    with pytest.raises(registry_auth.AuthError):
        v.authorize(token, "agent-x")
    assert calls["revoked"] == 2  # re-fetched after the window


def test_verifier_uses_last_good_revocation_when_refresh_fails():
    priv_pem, pub_pem = _keypair()
    state = {"fail": False}

    def revoked_loader():
        if state["fail"]:
            raise OSError("registry unreachable")
        return {"agent-x"}

    clock = _FakeClock()
    v = registry_auth.RegistryVerifier(
        pubkey_loader=lambda: pub_pem, revoked_loader=revoked_loader,
        refresh_interval=300, clock=clock,
    )
    token = _sign(priv_pem, {"sub": "agent-x"})
    with pytest.raises(registry_auth.AuthError):
        v.authorize(token, "agent-x")  # primes cache: agent-x revoked

    # registry goes down; the cached revocation must still hold (fail-safe)
    state["fail"] = True
    clock.advance(301)
    with pytest.raises(registry_auth.AuthError):
        v.authorize(token, "agent-x")


def test_verifier_fails_closed_when_revocation_never_loaded():
    # If the revocation feed has NEVER loaded, we cannot prove an agent is not
    # revoked, so authorize must reject (fail-closed) even for a valid signature.
    priv_pem, pub_pem = _keypair()

    def always_fails():
        raise OSError("registry unreachable at startup")

    v = registry_auth.RegistryVerifier(
        pubkey_loader=lambda: pub_pem, revoked_loader=always_fails,
    )
    token = _sign(priv_pem, {"sub": "agent-1"})
    with pytest.raises(registry_auth.AuthError):
        v.authorize(token, "agent-1")


# --- response parsers (tolerant of the registry's exact JSON shape) ----------

_PEM = ("-----BEGIN PUBLIC KEY-----\nMCowBQYDK2VwAyEA\n"
        "-----END PUBLIC KEY-----\n")


def test_parse_pubkey_accepts_raw_pem_body():
    assert registry_auth.parse_pubkey_response(_PEM) == _PEM


def test_parse_pubkey_accepts_json_pubkey_field():
    import json
    body = json.dumps({"pubkey": _PEM})
    assert registry_auth.parse_pubkey_response(body) == _PEM


def test_parse_pubkey_accepts_json_public_key_field():
    import json
    body = json.dumps({"public_key": _PEM})
    assert registry_auth.parse_pubkey_response(body) == _PEM


def test_parse_revoked_reads_canonical_ids_from_list_of_records():
    import json
    body = json.dumps([
        {"canonical_id": "a-1", "revoked_at": 123.0},
        {"canonical_id": "b-2", "revoked_at": 456.0},
    ])
    assert registry_auth.parse_revoked_response(body) == {"a-1", "b-2"}


def test_parse_revoked_accepts_wrapped_object():
    import json
    body = json.dumps({"revoked": [{"canonical_id": "c-3", "revoked_at": 1}]})
    assert registry_auth.parse_revoked_response(body) == {"c-3"}


def test_parse_revoked_empty_is_empty_set():
    assert registry_auth.parse_revoked_response("[]") == set()


def test_verifier_from_url_fetches_pubkey_and_revoked_from_contract_paths():
    import json
    priv_pem, pub_pem = _keypair()
    seen = []

    def fake_opener(url, token=None):
        seen.append(url)
        if url.endswith(registry_auth.PUBKEY_PATH):
            return json.dumps({"pubkey": pub_pem})
        if url.endswith(registry_auth.REVOKED_PATH):
            return json.dumps([{"canonical_id": "agent-gone", "revoked_at": 1}])
        raise AssertionError(f"unexpected url {url}")

    v = registry_auth.verifier_from_url("http://taos:8000/", opener=fake_opener)

    # a live agent authorises
    assert v.authorize(_sign(priv_pem, {"sub": "agent-ok"}), "agent-ok")["sub"] == "agent-ok"
    # a revoked agent is rejected
    with pytest.raises(registry_auth.AuthError):
        v.authorize(_sign(priv_pem, {"sub": "agent-gone"}), "agent-gone")

    assert "http://taos:8000" + registry_auth.PUBKEY_PATH in seen
    assert "http://taos:8000" + registry_auth.REVOKED_PATH in seen


# --- #710 contract: revoked feed is auth-gated, pubkey feed is public --------


def test_verifier_from_url_sends_token_on_revoked_feed_only():
    # Per #710 the revoked feed requires Authorization: Bearer <token>; the
    # pubkey endpoint is public and must be fetched WITHOUT the token.
    import json
    priv_pem, pub_pem = _keypair()
    calls = []

    def fake_opener(url, token=None):
        calls.append((url, token))
        if url.endswith(registry_auth.PUBKEY_PATH):
            return json.dumps({"public_key": pub_pem})
        if url.endswith(registry_auth.REVOKED_PATH):
            return json.dumps({"revoked": []})
        raise AssertionError(f"unexpected url {url}")

    v = registry_auth.verifier_from_url(
        "http://taos:8000", opener=fake_opener, revoked_token="admin-token")
    v.authorize(_sign(priv_pem, {"sub": "a"}), "a")

    pubkey_tokens = [t for u, t in calls if u.endswith(registry_auth.PUBKEY_PATH)]
    revoked_tokens = [t for u, t in calls if u.endswith(registry_auth.REVOKED_PATH)]
    assert pubkey_tokens == [None]            # public: no token
    assert revoked_tokens == ["admin-token"]  # auth-gated: token sent


def test_http_get_sets_authorization_header_when_token_given(monkeypatch):
    captured = {}

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"body-bytes"

    def fake_urlopen(req, timeout=None):
        captured["auth"] = req.get_header("Authorization")
        return _FakeResp()

    monkeypatch.setattr(registry_auth.urllib.request, "urlopen", fake_urlopen)
    out = registry_auth._http_get("http://reg/x", token="secret-tok")
    assert out == "body-bytes"
    assert captured["auth"] == "Bearer secret-tok"


def test_http_get_omits_authorization_header_without_token(monkeypatch):
    captured = {}

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"ok"

    def fake_urlopen(req, timeout=None):
        captured["auth"] = req.get_header("Authorization")
        return _FakeResp()

    monkeypatch.setattr(registry_auth.urllib.request, "urlopen", fake_urlopen)
    registry_auth._http_get("http://reg/x")
    assert captured["auth"] is None
