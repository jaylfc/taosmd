"""Unit tests for GrantsVerifier and parse_grants_response."""
import time
import pytest
from taosmd.registry_auth import (
    AuthError,
    GrantsVerifier,
    parse_grants_response,
    grants_verifier_from_url,
)


# ---------------------------------------------------------------------------
# parse_grants_response
# ---------------------------------------------------------------------------

def test_parse_grants_response_basic():
    body = '{"grants": [{"canonical_id": "agent-a", "scope": "memory"}]}'
    grants = parse_grants_response(body)
    assert len(grants) == 1
    assert grants[0]["canonical_id"] == "agent-a"


def test_parse_grants_response_list_directly():
    body = '[{"canonical_id": "agent-b", "scope": "a2a"}]'
    grants = parse_grants_response(body)
    assert grants[0]["canonical_id"] == "agent-b"


def test_parse_grants_response_empty():
    assert parse_grants_response('{"grants": []}') == []


def test_parse_grants_response_skips_missing_canonical_id():
    body = '{"grants": [{"scope": "memory"}, {"canonical_id": "ok"}]}'
    grants = parse_grants_response(body)
    assert len(grants) == 1
    assert grants[0]["canonical_id"] == "ok"


# ---------------------------------------------------------------------------
# GrantsVerifier.has_grant
# ---------------------------------------------------------------------------

def _make_verifier(grants, clock=None):
    return GrantsVerifier(
        grants_loader=lambda: grants,
        refresh_interval=300.0,
        clock=clock or time.time,
    )


def test_has_grant_returns_true_when_present():
    v = _make_verifier([{"canonical_id": "agent-a", "scope": "memory"}])
    assert v.has_grant("agent-a") is True


def test_has_grant_returns_false_when_absent():
    v = _make_verifier([{"canonical_id": "agent-a", "scope": "memory"}])
    assert v.has_grant("agent-b") is False


def test_has_grant_scope_filter_matches():
    v = _make_verifier([{"canonical_id": "agent-a", "scope": "memory"}])
    assert v.has_grant("agent-a", scope="memory") is True


def test_has_grant_scope_filter_no_match():
    v = _make_verifier([{"canonical_id": "agent-a", "scope": "memory"}])
    assert v.has_grant("agent-a", scope="a2a") is False


def test_has_grant_skips_expired():
    past = time.time() - 1.0
    v = _make_verifier([{"canonical_id": "agent-a", "expires_at": past}])
    assert v.has_grant("agent-a") is False


def test_has_grant_accepts_null_expires_at():
    v = _make_verifier([{"canonical_id": "agent-a", "expires_at": None}])
    assert v.has_grant("agent-a") is True


def test_has_grant_accepts_future_expires_at():
    future = time.time() + 3600.0
    v = _make_verifier([{"canonical_id": "agent-a", "expires_at": future}])
    assert v.has_grant("agent-a") is True


# ---------------------------------------------------------------------------
# Fail semantics
# ---------------------------------------------------------------------------

def test_has_grant_raises_auth_error_when_feed_never_loaded():
    def broken():
        raise ConnectionError("network down")
    v = GrantsVerifier(grants_loader=broken)
    with pytest.raises(AuthError, match="grants feed unavailable"):
        v.has_grant("agent-a")


def test_has_grant_keeps_last_good_on_refresh_failure():
    calls = []

    def loader():
        calls.append(1)
        if len(calls) == 1:
            return [{"canonical_id": "agent-a"}]
        raise ConnectionError("refresh failed")

    now = [0.0]

    def clock():
        return now[0]

    v = GrantsVerifier(grants_loader=loader, refresh_interval=10.0, clock=clock)
    # First load: succeeds, agent-a is granted.
    assert v.has_grant("agent-a") is True
    # Advance past refresh interval so next call tries to refresh.
    now[0] = 20.0
    # Refresh fails; last-good set is kept, so agent-a still has a grant.
    assert v.has_grant("agent-a") is True
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# grants_verifier_from_url (integration smoke)
# ---------------------------------------------------------------------------

def test_grants_verifier_from_url_builds_correctly():
    fetched = []

    def opener(url, timeout=5.0, token=None):
        fetched.append((url, token))
        return '{"grants": [{"canonical_id": "agent-x"}]}'

    v = grants_verifier_from_url(
        "http://registry.local",
        opener=opener,
        grants_token="admin-tok",
    )
    assert v.has_grant("agent-x") is True
    assert fetched[0][0].endswith("/api/agents/registry/grants")
    assert fetched[0][1] == "admin-tok"
