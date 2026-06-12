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


# ---------------------------------------------------------------------------
# has_grant project_id matching (taOS#744 contract)
# ---------------------------------------------------------------------------

def test_has_grant_project_exact_match():
    """A grant row with matching project_id passes when project_id param matches."""
    v = _make_verifier([{"canonical_id": "agent-a", "project_id": "proj-x"}])
    assert v.has_grant("agent-a", project_id="proj-x") is True


def test_has_grant_project_mismatch_fails():
    """A grant row with a different project_id is not a match."""
    v = _make_verifier([{"canonical_id": "agent-a", "project_id": "proj-x"}])
    assert v.has_grant("agent-a", project_id="proj-y") is False


def test_has_grant_global_row_matches_any_project():
    """A grant row with no project_id (global grant) matches any project_id param."""
    v = _make_verifier([{"canonical_id": "agent-a"}])
    assert v.has_grant("agent-a", project_id="proj-anything") is True


def test_has_grant_project_param_none_ignores_project():
    """When project_id param is None, the project dimension is not checked."""
    v = _make_verifier([{"canonical_id": "agent-a", "project_id": "proj-x"}])
    # param is None -> skip project check entirely; grant matches on canonical_id alone
    assert v.has_grant("agent-a", project_id=None) is True


def test_has_grant_project_param_none_also_passes_global_row():
    """project_id=None also matches global (no project_id) rows."""
    v = _make_verifier([{"canonical_id": "agent-a"}])
    assert v.has_grant("agent-a", project_id=None) is True


def test_has_grant_multiple_rows_one_matching_project():
    """Only the row with the correct project_id satisfies the constraint."""
    v = _make_verifier([
        {"canonical_id": "agent-a", "project_id": "proj-x"},
        {"canonical_id": "agent-a", "project_id": "proj-y"},
    ])
    assert v.has_grant("agent-a", project_id="proj-y") is True
    assert v.has_grant("agent-a", project_id="proj-z") is False


def test_parse_grants_response_preserves_project_id():
    """parse_grants_response keeps the project_id field on each row."""
    body = '{"grants": [{"canonical_id": "agent-a", "project_id": "proj-x", "scope": "memory"}]}'
    grants = parse_grants_response(body)
    assert grants[0]["project_id"] == "proj-x"


def test_parse_grants_response_tolerates_missing_project_id():
    """Rows without project_id are still returned (they become global grants)."""
    body = '[{"canonical_id": "agent-b"}]'
    grants = parse_grants_response(body)
    assert grants[0].get("project_id") is None


def test_has_grant_parses_iso_expires_at_past():
    """ISO-8601 expiry strings (the registry's actual feed format) parse and
    expire correctly instead of being dropped as unparseable."""
    from taosmd.registry_auth import GrantsVerifier
    grants = [{"canonical_id": "a", "expires_at": "2020-01-01T00:00:00+00:00"}]
    gv = GrantsVerifier(grants_loader=lambda: grants)
    assert gv.has_grant("a") is False


def test_has_grant_parses_iso_expires_at_future():
    from taosmd.registry_auth import GrantsVerifier
    grants = [{"canonical_id": "a", "expires_at": "2099-01-01T00:00:00Z"}]
    gv = GrantsVerifier(grants_loader=lambda: grants)
    assert gv.has_grant("a") is True


def test_parse_expires_at_naive_iso_read_as_utc():
    from taosmd.registry_auth import _parse_expires_at
    assert _parse_expires_at("2020-01-01T00:00:00") == 1577836800.0
    assert _parse_expires_at(1234.5) == 1234.5
    assert _parse_expires_at("not a date") is None
