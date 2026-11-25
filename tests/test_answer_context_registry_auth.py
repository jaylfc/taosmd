# ruff: noqa: F811, F401
"""/answer-context honours the registry token exactly like /search: the token's
project overrides the body, and an invalid token is refused."""

from __future__ import annotations

import pytest

pytest.importorskip("jwt")
pytest.importorskip("cryptography")

from taosmd import api as taosmd_api
from taosmd import registry_auth

from tests.test_http_server_registry_auth import _make_token, _post_json, project_server


def _patch_all_embedders():
    for stores in taosmd_api._stores_cache.values():
        async def fake(text, task="search_document"):
            h = hash(text) & 0xFFFFFFFF
            return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

        stores["vector"].embed = fake


def _seed(server):
    """agent-1 has memories in proj-a and in proj-b (written tokenless, body project)."""
    _patch_all_embedders()
    for proj, marker in (("proj-a", "alpha marker"), ("proj-b", "bravo marker")):
        for i, text in enumerate((f"{marker} first", f"{marker} second")):
            s, b = _post_json(server, "/ingest", {
                "text": text, "agent": "agent-1", "project": proj, "role": "user",
                "session_id": f"sess-{proj}", "turn_index": i,
                "session_date": "2023/05/20 (Sat) 02:21"})
            assert s == 200, b


def _ask(server, token=None, **extra):
    body = {"query": "marker", "agent": "agent-1", "budget_tokens": 50_000,
            "now": "2023/06/10 (Sat) 10:00", **extra}
    return _post_json(server, "/answer-context", body, token=token)


def test_token_project_overrides_the_body_project(project_server):
    _seed(project_server)
    token = _make_token("agent-1", project_id="proj-a", iss=registry_auth.REGISTRY_ISS)
    status, out = _ask(project_server, token=token, project="proj-b")  # body asks for proj-b
    assert status == 200, out
    assert "alpha marker" in out["context"]
    assert "bravo marker" not in out["context"]
    # without a token the body project is honoured, as for /search
    status, out = _ask(project_server, project="proj-b")
    assert status == 200
    assert "bravo marker" in out["context"] and "alpha marker" not in out["context"]


def test_invalid_token_gets_what_search_gets(project_server):
    from tests.test_http_server_registry_auth import _keypair
    import jwt as pyjwt

    bad_priv, _ = _keypair()
    token = pyjwt.encode({"sub": "agent-1", "iss": registry_auth.REGISTRY_ISS,
                          "project_id": "proj-a"}, bad_priv, algorithm="EdDSA")
    s_search, _ = _post_json(project_server, "/search", {"query": "x", "agent": "agent-1"}, token=token)
    s_ctx, _ = _ask(project_server, token=token)
    assert s_search == s_ctx == 403


def test_token_without_a_grant_for_its_project_is_refused(project_server):
    token = _make_token("agent-1", project_id="proj-b", iss=registry_auth.REGISTRY_ISS)
    s_search, _ = _post_json(project_server, "/search", {"query": "x", "agent": "agent-1"}, token=token)
    s_ctx, _ = _ask(project_server, token=token)
    assert s_search == s_ctx == 403
