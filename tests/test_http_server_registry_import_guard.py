"""Startup import guard for registry auth deps.

When ``registry_url`` is configured but the optional crypto stack
(``pyjwt`` + ``cryptography``) is missing, the server must fail at startup
with an actionable error naming ``taosmd[registry]`` -- not start silently
and blow up on the first auth-gated request.
"""
from __future__ import annotations

import sys

import pytest

from taosmd import config, http_server


@pytest.fixture
def data_dir_with_registry(tmp_path, monkeypatch):
    monkeypatch.delenv("TAOSMD_REGISTRY_URL", raising=False)
    data_dir = tmp_path / "taosmd-data"
    data_dir.mkdir()
    config.set_registry_url("http://reg.test", data_dir=str(data_dir))
    return str(data_dir)


def test_missing_jwt_fails_at_startup(data_dir_with_registry, monkeypatch):
    monkeypatch.setitem(sys.modules, "jwt", None)
    with pytest.raises(Exception, match=r"taosmd\[registry\]"):
        http_server.make_server("127.0.0.1", 0, data_dir=data_dir_with_registry)


def test_missing_cryptography_fails_at_startup(data_dir_with_registry, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "cryptography.hazmat.primitives.asymmetric.ed25519",
        None,
    )
    with pytest.raises(Exception, match=r"taosmd\[registry\]"):
        http_server.make_server("127.0.0.1", 0, data_dir=data_dir_with_registry)


def test_server_starts_when_deps_present(data_dir_with_registry):
    httpd = http_server.make_server("127.0.0.1", 0, data_dir=data_dir_with_registry)
    try:
        assert httpd.server_address[:2]
    finally:
        httpd.service_loop.close()
        httpd.server_close()
