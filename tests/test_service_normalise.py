"""Tests for taosmd.service._normalise_handle.

The helper is a slug match, not an identity check: it strips any leading
``@``, casefolds, and optionally strips a timestamp mint stamp. Two
distinct agents that share a stem will unify under it.
"""
from __future__ import annotations

from taosmd.service import _normalise_handle


class TestNormaliseHandle:
    def test_strips_leading_at(self):
        assert _normalise_handle("@bob") == "bob"

    def test_casefolds(self):
        assert _normalise_handle("TaOSmd") == "taosmd"

    def test_bare_slug_unchanged(self):
        assert _normalise_handle("bob") == "bob"

    def test_at_plus_casefold(self):
        assert _normalise_handle("@TaOSmd-dev") == "taosmd-dev"

    def test_mint_strip_default_false(self):
        assert _normalise_handle("hermes-20260727-001415") == "hermes-20260727-001415"

    def test_mint_strip_true_strips_date_suffix(self):
        assert _normalise_handle("hermes-20260727-001415", mint_strip=True) == "hermes"

    def test_mint_strip_true_strips_date_only_suffix(self):
        assert _normalise_handle("taOSmd-20260609", mint_strip=True) == "taosmd"

    def test_install_discriminator_survives_mint_strip(self):
        assert _normalise_handle("@taOS-agent-1a2b3c4d", mint_strip=True) == "taos-agent-1a2b3c4d"

    def test_install_discriminator_survives_without_mint_strip(self):
        assert _normalise_handle("@taOS-agent-1a2b3c4d") == "taos-agent-1a2b3c4d"

    def test_slug_match_not_identity_check(self):
        assert _normalise_handle("bob") == _normalise_handle("@BOB")
