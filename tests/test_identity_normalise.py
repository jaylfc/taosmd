"""Test that identity comparison uses _normalise_handle."""
from __future__ import annotations

import pytest

from taosmd.service import _normalise_handle


class TestNormaliseHandle:
    def test_strips_at_prefix(self):
        assert _normalise_handle("@taos-agent") == "taos-agent"

    def test_case_fold(self):
        assert _normalise_handle("@TaOS-Agent") == "taos-agent"
        assert _normalise_handle("TAOS-AGENT") == "taos-agent"

    def test_same_handle_no_prefix(self):
        assert _normalise_handle("taos-agent") == "taos-agent"

    def test_different_handles_still_differ(self):
        assert _normalise_handle("taos-agent") != _normalise_handle("other-agent")
