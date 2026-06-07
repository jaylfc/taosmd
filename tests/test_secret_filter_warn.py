"""Tests for secret_filter.filter_text warn/redact modes."""

from __future__ import annotations

import logging

from taosmd.secret_filter import filter_text


SECRET = "my key is sk-abcdefghijklmnopqrstuvwxyz0123456789"


def test_warn_mode_logs_and_does_not_alter(caplog):
    """warn mode logs a warning when secrets are present but returns text unchanged."""
    with caplog.at_level(logging.WARNING, logger="taosmd.secret_filter"):
        out = filter_text(SECRET, mode="warn")
    assert out == SECRET  # unredacted
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_warn_mode_no_log_when_clean(caplog):
    """warn mode stays silent when there are no secrets."""
    clean = "just an ordinary sentence with no credentials"
    with caplog.at_level(logging.WARNING, logger="taosmd.secret_filter"):
        out = filter_text(clean, mode="warn")
    assert out == clean
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_redact_mode_redacts():
    """redact mode (default) still replaces the secret."""
    out = filter_text(SECRET, mode="redact")
    assert "sk-abcdefghijklmnopqrstuvwxyz0123456789" not in out
    assert "[REDACTED:openai_key]" in out
