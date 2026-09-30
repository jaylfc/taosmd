#!/usr/bin/env python3
"""Test for secret_filter bug fix - Swift argument label should not be treated as credential."""

import pytest
from taosmd.secret_filter import filter_text


def test_swift_argument_label_is_not_treated_as_a_credential():
    """Swift argument label with identifier-colon-identifier chain should NOT be redacted."""
    # This should NOT match because it's a Swift argument label, not a credential assignment
    text = "join(email:password:deviceName:) and leaving..."
    result = filter_text(text, mode="redact")
    assert result == text, f"Expected '{text}' but got '{result}'"


def test_real_credential_is_still_caught():
    """Real credential assignment should still be caught."""
    # This SHOULD match because it's a real credential assignment
    text = "password=hunter2SuperSecret"
    result = filter_text(text, mode="redact")
    assert result != text, "Expected credential to be redacted but it wasn't"
    assert "[REDACTED:password]" in result
