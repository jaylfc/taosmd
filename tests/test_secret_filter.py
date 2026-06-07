"""Tests for taosmd.secret_filter — detection, redaction, and filter modes.

Behaviour-only, fully offline (no models, no network). Covers every existing
pattern category plus the cloud/payment/email providers added in the audit
batch (Stripe, SendGrid, GCP, Azure).
"""

from __future__ import annotations

import pytest

from taosmd.secret_filter import (
    SECRET_PATTERNS,
    contains_secrets,
    filter_text,
    redact_secrets,
)


# Synthetic Stripe-style key bodies, assembled at runtime so no contiguous
# real-looking key literal sits in source (avoids tripping push/secret scanners
# while still exercising the sk_(live|test)_ regex).
_STRIPE_BODY = "0123456789abcdefABCDEFxyz"  # 25 chars, >= the required 24


def _stripe(kind: str) -> str:
    return "sk_" + kind + "_" + _STRIPE_BODY


# Each case: (name, sample_text, expected_redaction_type)
# The expected type is the placeholder fragment that should appear after
# redaction (and the name reported by redact_secrets()).
SECRET_CASES = [
    # --- existing patterns ---
    ("openai", "key is sk-ABCDEFGHIJKLMNOPQRSTUVWX here", "openai_key"),
    ("anthropic", "sk-ant-ABCDEFGHIJKLMNOPQRSTUVWX1234", "anthropic_key"),
    ("github_pat", "ghp_" + "A" * 36, "github_pat"),
    ("github_oauth", "gho_" + "B" * 36, "github_oauth"),
    ("github_app_ghu", "ghu_" + "C" * 36, "github_token"),
    ("github_app_ghs", "ghs_" + "D" * 36, "github_token"),
    ("github_app_ghr", "ghr_" + "E" * 36, "github_token"),
    ("gitlab", "glpat-ABCDEFGHIJKLMNOPQRST", "gitlab_pat"),
    ("aws_access_key", "AKIAIOSFODNN7EXAMPLE", "aws_key"),
    ("aws_secret", "aws_secret_access_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "aws_secret"),
    ("bearer", "Authorization: Bearer abc.def-ghi_jkl+mno/pqr=", "bearer"),
    (
        "jwt",
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
        "jwt",
    ),
    ("npm", "npm_" + "Z" * 36, "npm_token"),
    ("digitalocean", "dop_v1_" + "a" * 64, "do_token"),
    ("slack", "xoxb-0123456789-ABCDEFGHIJ", "slack_token"),
    ("generic_api_key", "api_key=ABCDEFGHIJKLMNOPQRSTUVWX", "api_key"),
    (
        "pem_private_key",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIabc123\n-----END RSA PRIVATE KEY-----",
        "private_key",
    ),
    ("password_field", "password: hunter2pass", "password"),
    ("connection_string", "postgres://user:pw@host:5432/db", "connection_string"),
    ("private_tag", "<private>my secret note</private>", "private"),
    # --- new patterns (audit batch) ---
    ("stripe_live", _stripe("live"), "stripe_key"),
    ("stripe_test", _stripe("test"), "stripe_key"),
    (
        "sendgrid",
        "SG." + "A" * 22 + "." + "B" * 43,
        "sendgrid_key",
    ),
    (
        "gcp_service_account",
        '{"type": "service_account", "project_id": "demo"}',
        "gcp_service_account",
    ),
    (
        "gcp_private_key",
        '{"private_key": "-----BEGIN PRIVATE KEY-----\\nMIIabc"}',
        "gcp_private_key",
    ),
    (
        "azure_account_key",
        "AccountKey=" + "A" * 44 + "==",
        "azure_storage_key",
    ),
    (
        "azure_connection_string",
        "DefaultEndpointsProtocol=https;AccountName=demo;"
        "AccountKey=" + "B" * 44 + "==;EndpointSuffix=core.windows.net",
        "azure_storage_key",
    ),
]


@pytest.mark.parametrize("name, sample, _type", SECRET_CASES)
def test_contains_secrets_detects(name, sample, _type):
    assert contains_secrets(sample) is True, f"{name} not detected"


@pytest.mark.parametrize("name, sample, expected_type", SECRET_CASES)
def test_redact_secrets_redacts(name, sample, expected_type):
    redacted, found = redact_secrets(sample)
    placeholder = f"[REDACTED:{expected_type}]"
    assert placeholder in redacted, f"{name}: {placeholder} missing from {redacted!r}"
    # The redacted output must not contain the raw secret material that the
    # placeholder replaced (sanity check that substitution actually happened).
    assert redacted != sample, f"{name}: text unchanged after redaction"
    assert found, f"{name}: no redaction types reported"


def test_redact_reports_pattern_name():
    _, found = redact_secrets("ghp_" + "A" * 36)
    assert "github_pat" in found


def test_redact_handles_multiple_secrets_at_once():
    text = f"openai sk-ABCDEFGHIJKLMNOPQRSTUVWX and stripe {_stripe('live')} together"
    redacted, found = redact_secrets(text)
    assert "[REDACTED:openai_key]" in redacted
    assert "[REDACTED:stripe_key]" in redacted
    assert "openai_key" in found and "stripe_key" in found


# --- clean text must pass untouched ---

CLEAN_SAMPLES = [
    "This is a perfectly normal sentence about cats and dogs.",
    "The user prefers dark mode and lives in Berlin.",
    "Meeting notes: discussed Q3 roadmap, no blockers.",
    "I bought sketches for the new design (skateboarding theme).",
    "",
]


@pytest.mark.parametrize("sample", CLEAN_SAMPLES)
def test_clean_text_not_flagged(sample):
    assert contains_secrets(sample) is False


@pytest.mark.parametrize("sample", CLEAN_SAMPLES)
def test_clean_text_unchanged_by_redact(sample):
    redacted, found = redact_secrets(sample)
    assert redacted == sample
    assert found == []


# --- filter_text modes ---

def test_filter_text_redact_mode_redacts():
    out = filter_text(f"token {_stripe('live')}", mode="redact")
    assert "[REDACTED:stripe_key]" in out
    assert _STRIPE_BODY not in out


def test_filter_text_default_mode_is_redact():
    out = filter_text(f"token {_stripe('live')}")
    assert "[REDACTED:stripe_key]" in out


def test_filter_text_warn_mode_returns_unchanged():
    text = f"token {_stripe('live')}"
    assert filter_text(text, mode="warn") == text


def test_filter_text_reject_mode_raises_on_secret():
    with pytest.raises(ValueError):
        filter_text(f"token {_stripe('live')}", mode="reject")


def test_filter_text_reject_mode_passes_clean_text():
    text = "nothing secret here at all"
    assert filter_text(text, mode="reject") == text


def test_pattern_table_is_well_formed():
    # Every entry is (name, compiled_pattern, replacement) and names are unique.
    names = [name for name, _, _ in SECRET_PATTERNS]
    assert len(names) == len(set(names)), "duplicate pattern names"
    for name, pattern, replacement in SECRET_PATTERNS:
        assert isinstance(name, str) and name
        assert hasattr(pattern, "search")
        assert replacement.startswith("[REDACTED:")
