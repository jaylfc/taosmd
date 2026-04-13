"""Secret & Credential Filtering (taOSmd).

Detects and redacts sensitive tokens (API keys, passwords, JWTs, etc.)
before text is stored in any memory layer. Runs on every ingest path:
vector memory, knowledge graph, and archive.

14 pattern categories covering all major cloud providers, CI/CD tokens,
and common secret formats.
"""

from __future__ import annotations

import re

# Each pattern: (name, regex, replacement)
SECRET_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    ("openai_key", re.compile(r"sk-[A-Za-z0-9]{20,}"), "[REDACTED:openai_key]"),
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9\-]{20,}"), "[REDACTED:anthropic_key]"),
    ("github_pat", re.compile(r"ghp_[A-Za-z0-9]{36,}"), "[REDACTED:github_pat]"),
    ("github_oauth", re.compile(r"gho_[A-Za-z0-9]{36,}"), "[REDACTED:github_oauth]"),
    ("github_app", re.compile(r"(?:ghu|ghs|ghr)_[A-Za-z0-9]{36,}"), "[REDACTED:github_token]"),
    ("gitlab_pat", re.compile(r"glpat-[A-Za-z0-9\-]{20,}"), "[REDACTED:gitlab_pat]"),
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED:aws_key]"),
    ("aws_secret", re.compile(r"""(?:aws_secret_access_key|AWS_SECRET)\s*[=:]\s*['"]?([A-Za-z0-9/+=]{40})"""), "[REDACTED:aws_secret]"),
    ("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE), "[REDACTED:bearer]"),
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"), "[REDACTED:jwt]"),
    ("npm_token", re.compile(r"npm_[A-Za-z0-9]{36,}"), "[REDACTED:npm_token]"),
    ("digitalocean", re.compile(r"dop_v1_[A-Za-z0-9]{64}"), "[REDACTED:do_token]"),
    ("slack_token", re.compile(r"xox[bpras]-[A-Za-z0-9\-]{10,}"), "[REDACTED:slack_token]"),
    ("generic_api_key", re.compile(r"""(?:api[_-]?key|apikey|api[_-]?secret|api[_-]?token)\s*[=:]\s*['"]?([A-Za-z0-9\-._]{20,})""", re.IGNORECASE), "[REDACTED:api_key]"),
    ("private_key_block", re.compile(r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |DSA )?PRIVATE KEY-----"), "[REDACTED:private_key]"),
    ("password_field", re.compile(r"""(?:password|passwd|pwd)\s*[=:]\s*['"]?(\S{8,})""", re.IGNORECASE), "[REDACTED:password]"),
    ("connection_string", re.compile(r"(?:mongodb|postgres|mysql|redis)://[^\s'\"]+", re.IGNORECASE), "[REDACTED:connection_string]"),
    ("private_tag", re.compile(r"<private>[\s\S]*?</private>"), "[REDACTED:private]"),
]


def contains_secrets(text: str) -> bool:
    """Check if text contains any known secret patterns."""
    for _, pattern, _ in SECRET_PATTERNS:
        if pattern.search(text):
            return True
    return False


def redact_secrets(text: str) -> tuple[str, list[str]]:
    """Redact all secrets from text.

    Returns (redacted_text, list_of_redaction_types).
    """
    redacted = text
    found: list[str] = []
    for name, pattern, replacement in SECRET_PATTERNS:
        if pattern.search(redacted):
            redacted = pattern.sub(replacement, redacted)
            found.append(name)
    return redacted, found


def filter_text(text: str, mode: str = "redact") -> str:
    """Filter secrets from text before storage.

    Modes:
        "redact" — replace secrets with [REDACTED:type] placeholders
        "reject" — raise ValueError if secrets are found
        "warn"   — return text unchanged but log a warning
    """
    if mode == "reject":
        if contains_secrets(text):
            raise ValueError("Text contains secrets and cannot be stored")
        return text

    if mode == "warn":
        # Caller should check contains_secrets() separately for logging
        return text

    # Default: redact
    redacted, _ = redact_secrets(text)
    return redacted
