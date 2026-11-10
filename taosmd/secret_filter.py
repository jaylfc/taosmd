"""Secret & Credential Filtering (taOSmd).

Detects and redacts sensitive tokens (API keys, passwords, JWTs, etc.)
before text is stored in any memory layer. Runs on every ingest path:
vector memory, knowledge graph, and archive.

Covers all major cloud providers (AWS, GCP, Azure), payment and email
providers (Stripe, SendGrid), CI/CD tokens, and common secret formats.
"""

from __future__ import annotations

import logging
import re
from typing import NamedTuple

logger = logging.getLogger(__name__)

# Each pattern: (name, regex, replacement)
SECRET_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    ("openai_key", re.compile(r"sk-[A-Za-z0-9]{20,}"), "[REDACTED:openai_key]"),
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9\-]{20,}"), "[REDACTED:anthropic_key]"),
    ("stripe_key", re.compile(r"sk_(?:live|test)_[0-9a-zA-Z]{24,}"), "[REDACTED:stripe_key]"),
    ("sendgrid_key", re.compile(r"SG\.[\w-]{22}\.[\w-]{43}"), "[REDACTED:sendgrid_key]"),
    ("gcp_service_account", re.compile(r'"type"\s*:\s*"service_account"'), "[REDACTED:gcp_service_account]"),
    ("gcp_private_key", re.compile(r'"private_key"\s*:\s*"-----BEGIN PRIVATE KEY-----'), "[REDACTED:gcp_private_key]"),
    ("azure_storage_key", re.compile(r"AccountKey=[A-Za-z0-9+/=]{40,}"), "[REDACTED:azure_storage_key]"),
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


class SecretSpan(NamedTuple):
    start: int
    end: int
    kind: str
    replacement: str


def _find_secret_spans_with_kinds(text: str) -> tuple[list[SecretSpan], list[list[str]]]:
    """Find all secret spans in the ORIGINAL text with contributing kinds.

    Returns (spans, span_kinds) where spans are disjoint, sorted SecretSpan
    objects and span_kinds[i] is the list of pattern names that contributed
    to spans[i], in SECRET_PATTERNS order. Overlapping candidates are merged
    into a single span covering the union of their ranges; the merged span
    keeps the kind and replacement of the candidate that opened it.
    """
    candidates: list[tuple[int, int, int, str, str]] = []
    for index, (name, pattern, replacement) in enumerate(SECRET_PATTERNS):
        for match in pattern.finditer(text):
            if match.end() == match.start():
                continue
            candidates.append((match.start(), match.end(), index, name, replacement))

    candidates.sort(key=lambda c: (c[0], -(c[1] - c[0]), c[2]))

    spans: list[SecretSpan] = []
    span_kinds: list[list[str]] = []
    last_end = 0
    for start, end, _index, name, replacement in candidates:
        if start >= last_end:
            spans.append(SecretSpan(start, end, name, replacement))
            span_kinds.append([name])
            last_end = end
        else:
            # Merge: extend the last span's end and add this kind
            last_span = spans[-1]
            new_end = max(last_span.end, end)
            spans[-1] = SecretSpan(last_span.start, new_end, last_span.kind, last_span.replacement)
            span_kinds[-1].append(name)
            last_end = new_end
    return spans, span_kinds


def find_secret_spans(text: str) -> list[SecretSpan]:
    """Find all secret spans in the ORIGINAL text.

    Returns disjoint, sorted spans (Python str indices over `text`). Every
    (name, pattern, replacement) in SECRET_PATTERNS contributes its
    pattern.finditer(text) whole-match (m.start(), m.end()) offsets; zero-length
    matches are skipped. Overlaps are resolved by sorting candidates by
    (start ascending, length descending, SECRET_PATTERNS index ascending) and
    walking them, merging any candidate whose start is before the current
    span's end into that span (extending the end to the union).
    """
    spans, _ = _find_secret_spans_with_kinds(text)
    return spans


def redact_secrets(text: str) -> tuple[str, list[str]]:
    """Redact all secrets from text.

    Returns (redacted_text, list_of_redaction_types).

    Spans are located on the original text via find_secret_spans and each
    span's replacement is spliced over text[start:end], so a later pattern can
    never match inside an earlier [REDACTED:...] placeholder and offsets always
    refer to the original text. The returned list contains every contributing
    kind (including those from merged spans), ordered by SECRET_PATTERNS.
    """
    spans, span_kinds = _find_secret_spans_with_kinds(text)

    parts: list[str] = []
    prev = 0
    for span in spans:
        parts.append(text[prev:span.start])
        parts.append(span.replacement)
        prev = span.end
    parts.append(text[prev:])
    redacted = "".join(parts)

    # Collect all contributing kinds from all spans, deduplicated, in SECRET_PATTERNS order
    all_kinds: set[str] = set()
    for kinds in span_kinds:
        all_kinds.update(kinds)
    found: list[str] = []
    for name, _, _ in SECRET_PATTERNS:
        if name in all_kinds:
            found.append(name)
    return redacted, found


def contains_secrets(text: str) -> bool:
    """Check if text contains any known secret patterns."""
    for _, pattern, _ in SECRET_PATTERNS:
        if pattern.search(text):
            return True
    return False


def filter_text(text: str, mode: str = "redact") -> str:
    """Filter secrets from text before storage.

    Modes:
        "redact": replace secrets with [REDACTED:type] placeholders
        "reject": raise ValueError if secrets are found
        "warn":   return text unchanged but log a warning
    """
    if mode == "reject":
        if contains_secrets(text):
            raise ValueError("Text contains secrets and cannot be stored")
        return text

    if mode == "warn":
        # warn = don't redact, just log when secrets are present.
        if contains_secrets(text):
            logger.warning("Secrets detected in text (warn mode): stored unredacted")
        return text

    # Default: redact
    redacted, _ = redact_secrets(text)
    return redacted
