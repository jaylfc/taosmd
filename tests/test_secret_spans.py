"""Tests for taosmd.secret_filter.find_secret_spans and redact_secrets splicing.

Behaviour-only, fully offline. find_secret_spans reports where secrets sit in
the ORIGINAL text as disjoint, sorted spans; redact_secrets is rebuilt on top
of it by splicing replacements over those spans.
"""

from __future__ import annotations

import pytest

from taosmd.secret_filter import (
    SECRET_PATTERNS,
    SecretSpan,
    find_secret_spans,
    redact_secrets,
)

from tests.test_secret_filter import SECRET_CASES


def _splice(text: str, spans: list[SecretSpan]) -> str:
    parts: list[str] = []
    prev = 0
    for span in spans:
        parts.append(text[prev : span.start])
        parts.append(span.replacement)
        prev = span.end
    parts.append(text[prev:])
    return "".join(parts)


def _expected_found(spans: list[SecretSpan]) -> list[str]:
    kinds = {span.kind for span in spans}
    return [name for name, _, _ in SECRET_PATTERNS if name in kinds]


def test_spans_index_original():
    text = "a ghp_" + 36 * "A" + " b"
    spans = find_secret_spans(text)
    assert len(spans) == 1
    assert spans[0].start == 2
    assert spans[0].end == len(text) - 2
    assert text[spans[0].start : spans[0].end] == "ghp_" + 36 * "A"
    assert spans[0].kind == "github_pat"
    assert spans[0].replacement == "[REDACTED:github_pat]"


def test_overlap_longest_wins():
    block = (
        "-----BEGIN PRIVATE KEY-----\n"
        "ghp_" + "A" * 36 + "\n"
        "-----END PRIVATE KEY-----"
    )
    spans = find_secret_spans(block)
    assert len(spans) == 1
    assert spans[0].start == 0
    assert spans[0].end == len(block)
    assert spans[0].kind == "private_key_block"
    assert block[spans[0].start : spans[0].end] == block


def test_spans_disjoint_sorted():
    text = (
        "sk-ABCDEFGHIJKLMNOPQRSTUVWX and "
        "ghp_" + "B" * 36 + " and "
        "-----BEGIN PRIVATE KEY-----\nMIIabc\n-----END PRIVATE KEY-----"
    )
    spans = find_secret_spans(text)
    assert len(spans) == 3
    starts = [span.start for span in spans]
    assert starts == sorted(starts)
    for first, second in zip(spans, spans[1:]):
        assert first.end <= second.start


@pytest.mark.parametrize("name, sample, _type", SECRET_CASES)
def test_redact_matches_spans(name, sample, _type):
    redacted, found = redact_secrets(sample)
    spans = find_secret_spans(sample)
    assert redacted == _splice(sample, spans)
    assert found == _expected_found(spans)


def test_no_match_inside_placeholder():
    redacted, found = redact_secrets("[REDACTED:bearer]")
    assert redacted == "[REDACTED:bearer]"
    assert found == []
