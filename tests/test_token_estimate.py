"""Tests for the content-aware token estimate heuristic.

These are pure-function, offline checks of estimate_tokens / truncate_to_tokens.
No models, no tokenizer dependency.
"""

from taosmd.context_assembler import (
    CHARS_PER_TOKEN,
    estimate_tokens,
    truncate_to_tokens,
)


def test_empty_string_is_zero_tokens():
    assert estimate_tokens("") == 0


def test_prose_matches_classic_estimate():
    """Plain prose should stay at roughly the old ~4 chars/token rate."""
    prose = "the quick brown fox jumps over the lazy dog every single morning"
    assert estimate_tokens(prose) == len(prose) // CHARS_PER_TOKEN


def test_cjk_counts_higher_than_flat_quarter():
    """CJK glyphs are ~1 token each, far more than a flat len//4 would give."""
    cjk = "这是一个测试" * 3  # 18 CJK glyphs
    flat = len(cjk) // CHARS_PER_TOKEN
    est = estimate_tokens(cjk)
    assert est > flat
    # Each glyph counts as ~1 token.
    assert est == len(cjk)


def test_mixed_cjk_and_ascii():
    """Mixed text counts CJK per-glyph plus the ASCII part by chars/token."""
    text = "user 名前 is 太郎"
    est = estimate_tokens(text)
    # 4 CJK glyphs counted individually, so the estimate must exceed the
    # number of CJK glyphs alone.
    assert est >= 4
    assert est > len(text) // CHARS_PER_TOKEN


def test_dense_json_denser_than_prose_rate():
    """Punctuation-heavy JSON should estimate more tokens than a flat len//4."""
    code = '{"a":1,"b":[2,3],"c":{"d":true}}'
    assert estimate_tokens(code) > len(code) // CHARS_PER_TOKEN


def test_truncate_keeps_short_text_unchanged():
    text = "short enough"
    assert truncate_to_tokens(text, 100) == text


def test_truncate_long_prose_fits_budget():
    text = "word " * 200
    out = truncate_to_tokens(text, 10)
    assert out.endswith("...")
    assert len(out) < len(text)
    # Result lands near the budget (allow the trailing-ellipsis overhead).
    assert estimate_tokens(out) <= 12


def test_truncate_cjk_respects_budget():
    cjk = "测试" * 100
    out = truncate_to_tokens(cjk, 5)
    assert out.endswith("...")
    # ~5 glyphs plus the ellipsis overhead, nowhere near the full 200 glyphs.
    assert estimate_tokens(out) <= 7
