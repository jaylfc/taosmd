"""Tests for channel-membership stem measurement."""

from __future__ import annotations

import pytest

from scripts.measure_membership_stems import (
    measure,
    stem_with_mint,
    stem_without_mint,
)


class TestStemFunctions:
    def test_strip_at_and_casefold(self):
        assert stem_without_mint("@taOSmd-dev") == "taosmd-dev"
        assert stem_without_mint("taOSmd-dev") == "taosmd-dev"
        assert stem_without_mint("@TAOSMD-DEV") == "taosmd-dev"

    def test_mint_stamp_stripped(self):
        assert stem_with_mint("taosmd-20260609-153000") == "taosmd"
        assert stem_with_mint("@taOSmd-20260609-153000") == "taosmd"
        assert stem_with_mint("taosmd-dev") == "taosmd-dev"

    def test_install_discriminator_preserved(self):
        assert stem_with_mint("@taOS-agent-abc12345") == "taos-agent-abc12345"
        assert stem_with_mint("taOS-agent-abc12345") == "taos-agent-abc12345"
        assert stem_with_mint("@taOS-agent-abc12345-20260813-192605") == "taos-agent-abc12345"


class TestMeasure:
    def test_empty(self):
        result = measure([])
        assert result["total_principals"] == 0
        assert result["multi_spell_stems_without_mint"] == {}
        assert result["multi_spell_stems_with_mint"] == {}
        assert result["canonical_twins"] == []
        assert result["collapse_without_mint"] == {}

    def test_single_principal(self):
        result = measure([("taosmd-dev", "general")])
        assert result["total_principals"] == 1
        assert result["multi_spell_stems_without_mint"] == {}
        assert result["multi_spell_stems_with_mint"] == {}

    def test_at_and_bare_same_agent(self):
        pairs = [
            ("@taOSmd-dev", "build"),
            ("taosmd-dev", "build"),
        ]
        result = measure(pairs)
        assert result["total_principals"] == 2
        assert len(result["multi_spell_stems_without_mint"]) == 1
        assert "taosmd-dev" in result["multi_spell_stems_without_mint"]
        assert result["multi_spell_stems_without_mint"]["taosmd-dev"] == [
            "@taOSmd-dev",
            "taosmd-dev",
        ]
        assert result["canonical_twins"] == []
        assert len(result["collapse_without_mint"]) == 1

    def test_canonical_with_bare_twin(self):
        pairs = [
            ("taosmd-20260609-153000", "general"),
            ("taosmd", "general"),
        ]
        result = measure(pairs)
        assert len(result["canonical_twins"]) == 1
        canonical, twins = result["canonical_twins"][0]
        assert canonical == "taosmd-20260609-153000"
        assert twins == ["taosmd"]

    def test_canonical_with_at_twin(self):
        pairs = [
            ("@taOS-20260609-153000", "general"),
            ("taOS", "general"),
            ("@taOS", "general"),
        ]
        result = measure(pairs)
        assert len(result["canonical_twins"]) == 1
        canonical, twins = result["canonical_twins"][0]
        assert canonical == "@taOS-20260609-153000"
        assert set(twins) == {"@taOS", "taOS"}

    def test_two_distinct_principals_same_stem(self):
        pairs = [
            ("taosmd-dev", "build"),
            ("taos-dev", "general"),
        ]
        result = measure(pairs)
        assert result["collapse_without_mint"] == {}

    def test_mint_stamp_merges_canonical_and_bare(self):
        pairs = [
            ("taosmd-20260609-153000", "general"),
            ("taosmd-20260813-192605", "general"),
            ("taosmd", "general"),
        ]
        result = measure(pairs)
        assert len(result["multi_spell_stems_with_mint"]) == 1
        assert result["multi_spell_stems_with_mint"]["taosmd"] == [
            "taosmd",
            "taosmd-20260609-153000",
            "taosmd-20260813-192605",
        ]

    def test_install_discriminators_not_merged(self):
        pairs = [
            ("@taOS-agent-abc12345", "general"),
            ("@taOS-agent-def67890", "general"),
        ]
        result = measure(pairs)
        assert result["multi_spell_stems_with_mint"] == {}
        assert result["collapse_without_mint"] == {}
