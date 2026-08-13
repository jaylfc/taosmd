"""Tests for channel-membership stem measurement."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from scripts.measure_membership_stems import (
    _collect_from_archive,
    _collect_from_bus_spool,
    _measure_channel,
    async_main,
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
        assert result["per_channel"] == {}

    def test_single_principal(self):
        result = measure([("taosmd-dev", "general")])
        assert result["total_principals"] == 1
        assert result["multi_spell_stems_without_mint"] == {}
        assert result["multi_spell_stems_with_mint"] == {}
        assert result["per_channel"]["general"]["total_principals"] == 1

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
        assert result["per_channel"]["build"]["total_principals"] == 2

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

    def test_two_canonicals_same_stem_do_not_collapse_without_mint(self):
        pairs = [
            ("hermes-20260608-153000", "build"),
            ("hermes-20260727-001415", "build"),
            ("hermes", "build"),
        ]
        result = measure(pairs)
        assert result["collapse_without_mint"] == {}
        assert "hermes" in result["multi_spell_stems_with_mint"]
        assert result["multi_spell_stems_with_mint"]["hermes"] == [
            "hermes",
            "hermes-20260608-153000",
            "hermes-20260727-001415",
        ]

    def test_per_channel_reports_separately(self):
        pairs = [
            ("@taOSmd-dev", "build"),
            ("taosmd-dev", "build"),
            ("taosmd-20260609-153000", "general"),
            ("taosmd", "general"),
        ]
        result = measure(pairs)
        assert "build" in result["per_channel"]
        assert "general" in result["per_channel"]
        assert result["per_channel"]["build"]["total_principals"] == 2
        assert result["per_channel"]["general"]["total_principals"] == 2
        assert result["per_channel"]["build"]["collapse_without_mint"] != {}
        assert result["per_channel"]["general"]["canonical_twins"] != []


class TestCollectors:
    def test_collect_from_bus_spool(self, tmp_path):
        spool = tmp_path / "bus-spool.jsonl"
        spool.write_text(
            '\n'.join([
                '{"body": "[bus/build] @taOSmd-dev: hello"}',
                '{"body": "[bus/build] taosmd-dev: world"}',
                '{"body": "[bus/general] taOS: test"}',
                '{"body": "hermes: [AUTO-ACK]"}',
            ]),
            encoding="utf-8",
        )
        pairs = _collect_from_bus_spool(str(spool))
        assert len(pairs) == 4
        assert ("@taOSmd-dev", "build") in pairs
        assert ("taosmd-dev", "build") in pairs
        assert ("taOS", "general") in pairs
        assert ("hermes", "agent-rules") in pairs

    def test_collect_from_archive_empty_dir(self, tmp_path):
        data_dir = tmp_path / "empty"
        data_dir.mkdir()
        (data_dir / "archive").mkdir()
        (data_dir / "archive-index.db").touch()
        pairs = asyncio.run(_collect_from_archive(str(data_dir)))
        assert pairs == []


class TestAsyncMain:
    def test_data_dir_empty_exits_nonzero(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "empty"
        data_dir.mkdir()
        (data_dir / "archive").mkdir()
        (data_dir / "archive-index.db").touch()
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        rc = asyncio.run(async_main(type("Args", (), {"data_dir": str(data_dir), "spool": None})()))
        assert rc == 1

    def test_spool_flag_uses_spool(self, tmp_path, capsys):
        spool = tmp_path / "bus-spool.jsonl"
        spool.write_text('{"body": "[bus/build] taosmd-dev: hello"}\n', encoding="utf-8")
        args = type("Args", (), {"data_dir": None, "spool": str(spool)})()
        rc = asyncio.run(async_main(args))
        assert rc == 0
        captured = capsys.readouterr()
        assert "bus-spool.jsonl" in captured.out
