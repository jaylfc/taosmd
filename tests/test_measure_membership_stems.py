"""Tests for channel-membership stem measurement."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from taosmd import service as taosmd_service
from taosmd import api as taosmd_api
from scripts.measure_membership_stems import (
    _collect_from_archive,
    _collect_from_bus_spool,
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

    def test_two_canonicals_share_one_stem_without_mint(self):
        pairs = [
            ("@taOSmd-20260609-153000", "general"),
            ("taosmd-20260609-153000", "general"),
        ]
        result = measure(pairs)
        assert len(result["collapse_without_mint"]) == 1
        assert "taosmd-20260609-153000" in result["collapse_without_mint"]
        assert result["collapse_without_mint"]["taosmd-20260609-153000"] == [
            "@taOSmd-20260609-153000",
            "taosmd-20260609-153000",
        ]


class TestCollectFromArchive:
    def test_collects_unique_members_from_archive(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "taosmd-test"
        data_dir.mkdir()
        monkeypatch.setattr(taosmd_api, "_stores_cache", {})
        stores = asyncio.run(taosmd_api._ensure_stores(str(data_dir)))

        async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
            return [0.0] * 8

        stores["vector"].embed = _fake_embed  # type: ignore[assignment]

        asyncio.run(taosmd_service.a2a_send("alice", "hello", thread="alpha", data_dir=str(data_dir)))
        asyncio.run(taosmd_service.a2a_send("@alice", "hi", thread="alpha", data_dir=str(data_dir)))
        asyncio.run(taosmd_service.a2a_send("bob", "hey", thread="beta", data_dir=str(data_dir)))

        pairs = asyncio.run(_collect_from_archive(str(data_dir)))
        expected = [
            ("alice", "alpha"),
            ("@alice", "alpha"),
            ("bob", "beta"),
        ]
        assert sorted(pairs) == sorted(expected)

        for s in list(taosmd_api._stores_cache.values()):
            for store in (s.get("archive"), s.get("vector"), s.get("kg")):
                if store and hasattr(store, "close"):
                    try:
                        asyncio.run(store.close())
                    except Exception:
                        pass


class TestCollectFromBusSpool:
    def test_parses_bus_spool_lines(self, tmp_path):
        spool = tmp_path / "bus-spool.jsonl"
        spool.write_text(
            "\n".join([
                json.dumps({"body": "[bus/alpha] alice: hi"}),
                json.dumps({"body": "[bus/beta] bob: hey"}),
                json.dumps({"body": "charlie: [AUTO-ACK]"}),
                json.dumps({"body": "not a match"}),
            ])
        )
        pairs = _collect_from_bus_spool(str(spool))
        assert sorted(pairs) == [
            ("alice", "alpha"),
            ("bob", "beta"),
            ("charlie", "agent-rules"),
        ]


class TestAsyncMain:
    def test_data_dir_with_no_rows_exits_nonzero(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = subprocess.run(
            [sys.executable, "scripts/measure_membership_stems.py", "--data-dir", str(empty_dir)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode != 0
        assert "No EVENT_A2A rows found" in result.stderr

    def test_data_dir_with_rows_succeeds(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "taosmd-test"
        data_dir.mkdir()
        monkeypatch.setattr(taosmd_api, "_stores_cache", {})
        stores = asyncio.run(taosmd_api._ensure_stores(str(data_dir)))

        async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
            return [0.0] * 8

        stores["vector"].embed = _fake_embed  # type: ignore[assignment]

        asyncio.run(taosmd_service.a2a_send("alice", "hi", thread="general", data_dir=str(data_dir)))

        env = os.environ.copy()
        env["TAOSMD_ONNX_PATH"] = "/nonexistent"
        result = subprocess.run(
            [sys.executable, "scripts/measure_membership_stems.py", "--data-dir", str(data_dir)],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        assert result.returncode == 0, result.stderr
        assert "Scope:" in result.stdout
        assert "Total distinct principals:" in result.stdout
        assert "Per-channel breakdown:" in result.stdout

        for s in list(taosmd_api._stores_cache.values()):
            for store in (s.get("archive"), s.get("vector"), s.get("kg")):
                if store and hasattr(store, "close"):
                    try:
                        asyncio.run(store.close())
                    except Exception:
                        pass
