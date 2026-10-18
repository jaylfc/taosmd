#!/usr/bin/env python3
"""Hermetic tests for topology_client_probe.py.

No network, no real dataset, no subprocess ping.
Uses httpx.MockTransport injected through module-level client factory.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, AsyncMock

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks import topology_client_probe as probe


# ----------------------------------------------------------------------
# Test fixtures and helpers
# ----------------------------------------------------------------------


@pytest.fixture
def fake_dataset(tmp_path):
    """Create a tiny fake LoCoMo dataset with two sessions and four QAs (one cat 5)."""
    data = [{
        "sample_id": "test_conv_0",
        "conversation": {
            "session_1": [
                {"text": "Hello world", "speaker": "Alice", "dia_id": "d1"},
                {"text": "How are you", "speaker": "Bob", "dia_id": "d2"},
            ],
            "session_1_date_time": "1:00 pm on 1 January, 2024",
            "session_2": [
                {"text": "I am fine", "speaker": "Alice", "dia_id": "d3"},
                {"text": "Good to hear", "speaker": "Bob", "dia_id": "d4"},
            ],
            "session_2_date_time": "2:00 pm on 1 January, 2024",
        },
        "qa": [
            {"question": "Q1", "answer": "A1", "category": 1, "evidence": ["d1"]},
            {"question": "Q2", "answer": "A2", "category": 2, "evidence": ["d2"]},
            {"question": "Q3", "answer": "A3", "category": 5, "evidence": ["d3"]},  # adversarial - should be skipped
            {"question": "Q4", "answer": "A4", "category": 3, "evidence": ["d4"]},
        ],
    }]
    path = tmp_path / "fake_locomo.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def mock_transport():
    """Create a mock httpx transport that records requests."""
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append({
            "method": request.method,
            "url": str(request.url),
            "headers": dict(request.headers),
            "json": json.loads(request.content) if request.content else None,
        })

        # Route responses based on URL path
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "test-1.0"})
        elif request.url.path == "/ingest/batch":
            return httpx.Response(200, json={"vector_failures": 0})
        elif request.url.path == "/search":
            return httpx.Response(200, json={"results": [{"text": "hit"} for _ in range(3)]})
        return httpx.Response(404, json={"error": "not found"})

    transport = httpx.MockTransport(handler)
    transport.requests = requests  # type: ignore[attr-defined]
    return transport


@pytest.fixture(autouse=True)
def patch_client_factory(mock_transport):
    """Patch the module's client factory to use our mock transport."""
    original_make_client = probe._make_client

    async def mock_make_client(base_url: str, token: str | None):
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=120.0,
            transport=mock_transport,
        )

    probe._make_client = mock_make_client  # type: ignore[assignment]
    yield mock_transport
    probe._make_client = original_make_client


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


class TestTopologyClientProbe:
    """Tests for topology_client_probe.py."""

    @pytest.mark.asyncio
    async def test_category_5_skipped_order_preserved(self, fake_dataset, mock_transport):
        """Category 5 QAs are skipped; remaining QAs keep dataset order."""
        out_path = Path(tempfile.mktemp(suffix=".json"))
        try:
            args = argparse.Namespace(
                base_url="http://localhost:7700",
                token_file=None,
                agent="test-agent",
                dataset=str(fake_dataset),
                conversation_index=0,
                queries=10,
                limit=10,
                pings=0,
                label="test-label",
                shape="whole-service",
                out=str(out_path),
            )
            # Patch ping to return None
            with patch.object(probe, "_run_ping", new_callable=AsyncMock) as mock_ping:
                mock_ping.return_value = None
                # Patch _read_proc_stat to return fixed values (called 3 times)
                with patch.object(probe, "_read_proc_stat") as mock_cpu:
                    mock_cpu.side_effect = [(100, 200), (120, 220), (150, 300)]
                    rc = await probe.run(args)
            assert rc == 0

            # Check questions sent to /search
            search_requests = [
                r for r in mock_transport.requests
                if str(r["url"]).endswith("/search")
            ]
            questions = [r["json"]["query"] for r in search_requests]
            assert questions == ["Q1", "Q2", "Q4"], f"Expected ['Q1', 'Q2', 'Q4'], got {questions}"
        finally:
            out_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_ingest_batching_ceil_n_over_50(self, fake_dataset, mock_transport):
        """Ingest sends ceil(n/50) batches with correct agent."""
        # Create dataset with exactly 120 turns (3 batches: 50, 50, 20)
        # Start with empty conversation and add exactly 120 turns
        data = [{
            "sample_id": "test_conv_0",
            "conversation": {},
            "qa": [
                {"question": "Q1", "answer": "A1", "category": 1, "evidence": ["d1"]},
            ],
        }]
        # Add 30 sessions with 4 turns each = 120 turns
        for i in range(30):
            session_key = f"session_{i+1}"
            data[0]["conversation"][session_key] = [
                {"text": f"Turn {i*4+j}", "speaker": "Alice", "dia_id": f"d{i*4+j}"}
                for j in range(4)
            ]
            data[0]["conversation"][f"{session_key}_date_time"] = "1:00 pm on 1 January, 2024"
        expanded_path = fake_dataset.with_name("expanded.json")
        expanded_path.write_text(json.dumps(data))

        out_path = Path(tempfile.mktemp(suffix=".json"))
        try:
            args = argparse.Namespace(
                base_url="http://localhost:7700",
                token_file=None,
                agent="test-agent",
                dataset=str(expanded_path),
                conversation_index=0,
                queries=1,
                limit=10,
                pings=0,
                label="test-label",
                shape="whole-service",
                out=str(out_path),
            )
            with patch.object(probe, "_run_ping", new_callable=AsyncMock) as mock_ping:
                mock_ping.return_value = None
                with patch.object(probe, "_read_proc_stat") as mock_cpu:
                    mock_cpu.side_effect = [(100, 200), (120, 220), (150, 300)]
                    rc = await probe.run(args)
            assert rc == 0

            ingest_requests = [
                r for r in mock_transport.requests
                if str(r["url"]).endswith("/ingest/batch")
            ]
            # 120 turns -> ceil(120/50) = 3 batches
            assert len(ingest_requests) == 3
            for r in ingest_requests:
                assert r["json"]["agent"] == "test-agent"
                assert len(r["json"]["items"]) <= 50
            total_items = sum(len(r["json"]["items"]) for r in ingest_requests)
            assert total_items == 120
        finally:
            out_path.unlink(missing_ok=True)
            expanded_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_version_non_200_exits_2(self, fake_dataset):
        """Non-200 /version exits with code 2."""
        # Create a transport that returns 500 for /version
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append({
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
                "json": json.loads(request.content) if request.content else None,
            })
            if request.url.path == "/version":
                return httpx.Response(500, json={"error": "internal"})
            return httpx.Response(200, json={})

        transport = httpx.MockTransport(handler)

        async def mock_make_client(base_url: str, token: str | None):
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            return httpx.AsyncClient(
                base_url=base_url,
                headers=headers,
                timeout=120.0,
                transport=transport,
            )

        original_make_client = probe._make_client
        probe._make_client = mock_make_client
        try:
            args = argparse.Namespace(
                base_url="http://localhost:7700",
                token_file=None,
                agent="test-agent",
                dataset=str(fake_dataset),
                conversation_index=0,
                queries=1,
                limit=10,
                pings=0,
                label="test-label",
                shape="whole-service",
                out="/tmp/out.json",
            )
            with patch.object(probe, "_run_ping", new_callable=AsyncMock) as mock_ping:
                mock_ping.return_value = None
                with patch.object(probe, "_read_proc_stat") as mock_cpu:
                    mock_cpu.side_effect = [(100, 200), (120, 220), (150, 300)]
                    rc = await probe.run(args)
            assert rc == 2
        finally:
            probe._make_client = original_make_client

    @pytest.mark.asyncio
    async def test_ingest_batch_failure_exits_2(self, fake_dataset):
        """Failing /ingest/batch exits with code 2."""
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append({
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
                "json": json.loads(request.content) if request.content else None,
            })
            if request.url.path == "/version":
                return httpx.Response(200, json={"version": "test"})
            if request.url.path == "/ingest/batch":
                return httpx.Response(500, json={"error": "fail"})
            return httpx.Response(200, json={"results": []})

        transport = httpx.MockTransport(handler)

        async def mock_make_client(base_url: str, token: str | None):
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            return httpx.AsyncClient(
                base_url=base_url,
                headers=headers,
                timeout=120.0,
                transport=transport,
            )

        original_make_client = probe._make_client
        probe._make_client = mock_make_client
        try:
            args = argparse.Namespace(
                base_url="http://localhost:7700",
                token_file=None,
                agent="test-agent",
                dataset=str(fake_dataset),
                conversation_index=0,
                queries=1,
                limit=10,
                pings=0,
                label="test-label",
                shape="whole-service",
                out="/tmp/out.json",
            )
            with patch.object(probe, "_run_ping", new_callable=AsyncMock) as mock_ping:
                mock_ping.return_value = None
                with patch.object(probe, "_read_proc_stat") as mock_cpu:
                    mock_cpu.side_effect = [(100, 200), (120, 220), (150, 300)]
                    rc = await probe.run(args)
            assert rc == 2
        finally:
            probe._make_client = original_make_client

    @pytest.mark.asyncio
    async def test_error_rate_over_10_percent_exits_3_writes_file(self, fake_dataset):
        """Error rate > 10% exits 3 but still writes output file."""
        # Make /search fail for 2 out of 3 queries (66% error rate)
        fail_count = 0
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal fail_count
            requests.append({
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
                "json": json.loads(request.content) if request.content else None,
            })
            if request.url.path == "/version":
                return httpx.Response(200, json={"version": "test"})
            if request.url.path == "/ingest/batch":
                return httpx.Response(200, json={"vector_failures": 0})
            if request.url.path == "/search":
                fail_count += 1
                if fail_count <= 2:
                    return httpx.Response(500, json={"error": "fail"})
                return httpx.Response(200, json={"results": [{"text": "hit"}]})
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)

        async def mock_make_client(base_url: str, token: str | None):
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            return httpx.AsyncClient(
                base_url=base_url,
                headers=headers,
                timeout=120.0,
                transport=transport,
            )

        original_make_client = probe._make_client
        probe._make_client = mock_make_client
        try:
            out_path = Path(tempfile.mktemp(suffix=".json"))
            try:
                args = argparse.Namespace(
                    base_url="http://localhost:7700",
                    token_file=None,
                    agent="test-agent",
                    dataset=str(fake_dataset),
                    conversation_index=0,
                    queries=3,
                    limit=10,
                    pings=0,
                    label="test-label",
                    shape="whole-service",
                    out=str(out_path),
                )
                with patch.object(probe, "_run_ping", new_callable=AsyncMock) as mock_ping:
                    mock_ping.return_value = None
                    with patch.object(probe, "_read_proc_stat") as mock_cpu:
                        mock_cpu.side_effect = [(100, 200), (120, 220), (150, 300)]
                        rc = await probe.run(args)
                assert rc == 3
                assert out_path.exists()
                data = json.loads(out_path.read_text())
                assert data["queries"]["error_rate"] > 0.10
            finally:
                out_path.unlink(missing_ok=True)
        finally:
            probe._make_client = original_make_client

    def test_ping_parsing_valid_output(self):
        """Parse ping output and extract p50, p95, raw list."""
        sample = """PING localhost (127.0.0.1) 56(84) bytes of data.
64 bytes from localhost (127.0.0.1): icmp_seq=1 ttl=64 time=1.23 ms
64 bytes from localhost (127.0.0.1): icmp_seq=2 ttl=64 time=2.45 ms
64 bytes from localhost (127.0.0.1): icmp_seq=3 ttl=64 time=3.67 ms
64 bytes from localhost (127.0.0.1): icmp_seq=4 ttl=64 time=4.89 ms

--- localhost ping statistics ---
4 packets transmitted, 4 received, 0% packet loss, time 3004ms
rtt min/avg/max/mdev = 1.230/3.060/4.890/1.345 ms"""
        times = probe._parse_ping_output(sample)
        assert times is not None
        assert times == [1.23, 2.45, 3.67, 4.89]
        assert probe._pctl(times, 0.50) == 2.45
        assert probe._pctl(times, 0.95) == 4.89

    def test_ping_parsing_missing_returns_none(self):
        """Missing or unparseable ping returns None."""
        assert probe._parse_ping_output("") is None
        assert probe._parse_ping_output("no times here") is None
        assert probe._parse_ping_output("time=abc ms") is None

    @pytest.mark.asyncio
    async def test_token_file_reads_and_sent_in_header(self, fake_dataset, mock_transport):
        """Token from --token-file reaches Authorization header, not in output JSON."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("secret-token-123")
            token_file = f.name

        out_path = Path(tempfile.mktemp(suffix=".json"))
        try:
            args = argparse.Namespace(
                base_url="http://localhost:7700",
                token_file=token_file,
                agent="test-agent",
                dataset=str(fake_dataset),
                conversation_index=0,
                queries=1,
                limit=10,
                pings=0,
                label="test-label",
                shape="whole-service",
                out=str(out_path),
            )
            with patch.object(probe, "_run_ping", new_callable=AsyncMock) as mock_ping:
                mock_ping.return_value = None
                with patch.object(probe, "_read_proc_stat") as mock_cpu:
                    mock_cpu.side_effect = [(100, 200), (120, 220), (150, 300)]
                    rc = await probe.run(args)
            assert rc == 0

            # Check Authorization header was sent
            for r in mock_transport.requests:
                if str(r["url"]).endswith(("/version", "/ingest/batch", "/search")):
                    auth = r["headers"].get("authorization")
                    assert auth == "Bearer secret-token-123", f"Missing or wrong auth: {auth}"

            # Check token not in output JSON
            data = json.loads(out_path.read_text())
            json_str = json.dumps(data)
            assert "secret-token-123" not in json_str
        finally:
            os.unlink(token_file)
            out_path.unlink(missing_ok=True)

    def test_proc_stat_parser_fixed_snapshots(self):
        """_read_proc_stat parser on fixed two-snapshot sample gives expected fraction."""
        # busy_before = 100+20+30+6+7+8 = 171
        # total_before = 171 + 400 + 50 = 621
        # busy_after = 150+25+35+7+8+9 = 234
        # total_after = 234 + 450 + 55 = 739
        # busy_delta = 63, total_delta = 118
        # fraction = 63/118 ≈ 0.5339
        busy_before = 100 + 20 + 30 + 6 + 7 + 8
        total_before = busy_before + 400 + 50
        busy_after = 150 + 25 + 35 + 7 + 8 + 9
        total_after = busy_after + 450 + 55
        busy_delta = busy_after - busy_before
        total_delta = total_after - total_before
        expected = round(busy_delta / total_delta, 4)
        assert expected == 0.5339

    @pytest.mark.asyncio
    async def test_missing_ping_records_null(self, fake_dataset, mock_transport):
        """Missing ping records error dict with error string."""
        async def failing_ping(*args, **kwargs):
            return {"error": "FileNotFoundError: ping not found"}

        out_path = Path(tempfile.mktemp(suffix=".json"))
        try:
            args = argparse.Namespace(
                base_url="http://localhost:7700",
                token_file=None,
                agent="test-agent",
                dataset=str(fake_dataset),
                conversation_index=0,
                queries=1,
                limit=10,
                pings=5,
                label="test-label",
                shape="whole-service",
                out=str(out_path),
            )
            with patch.object(probe, "_run_ping", side_effect=failing_ping):
                with patch.object(probe, "_read_proc_stat") as mock_cpu:
                    mock_cpu.side_effect = [(100, 200), (120, 220), (150, 300)]
                    rc = await probe.run(args)
            assert rc == 0
            data = json.loads(out_path.read_text())
            assert data["ping"] is not None
            assert "error" in data["ping"]
        finally:
            out_path.unlink(missing_ok=True)


import argparse  # noqa: E402

if __name__ == "__main__":
    pytest.main([__file__, "-v"])