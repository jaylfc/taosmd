"""Tests for benchmarks/retrieval_latency_probe.py run() wiring (red-first)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Make the benchmark module importable from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.retrieval_latency_probe import (  # noqa: E402
    _summarize,
    run,
)


class TestSummarizeByCategoryRAt:
    """Tests for _summarize by_category r_at exact values."""

    def test_by_category_r_at_exact_values_two_categories(self):
        """_summarize on fixed rows from two categories gives exact by_category r_at values."""
        # Category 1: 3 rows with evidence, ranks 1, 3, 5 (n_hits=10)
        # Category 2: 2 rows with evidence, ranks 2, 10 (n_hits=10)
        rows = [
            # Category 1 rows
            {
                "conversation_id": "c1",
                "question": "q1",
                "category": 1,
                "n_hits": 10,
                "n_adjacent": 0,
                "evidence_hits": 1,
                "evidence_total": 1,
                "retrieval_ms": 10.0,
                "adjacency_ms": 0.0,
                "first_evidence_rank": 1,
            },
            {
                "conversation_id": "c1",
                "question": "q2",
                "category": 1,
                "n_hits": 10,
                "n_adjacent": 0,
                "evidence_hits": 1,
                "evidence_total": 1,
                "retrieval_ms": 10.0,
                "adjacency_ms": 0.0,
                "first_evidence_rank": 3,
            },
            {
                "conversation_id": "c1",
                "question": "q3",
                "category": 1,
                "n_hits": 10,
                "n_adjacent": 0,
                "evidence_hits": 1,
                "evidence_total": 1,
                "retrieval_ms": 10.0,
                "adjacency_ms": 0.0,
                "first_evidence_rank": 5,
            },
            # Category 2 rows
            {
                "conversation_id": "c2",
                "question": "q4",
                "category": 2,
                "n_hits": 10,
                "n_adjacent": 0,
                "evidence_hits": 1,
                "evidence_total": 1,
                "retrieval_ms": 10.0,
                "adjacency_ms": 0.0,
                "first_evidence_rank": 2,
            },
            {
                "conversation_id": "c2",
                "question": "q5",
                "category": 2,
                "n_hits": 10,
                "n_adjacent": 0,
                "evidence_hits": 1,
                "evidence_total": 1,
                "retrieval_ms": 10.0,
                "adjacency_ms": 0.0,
                "first_evidence_rank": 10,
            },
        ]
        errors = []
        ingest_stats = []
        summary = _summarize(rows, errors, ingest_stats)

        # by_category keys are strings (coerced from int in _summarize)
        assert all(isinstance(k, str) for k in summary["by_category"].keys())

        # Category 1: 3 rows with evidence, ranks 1, 3, 5
        # R@1: 1/3, R@3: 2/3, R@5: 3/3, R@10: 3/3
        cat1_r_at = summary["by_category"]["1"]["r_at"]
        assert cat1_r_at["1"] == pytest.approx(1 / 3)
        assert cat1_r_at["3"] == pytest.approx(2 / 3)
        assert cat1_r_at["5"] == pytest.approx(1.0)
        assert cat1_r_at["10"] == pytest.approx(1.0)

        # Category 2: 2 rows with evidence, ranks 2, 10
        # R@1: 0/2, R@3: 1/2, R@5: 1/2, R@10: 2/2
        cat2_r_at = summary["by_category"]["2"]["r_at"]
        assert cat2_r_at["1"] == pytest.approx(0.0)
        assert cat2_r_at["3"] == pytest.approx(0.5)
        assert cat2_r_at["5"] == pytest.approx(0.5)
        assert cat2_r_at["10"] == pytest.approx(1.0)


class TestRunWiring:
    """Tests for run() row building and summary wiring."""

    @pytest.fixture
    def fake_dataset(self, tmp_path: Path) -> Path:
        """Create a minimal fake LoCoMo dataset JSON."""
        dataset = [
            {
                "sample_id": "conv_test",
                "session_id": "session_test",
                "turns": [
                    {"role": "user", "text": "Hello"},
                    {"role": "assistant", "text": "Hi there"},
                ],
                "qa": [
                    {
                        "question": "What did the user say?",
                        "answer": "Hello",
                        "category": 1,
                        "evidence": ["conv_test:0"],
                    },
                    {
                        "question": "What did the assistant say?",
                        "answer": "Hi there",
                        "category": 2,
                        "evidence": ["conv_test:1"],
                    },
                ],
            }
        ]
        dataset_path = tmp_path / "fake_locomo.json"
        dataset_path.write_text(json.dumps(dataset))
        return dataset_path

    @pytest.fixture
    def fake_args(self, fake_dataset: Path, tmp_path: Path) -> MagicMock:
        """Create a fake argparse.Namespace with minimal required args."""
        args = MagicMock()
        args.dataset = str(fake_dataset)
        args.limit = 2
        args.conversations = 1
        args.top_k = 10
        args.retrieval_top_k = 10
        args.adjacent_turns = 0
        args.fusion = "none"
        args.embed_mode = "onnx"
        model_dir = tmp_path / "onnx"
        model_dir.mkdir()
        (model_dir / "model.onnx").write_bytes(b"x")
        args.onnx_path = str(model_dir)
        args.qmd_url = "http://localhost:7832"
        args.reranker = "off"
        args.binary_quant = False
        args.late_interaction = False
        args.colbert_model = ""
        args.max_error_rate = 0.10
        args.timeout = 120.0
        args.out = str(tmp_path / "output.json")
        return args

    @pytest.mark.asyncio
    @patch("benchmarks.retrieval_latency_probe.VectorMemory")
    @patch("benchmarks.retrieval_latency_probe._retrieve")
    @patch("benchmarks.retrieval_latency_probe._load_reranker")
    @patch("benchmarks.retrieval_latency_probe._ingest_conversation")
    @patch("benchmarks.retrieval_latency_probe.httpx.AsyncClient")
    async def test_run_writes_first_evidence_rank_and_summary_r_at(
        self,
        mock_async_client,
        mock_ingest_conversation,
        mock_load_reranker,
        mock_retrieve,
        mock_vector_memory_class,
        fake_args: MagicMock,
        tmp_path: Path,
    ):
        """Drive run() with fakes and assert written rows carry exact first_evidence_rank and summary r_at."""
        # Setup mocks
        mock_client = AsyncMock()
        mock_async_client.return_value.__aenter__.return_value = mock_client

        mock_vmem = AsyncMock()
        mock_vector_memory_class.return_value = mock_vmem
        mock_vmem.init = AsyncMock()
        mock_vmem.embed = AsyncMock(return_value=[0.1] * 384)  # canary succeeds
        mock_vmem.close = AsyncMock()

        mock_ingest_conversation.return_value = (10, 0.5, {0: "turn0", 1: "turn1"})

        mock_load_reranker.return_value = None

        # Scripted hits for two QAs with known dia_ids
        # QA 1 (category 1): evidence = ["conv_test:0"], hit at rank 1
        # QA 2 (category 2): evidence = ["conv_test:1"], hit at rank 3
        hits_qa1 = [
            {"metadata": {"dia_id": "conv_test:0"}, "text": "Hello"},
            {"metadata": {"dia_id": "conv_test:2"}, "text": "Other"},
            {"metadata": {"dia_id": "conv_test:3"}, "text": "Other"},
            {"metadata": {"dia_id": "conv_test:4"}, "text": "Other"},
            {"metadata": {"dia_id": "conv_test:5"}, "text": "Other"},
            {"metadata": {"dia_id": "conv_test:6"}, "text": "Other"},
            {"metadata": {"dia_id": "conv_test:7"}, "text": "Other"},
            {"metadata": {"dia_id": "conv_test:8"}, "text": "Other"},
            {"metadata": {"dia_id": "conv_test:9"}, "text": "Other"},
            {"metadata": {"dia_id": "conv_test:10"}, "text": "Other"},
        ]
        hits_qa2 = [
            {"metadata": {"dia_id": "conv_test:2"}, "text": "Other"},
            {"metadata": {"dia_id": "conv_test:3"}, "text": "Other"},
            {"metadata": {"dia_id": "conv_test:1"}, "text": "Hi there"},
            {"metadata": {"dia_id": "conv_test:4"}, "text": "Other"},
            {"metadata": {"dia_id": "conv_test:5"}, "text": "Other"},
            {"metadata": {"dia_id": "conv_test:6"}, "text": "Other"},
            {"metadata": {"dia_id": "conv_test:7"}, "text": "Other"},
            {"metadata": {"dia_id": "conv_test:8"}, "text": "Other"},
            {"metadata": {"dia_id": "conv_test:9"}, "text": "Other"},
            {"metadata": {"dia_id": "conv_test:10"}, "text": "Other"},
        ]
        mock_retrieve.side_effect = [hits_qa1, hits_qa2]

        # Run the probe
        exit_code = await run(fake_args)
        assert exit_code == 0

        # Read the output JSON
        output_path = Path(fake_args.out)
        assert output_path.exists()
        data = json.loads(output_path.read_text())

        results = data["results"]
        summary = data["summary"]

        # Assert rows have correct first_evidence_rank
        assert len(results) == 2
        # Find results by category
        r1 = next(r for r in results if r["category"] == 1)
        r2 = next(r for r in results if r["category"] == 2)

        assert r1["first_evidence_rank"] == 1
        assert r2["first_evidence_rank"] == 3

        # Assert summary r_at values
        # Category 1: 1 row with evidence, rank 1 -> R@1=1.0, R@3=1.0, R@5=1.0, R@10=1.0
        cat1_r_at = summary["by_category"]["1"]["r_at"]
        assert cat1_r_at["1"] == pytest.approx(1.0)
        assert cat1_r_at["3"] == pytest.approx(1.0)
        assert cat1_r_at["5"] == pytest.approx(1.0)
        assert cat1_r_at["10"] == pytest.approx(1.0)

        # Category 2: 1 row with evidence, rank 3 -> R@1=0.0, R@3=1.0, R@5=1.0, R@10=1.0
        cat2_r_at = summary["by_category"]["2"]["r_at"]
        assert cat2_r_at["1"] == pytest.approx(0.0)
        assert cat2_r_at["3"] == pytest.approx(1.0)
        assert cat2_r_at["5"] == pytest.approx(1.0)
        assert cat2_r_at["10"] == pytest.approx(1.0)

        # Top-level r_at: 2 rows with evidence, ranks 1 and 3
        # R@1: 1/2, R@3: 2/2, R@5: 2/2, R@10: 2/2
        top_r_at = summary["r_at"]
        assert top_r_at["1"] == pytest.approx(0.5)
        assert top_r_at["3"] == pytest.approx(1.0)
        assert top_r_at["5"] == pytest.approx(1.0)
        assert top_r_at["10"] == pytest.approx(1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
