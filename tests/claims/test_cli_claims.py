"""CLI parser tests for ``taosmd claims status`` and ``taosmd verify``.

Tests build the parser via the exported ``_build_parser()`` helper (refactored
out of ``main()`` in cli.py with identical behaviour) and assert that the two
new subcommands parse and route to the right handler.
"""
from __future__ import annotations

import pytest

from taosmd.cli import _build_parser


def test_claims_status_parses():
    parser = _build_parser()
    args = parser.parse_args(["--data-dir", "/tmp/td", "claims", "status"])
    assert args.cmd == "claims"
    assert args.claims_cmd == "status"
    assert args.data_dir == "/tmp/td"


def test_claims_requires_subcommand():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["claims"])


def test_verify_parses_defaults():
    parser = _build_parser()
    args = parser.parse_args(["verify"])
    assert args.cmd == "verify"
    assert args.model == "qwen3:4b-instruct-2507"
    assert args.ollama_url == "http://localhost:11434"
    assert args.batch == 100


def test_verify_parses_overrides():
    parser = _build_parser()
    args = parser.parse_args([
        "verify",
        "--model", "llama3:8b",
        "--ollama-url", "http://192.168.1.2:11434",
        "--batch", "50",
    ])
    assert args.cmd == "verify"
    assert args.model == "llama3:8b"
    assert args.ollama_url == "http://192.168.1.2:11434"
    assert args.batch == 50


def test_verify_parses_with_data_dir():
    parser = _build_parser()
    args = parser.parse_args(["--data-dir", "/custom/dir", "verify"])
    assert args.data_dir == "/custom/dir"
    assert args.cmd == "verify"
