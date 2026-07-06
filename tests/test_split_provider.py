"""Unit tests for generator_profiles.split_provider.

The profile registry stores "provider:model" strings ("ollama:qwen3.5:9b").
Live backends want the bare model name ("qwen3.5:9b"); sending the prefixed
form makes Ollama reject the request and the caller silently degrades. Model
names legitimately contain colons, so only a LEADING known provider token may
be split off.
"""
from __future__ import annotations

import pytest

from taosmd.generator_profiles import split_provider


def test_ollama_prefix_is_split():
    assert split_provider("ollama:qwen3.5:9b") == ("ollama", "qwen3.5:9b")


def test_bare_model_with_colon_passes_through():
    assert split_provider("llama3.1:8b") == ("", "llama3.1:8b")


def test_empty_string_handled():
    assert split_provider("") == ("", "")


def test_bare_model_without_colon_passes_through():
    assert split_provider("gemma4") == ("", "gemma4")


def test_unknown_prefix_is_not_split():
    # "qwen3.5" is a model family, not a provider; the colon belongs to the tag.
    assert split_provider("qwen3.5:9b") == ("", "qwen3.5:9b")


def test_prefix_only_leading_token_counts():
    # A provider token later in the string must not be split.
    assert split_provider("myollama:ollama:x") == ("", "myollama:ollama:x")


def test_double_prefix_strips_one_layer():
    provider, model = split_provider("ollama:ollama:qwen3.5:9b")
    assert provider == "ollama"
    assert model == "ollama:qwen3.5:9b"


@pytest.mark.parametrize("value", ["default", "none"])
def test_sentinels_pass_through(value):
    assert split_provider(value) == ("", value)
