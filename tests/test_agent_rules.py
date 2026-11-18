import taosmd
import pytest


def test_agent_rules_loads_bundled_markdown() -> None:
    text = taosmd.agent_rules()
    assert text.strip(), "agent_rules() returned empty markdown"
    assert "# " in text or "## " in text, "expected markdown headings in agent_rules.md"


def test_agent_rules_none_mode_returns_original() -> None:
    original = taosmd.agent_rules()
    assert taosmd.agent_rules(None) == original
    # tool overrides are ignored for None mode
    assert taosmd.agent_rules(None, search_tool="foo", write_tool="bar") == original


def test_agent_rules_framework_mode_returns_empty() -> None:
    assert taosmd.agent_rules("framework") == ""


def test_agent_rules_taosmd_mode_contains_tools() -> None:
    text = taosmd.agent_rules("taosmd")
    assert "taosmd_search" in text
    assert "taosmd_remember" in text


def test_agent_rules_both_mode_contains_tools_and_framework_mention() -> None:
    text = taosmd.agent_rules("both")
    assert "taosmd_search" in text
    assert "taosmd_remember" in text
    assert "framework" in text.lower()


def test_agent_rules_unknown_mode_raises() -> None:
    try:
        taosmd.agent_rules("nonsense")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "nonsense" in str(e)


def test_agent_rules_both_mode_with_custom_tools() -> None:
    text = taosmd.agent_rules("both", search_tool="memory_search", write_tool="memory_ingest")
    assert "memory_search" in text
    assert "memory_ingest" in text
    assert "taosmd_search" not in text
    assert "taosmd_remember" not in text
    assert "{search_tool}" not in text
    assert "{write_tool}" not in text


def test_agent_rules_taosmd_mode_with_custom_tools() -> None:
    text = taosmd.agent_rules("taosmd", search_tool="memory_search", write_tool="memory_ingest")
    assert "memory_search" in text
    assert "memory_ingest" in text
    assert "taosmd_search" not in text
    assert "taosmd_remember" not in text
    assert "{search_tool}" not in text
    assert "{write_tool}" not in text


def test_agent_rules_supplements_do_not_repeat_the_base_contract() -> None:
    for mode in ("taosmd", "both"):
        text = taosmd.agent_rules(mode)
        assert text.count("### How to consult her") == 1, mode
        assert text.count("### Pending decisions") == 1, mode


def test_agent_rules_have_no_em_dashes() -> None:
    for mode in (None, "taosmd", "both"):
        assert "\u2014" not in taosmd.agent_rules(mode), mode


@pytest.mark.parametrize("mode", ["taosmd", "both"])
def test_lead_line_is_first_for_taosmd_and_both(mode: str) -> None:
    out = taosmd.agent_rules(mode=mode, search_tool="my_search", write_tool="my_write")
    expected_first_line = (
        "First rule: for anything about this system, the user's own setup, its local docs, "
        "or something the user told you before, call `my_search` first. Use web search only for things outside it."
    )
    assert out.splitlines()[0] == expected_first_line
    assert "{search_tool}" not in out
    assert out.count("First rule:") == 1
    # The base text (taosmd.agent_rules()) appears in out AFTER the first line
    base = taosmd.agent_rules()
    # Find the base text starting after the first line
    first_newline = out.find("\n")
    rest = out[first_newline + 1 :]
    assert base in rest


def test_lead_absent_for_default_and_framework() -> None:
    # default (None) mode is byte-identical to docs/agent-rules.md and does not contain 'First rule:'
    base = taosmd.agent_rules()
    assert "First rule:" not in base
    # framework mode returns empty string
    assert taosmd.agent_rules(mode="framework") == ""


def test_lead_is_short() -> None:
    out = taosmd.agent_rules(mode="both")
    first_line = out.splitlines()[0]
    assert len(first_line) <= 220