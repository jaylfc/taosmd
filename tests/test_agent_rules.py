import taosmd


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
