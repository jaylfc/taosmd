import taosmd


def test_agent_rules_loads_bundled_markdown() -> None:
    text = taosmd.agent_rules()
    assert text.strip(), "agent_rules() returned empty markdown"
    assert "# " in text or "## " in text, "expected markdown headings in agent_rules.md"
