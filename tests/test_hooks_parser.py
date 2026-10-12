"""Tests for taosmd/hooks/parser.py.

Coverage:
- Fixture parsing with exact item counts per kind.
- ID format rules.
- Timestamp conversion from ISO-8601.
- Byte capping (8 KB limit per field).
- Noise filtering (queue-operation, attachment, system, file-history-snapshot,
  last-prompt, ai-title, cost-state).
"""

import json
from pathlib import Path

import pytest

from taosmd.hooks.parser import parse_entry

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "claude_code_transcript.jsonl"


def test_fixture_parsing():
    """Verify fixture produces expected items.

    From claude_code_transcript.jsonl (16 lines):
    - Lines 1, 2: user messages (text content)
    - Line 3: assistant message (text content)
    - Line 5: tool_use entry (assistant)
    - Line 6: tool_result entry (user)
    - Line 7: assistant with text + tool_use (both items)
    - Line 8: tool_result entry (user)
    - Lines 9-15: noise types, filtered out
    - Line 16: session start marker, filtered out

    Expected:
    - 2 user messages (kind="message")
    - 2 assistant messages (kind="message")
    - 2 tool_use entries (kind="tool_use")
    - 2 tool_result entries (kind="tool_result")
    """
    with FIXTURE_PATH.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    all_items = []
    for i, line in enumerate(lines):
        byte_offset = sum(len(l) for l in lines[:i])
        session_id = "s1"
        items, _ = parse_entry(byte_offset, line, session_id)
        if items:
            all_items.extend(items)

    # Count by kind
    by_kind = {}
    for item in all_items:
        kind = item["metadata"]["kind"]
        by_kind[kind] = by_kind.get(kind, 0) + 1

    # Verify counts
    assert by_kind.get("message", 0) == 4, f"Expected 4 message items, got {by_kind.get('message', 0)}"
    assert by_kind.get("tool_use", 0) == 2, f"Expected 2 tool_use items, got {by_kind.get('tool_use', 0)}"
    assert by_kind.get("tool_result", 0) == 2, f"Expected 2 tool_result items, got {by_kind.get('tool_result', 0)}"

    # Verify roles - only count message items, not tool_result
    user_msgs = [i for i in all_items if i["metadata"]["role"] == "user" and i["metadata"]["kind"] == "message"]
    assistant_msgs = [i for i in all_items if i["metadata"]["role"] == "assistant" and i["metadata"]["kind"] == "message"]
    assert len(user_msgs) == 2, f"Expected 2 user messages, got {len(user_msgs)}"
    assert len(assistant_msgs) == 2, f"Expected 2 assistant messages, got {len(assistant_msgs)}"

    # Verify specific content
    assistant_with_tool_use = [
        i for i in all_items if i["metadata"]["role"] == "assistant" and i["metadata"]["kind"] == "tool_use"
    ]
    assert len(assistant_with_tool_use) == 2

    # Verify tool_result has both string and list content
    tool_results = [i for i in all_items if i["metadata"]["kind"] == "tool_result"]
    assert len(tool_results) == 2

    # Verify timestamps are floats from ISO strings
    for item in all_items:
        assert isinstance(item["metadata"]["timestamp"], float), \
            f"Timestamp not float: {item['metadata']['timestamp']}"

    # Verify IDs have correct format
    for item in all_items:
        assert item["id"].startswith("claude-code:"), \
            f"ID doesn't start with 'claude-code:': {item['id']}"


def test_fallback_id_for_missing_uuid():
    """When uuid is missing, use fallback id based on byte offset + line."""
    # Line 16 has no uuid, just {"sessionId":"s1"}
    with FIXTURE_PATH.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    byte_offset = sum(len(l) for l in lines[:15])  # Line 16 is index 15
    session_id = "s1"
    items, base_id = parse_entry(byte_offset, lines[15], session_id)

    # Should return None, None (noise type or invalid entry)
    assert items is None


def test_fallback_id_identical_lines_different_offsets():
    """Two identical lines at different byte offsets get different ids."""
    # Create two identical user entries without uuid
    identical_line = '{"type":"user","message":{"role":"user","content":"test"},"sessionId":"s1","timestamp":"2026-10-07T22:53:25.123Z"}'
    
    # Parse at byte offset 0
    items1, id1 = parse_entry(0, identical_line, "s1")
    assert items1 is not None
    assert len(items1) == 1
    assert id1 == items1[0]["id"]
    
    # Parse at byte offset 50 (different offset)
    items2, id2 = parse_entry(50, identical_line, "s1")
    assert items2 is not None
    assert len(items2) == 1
    assert id2 == items2[0]["id"]
    
    # The IDs should be different because byte offsets differ
    assert id1 != id2
    
    # The ID should equal sha256 of offset+raw_line
    import hashlib
    key = f"50{identical_line}".encode("utf-8")
    expected_id = hashlib.sha256(key).hexdigest()
    assert id2 == expected_id


def test_byte_capping():
    """Fields longer than 8 KB are capped and truncated flag set."""
    # Create a fixture line with >8KB of emoji text
    emoji_text = "😀" * 10000  # Each emoji is 4 bytes UTF-8, so ~40KB
    large_line = json.dumps(
        {
            "type": "user",
            "message": {"content": emoji_text},
            "uuid": "large1",
            "sessionId": "test",
            "timestamp": "2026-10-07T22:53:25.123Z",
        },
        ensure_ascii=False,
    )

    items, _ = parse_entry(0, large_line, "test")
    assert items is not None
    assert len(items) == 1
    item = items[0]

    # Check the text field is <= 8192 bytes
    encoded_text = item["text"].encode("utf-8")
    assert len(encoded_text) <= 8192, \
        f"Text field exceeds 8KB: {len(encoded_text)} bytes"

    # Check it decodes cleanly
    decoded = encoded_text.decode("utf-8")
    assert decoded == item["text"], "Text doesn't round-trip cleanly"

    # Check truncated flag
    assert item["metadata"]["truncated"], \
        "Truncated should be True for oversized field"


def test_noise_filtering():
    """Verify noise types are filtered out."""
    noise_types = [
        "queue-operation",
        "attachment",
        "system",
        "file-history-snapshot",
        "last-prompt",
        "ai-title",
        "cost-state",
    ]

    for noise_type in noise_types:
        # Create a noise entry
        noise_line = json.dumps(
            {
                "type": noise_type,
                "content": "noise",
                "timestamp": "2026-10-07T22:53:25.123Z",
                "sessionId": "s1",
            },
            ensure_ascii=False,
        )

        items, _ = parse_entry(0, noise_line, "s1")
        assert items is None, \
            f"Noise type '{noise_type}' should be filtered out but returned: {items}"


def test_tool_result_string_and_list():
    """tool_result content works as both string and list of text blocks."""
    # From fixture, line 8 has tool_result with content as list of text blocks
    with FIXTURE_PATH.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find the tool_result with list content (line 8)
    tool_result_list_line = lines[7]  # Index 7 for line 8
    byte_offset = sum(len(l) for l in lines[:7])

    session_id = "s1"
    items, _ = parse_entry(byte_offset, tool_result_list_line, session_id)

    # Should have one tool_result item
    assert items is not None
    assert len(items) == 1
    assert items[0]["metadata"]["kind"] == "tool_result"


def test_assistant_text_and_tool_use():
    """Assistant entry with both text and tool_use yields both items."""
    # From fixture, line 7 has assistant with text + tool_use
    with FIXTURE_PATH.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    assistant_with_both_line = lines[6]  # Index 6 for line 7
    byte_offset = sum(len(l) for l in lines[:6])

    session_id = "s1"
    items, _ = parse_entry(byte_offset, assistant_with_both_line, session_id)

    # Should have both message and tool_use items
    assert items is not None
    assert len(items) == 2

    kinds = {item["metadata"]["kind"] for item in items}
    assert "message" in kinds
    assert "tool_use" in kinds

    # Verify both have correct roles
    roles = {item["metadata"]["role"] for item in items}
    assert "assistant" in roles


def test_timestamp_conversion():
    """ISO-8601 timestamps are converted to epoch seconds."""
    test_cases = [
        ("2026-10-07T22:53:25.123Z", "Z suffix"),
        ("2026-10-07T22:53:25.123+00:00", "with offset"),
        ("", "empty string"),
    ]

    for ts_str, description in test_cases:
        from taosmd.hooks.parser import _iso_to_epoch
        epoch = _iso_to_epoch(ts_str)

        assert isinstance(epoch, float), \
            f"Timestamp conversion failed for {description}: {epoch}"

        if ts_str:
            # Should be around 2026, not current time
            assert 1686220800 < epoch < 1900000000, \
                f"Timestamp out of range for {description}: {epoch}"


def test_id_format_with_uuid():
    """Items with uuid use claude-code:<sessionId>:<uuid> format."""
    # User entry with uuid u1
    user_line = '{"type":"user","message":{"role":"user","content":"plain string prompt"},"uuid":"u1","parentUuid":"","timestamp":"2026-10-07T22:53:25.123Z","sessionId":"s1"}'

    items, base_id = parse_entry(0, user_line, "s1")
    assert items is not None
    assert len(items) == 1

    # The base_id should include the uuid
    assert "u1" in base_id, f"Base ID should contain uuid: {base_id}"

    # Item id should match base_id (since it's a message, not tool_use/tool_result)
    assert items[0]["id"] == base_id, \
        f"Item ID {items[0]['id']} should equal base ID {base_id}"


def test_id_format_fallback():
    """Items without uuid use fallback id based on byte offset + line."""
    # Entry without uuid
    no_uuid_line = '{"type":"user","message":{"role":"user","content":"no uuid"},"sessionId":"s1","timestamp":"2026-10-07T22:53:25.123Z"}'

    items, base_id = parse_entry(0, no_uuid_line, "s1")
    assert items is not None
    assert len(items) == 1

    # Base ID should be a sha256 hash
    assert len(base_id) == 64, f"Fallback ID should be 64 chars (sha256), got: {base_id}"


def test_empty_lines_filtered():
    """Empty lines are filtered out."""
    empty_lines = ["\n", "   \n", "\t\n"]

    for line in empty_lines:
        items, _ = parse_entry(0, line, "s1")
        assert items is None, \
            f"Empty line should be filtered out but got: {items}"


def test_invalid_json_filtered():
    """Invalid JSON lines are filtered out."""
    invalid_lines = [
        "not json at all",
        '{"type": invalid}',
        "{incomplete json",
    ]

    for line in invalid_lines:
        items, _ = parse_entry(0, line, "s1")
        assert items is None, \
            f"Invalid JSON line should be filtered out but got: {items}"


def test_missing_type_filtered():
    """Entries without 'type' field are filtered out."""
    no_type_line = '{"message":{"content":"test"},"sessionId":"s1"}'

    items, _ = parse_entry(0, no_type_line, "s1")
    assert items is None, \
        f"Entry without type should be filtered out but got: {items}"


def test_user_message_string_content():
    """User message with string content works."""
    user_line = '{"type":"user","message":{"role":"user","content":"plain string"},"uuid":"u1","timestamp":"2026-10-07T22:53:25.123Z","sessionId":"s1"}'

    items, _ = parse_entry(0, user_line, "s1")
    assert items is not None
    assert len(items) == 1
    assert items[0]["metadata"]["role"] == "user"
    assert items[0]["metadata"]["kind"] == "message"
    assert items[0]["text"] == "plain string"


def test_user_message_list_content():
    """User message with list content works (text blocks)."""
    user_line = '{"type":"user","message":{"role":"user","content":[{"type":"text","text":"block1"},{"type":"text","text":"block2"}]},"uuid":"u2","timestamp":"2026-10-07T22:53:25.123Z","sessionId":"s1"}'

    items, _ = parse_entry(0, user_line, "s1")
    assert items is not None
    assert len(items) == 1
    assert items[0]["text"] == "block1\nblock2"


def test_assistant_message_content():
    """Assistant message content works."""
    assistant_line = '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"assistant reply"}]},"uuid":"a1","timestamp":"2026-10-07T22:53:25.123Z","sessionId":"s1"}'

    items, _ = parse_entry(0, assistant_line, "s1")
    assert items is not None
    assert len(items) == 1
    assert items[0]["metadata"]["role"] == "assistant"
    assert items[0]["metadata"]["kind"] == "message"
    assert items[0]["text"] == "assistant reply"


def test_tool_use_entry():
    """Tool use entry works."""
    tool_use_line = '{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"Bash","input":{"command":"ls"}}]},"uuid":"a1","timestamp":"2026-10-07T22:53:25.123Z","sessionId":"s1"}'

    items, _ = parse_entry(0, tool_use_line, "s1")
    assert items is not None
    assert len(items) == 1
    assert items[0]["metadata"]["kind"] == "tool_use"
    assert items[0]["metadata"]["tool_name"] == "Bash"

    # Tool use text should be JSON string of input
    assert '"command":"ls"' in items[0]["text"] or "'command':'ls'" in items[0]["text"]


def test_truncated_flag():
    """Truncated flag is set correctly."""
    # Create a line that will exceed 8KB
    long_content = "x" * 10000  # 10KB
    long_line = json.dumps(
        {
            "type": "user",
            "message": {"content": long_content},
            "uuid": "large",
            "sessionId": "s1",
            "timestamp": "2026-10-07T22:53:25.123Z",
        },
        ensure_ascii=False,
    )

    items, _ = parse_entry(0, long_line, "s1")
    assert items is not None
    assert len(items) == 1

    # The text field should be capped
    assert len(items[0]["text"]) < len(long_content)

    # Check metadata has truncated flag
    assert "truncated" in items[0]["metadata"]


def test_source_metadata():
    """Source metadata is always 'hook:claude-code'."""
    user_line = '{"type":"user","message":{"role":"user","content":"test"},"uuid":"u1","timestamp":"2026-10-07T22:53:25.123Z","sessionId":"s1"}'

    items, _ = parse_entry(0, user_line, "s1")
    assert items is not None
    assert items[0]["metadata"]["source"] == "hook:claude-code"


def test_session_id_override():
    """Session ID from entry overrides passed session_id."""
    # Entry has sessionId "s2", but we pass "s1"
    user_line = '{"type":"user","message":{"role":"user","content":"test"},"uuid":"u1","sessionId":"s2","timestamp":"2026-10-07T22:53:25.123Z"}'

    items, _ = parse_entry(0, user_line, "s1")
    assert items is not None
    assert items[0]["metadata"]["session_id"] == "s2", \
        f"Expected session_id 's2', got {items[0]['metadata']['session_id']}"


def test_assistant_no_text_no_tool_use():
    """Assistant entry with no text or tool_use is filtered out."""
    assistant_line = '{"type":"assistant","message":{"role":"assistant","content":[{"type":"thinking","thinking":"hmm"}]},"uuid":"a1","timestamp":"2026-10-07T22:53:25.123Z","sessionId":"s1"}'

    items, _ = parse_entry(0, assistant_line, "s1")
    assert items is None, \
        f"Assistant with no text or tool_use should be filtered out but got: {items}"


def test_user_no_content():
    """User entry with no content is filtered out."""
    user_line = '{"type":"user","message":{"role":"user"},"uuid":"u1","timestamp":"2026-10-07T22:53:25.123Z","sessionId":"s1"}'

    items, _ = parse_entry(0, user_line, "s1")
    assert items is None, \
        f"User with no content should be filtered out but got: {items}"


def test_tool_result_no_content():
    """tool_result with no content is filtered out."""
    tool_result_line = '{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1"}]},"uuid":"u1","timestamp":"2026-10-07T22:53:25.123Z","sessionId":"s1"}'
    
    items, _ = parse_entry(0, tool_result_line, "s1")
    # tool_result with empty content should be filtered out
    assert items is None


@pytest.mark.parametrize(
    "label,line",
    [
        ("type_as_list", '{"type": ["x"],"timestamp":"2026-10-07T22:53:25.123Z","sessionId":"s1"}'),
        ("type_as_dict", '{"type": {"a": 1},"timestamp":"2026-10-07T22:53:25.123Z","sessionId":"s1"}'),
        ("type_as_int", '{"type": 5,"timestamp":"2026-10-07T22:53:25.123Z","sessionId":"s1"}'),
        ("user_text_int", '{"type":"user","message":{"role":"user","content":[{"type":"text","text":5}]},"uuid":"u1","timestamp":"2026-10-07T22:53:25.123Z","sessionId":"s1"}'),
        ("assistant_text_int", '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":5}]},"uuid":"a1","timestamp":"2026-10-07T22:53:25.123Z","sessionId":"s1"}'),
        ("tool_result_text_int", '{"type":"user","message":{"role":"user","content":[{"type":"tool_result","content":[{"type":"text","text":7}]}]},"uuid":"u1","timestamp":"2026-10-07T22:53:25.123Z","sessionId":"s1"}'),
        ("uuid_as_dict", '{"type":"user","message":{"role":"user","content":"hi"},"uuid":{"a":1},"timestamp":"2026-10-07T22:53:25.123Z","sessionId":"s1"}'),
        ("sessionId_as_list", '{"type":"user","message":{"role":"user","content":"hi"},"uuid":"u1","timestamp":"2026-10-07T22:53:25.123Z","sessionId":["s1"]}'),
    ],
    ids=[
        "type-list",
        "type-dict",
        "type-int",
        "user-text-int",
        "assistant-text-int",
        "tool-result-text-int",
        "uuid-dict",
        "sessionid-list",
    ],
)
def test_malformed_field_types_never_raise(label, line):
    """parse_entry must never raise on JSON whose field types are unexpected."""
    session_id = "s1"
    try:
        items, base_id = parse_entry(0, line, session_id)
    except Exception as exc:
        pytest.fail(f"{label}: parse_entry raised {type(exc).__name__}: {exc}")

    if label.startswith("type_as_"):
        assert (items, base_id) == (None, None), (
            f"{label}: expected (None, None), got ({items!r}, {base_id!r})"
        )
    elif label == "uuid_as_dict":
        assert not base_id.startswith("claude-code:"), (
            f"uuid-as-dict: expected fallback id without 'claude-code:' prefix, got {base_id!r}"
        )
    elif label == "sessionId_as_list":
        assert items is not None and len(items) == 1, (
            f"sessionId-as-list: expected 1 item, got {items!r}"
        )
        assert items[0]["metadata"]["session_id"] == session_id, (
            f"sessionId-as-list: expected session_id {session_id!r}, "
            f"got {items[0]['metadata']['session_id']!r}"
        )


def test_user_text_and_tool_result_both_kept():
    """A user entry with both text and tool_result blocks keeps both items."""
    line = json.dumps(
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "text", "text": "USER TEXT"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [{"type": "text", "text": "out"}],
                    },
                ],
            },
            "uuid": "u1",
            "timestamp": "2026-10-07T22:53:25.123Z",
            "sessionId": "s1",
        }
    )
    items, base_id = parse_entry(0, line, "s1")
    assert items is not None, "expected items, got None"
    assert len(items) == 2, f"expected 2 items, got {len(items)}"

    kinds = [item["metadata"]["kind"] for item in items]
    assert kinds == ["message", "tool_result"], f"expected [message, tool_result], got {kinds}"

    ids = [item["id"] for item in items]
    assert ids == [base_id, f"{base_id}:tool_result:0"], f"ids mismatch: {ids}"

    texts = [item["text"] for item in items]
    assert texts == ["USER TEXT", "out"], f"texts mismatch: {texts}"


def test_multiple_tool_use_ids_distinct():
    """Two tool_use blocks in one assistant entry get :tool_use:0 and :tool_use:1 ids."""
    line = json.dumps(
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}},
                    {"type": "tool_use", "id": "toolu_2", "name": "Bash", "input": {"command": "pwd"}},
                ],
            },
            "uuid": "a1",
            "timestamp": "2026-10-07T22:53:25.123Z",
            "sessionId": "s1",
        }
    )
    items, base_id = parse_entry(0, line, "s1")
    assert items is not None, "expected items, got None"
    assert len(items) == 2, f"expected 2 items, got {len(items)}"

    ids = [item["id"] for item in items]
    assert ids[0].endswith(":tool_use:0"), f"first id should end with :tool_use:0, got {ids[0]}"
    assert ids[1].endswith(":tool_use:1"), f"second id should end with :tool_use:1, got {ids[1]}"
    assert len(set(ids)) == 2, f"ids must be unique, got {ids}"


def test_multiple_tool_result_ids_distinct():
    """Two tool_result blocks in one user entry get :tool_result:0 and :tool_result:1 ids."""
    line = json.dumps(
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "out1"},
                    {"type": "tool_result", "tool_use_id": "toolu_2", "content": "out2"},
                ],
            },
            "uuid": "u1",
            "timestamp": "2026-10-07T22:53:25.123Z",
            "sessionId": "s1",
        }
    )
    items, base_id = parse_entry(0, line, "s1")
    assert items is not None, "expected items, got None"
    assert len(items) == 2, f"expected 2 items, got {len(items)}"

    ids = [item["id"] for item in items]
    assert ids[0].endswith(":tool_result:0"), f"first id should end with :tool_result:0, got {ids[0]}"
    assert ids[1].endswith(":tool_result:1"), f"second id should end with :tool_result:1, got {ids[1]}"
    assert len(set(ids)) == 2, f"ids must be unique, got {ids}"


def test_user_is_meta_true_with_string_content_skipped():
    """User record with isMeta true and string content returns (None, None)."""
    line = json.dumps(
        {
            "type": "user",
            "message": {"role": "user", "content": "harness injection"},
            "uuid": "meta1",
            "isMeta": True,
            "timestamp": "2026-10-07T22:53:25.123Z",
            "sessionId": "s1",
        }
    )
    items, base_id = parse_entry(0, line, "s1")
    assert items is None
    assert base_id is None


def test_user_is_meta_true_with_list_content_skipped():
    """User record with isMeta true and list content returns (None, None)."""
    line = json.dumps(
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "text", "text": "skill body or caveat"},
                ],
            },
            "uuid": "meta2",
            "isMeta": True,
            "timestamp": "2026-10-07T22:53:25.123Z",
            "sessionId": "s1",
        }
    )
    items, base_id = parse_entry(0, line, "s1")
    assert items is None
    assert base_id is None


def test_user_is_meta_false_returns_item():
    """Control: isMeta false returns one message item."""
    line = json.dumps(
        {
            "type": "user",
            "message": {"role": "user", "content": "real user text"},
            "uuid": "meta3",
            "isMeta": False,
            "timestamp": "2026-10-07T22:53:25.123Z",
            "sessionId": "s1",
        }
    )
    items, base_id = parse_entry(0, line, "s1")
    assert items is not None
    assert len(items) == 1
    assert items[0]["metadata"]["kind"] == "message"
    assert items[0]["metadata"]["role"] == "user"


def test_user_is_meta_absent_returns_item():
    """Control: isMeta absent returns one message item."""
    line = json.dumps(
        {
            "type": "user",
            "message": {"role": "user", "content": "real user text"},
            "uuid": "meta4",
            "timestamp": "2026-10-07T22:53:25.123Z",
            "sessionId": "s1",
        }
    )
    items, base_id = parse_entry(0, line, "s1")
    assert items is not None
    assert len(items) == 1
    assert items[0]["metadata"]["kind"] == "message"
    assert items[0]["metadata"]["role"] == "user"


def test_user_is_meta_string_still_returns_item():
    """isMeta as string 'true' is not the literal boolean True."""
    line = json.dumps(
        {
            "type": "user",
            "message": {"role": "user", "content": "real user text"},
            "uuid": "meta5",
            "isMeta": "true",
            "timestamp": "2026-10-07T22:53:25.123Z",
            "sessionId": "s1",
        }
    )
    items, base_id = parse_entry(0, line, "s1")
    assert items is not None
    assert len(items) == 1
    assert items[0]["metadata"]["kind"] == "message"
    assert items[0]["metadata"]["role"] == "user"


def test_assistant_is_meta_true_with_text_returns_item():
    """Assistant record with isMeta true and text content is not skipped (only user records are)."""
    line = json.dumps(
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "assistant response"}]},
            "uuid": "meta1",
            "isMeta": True,
            "timestamp": "2026-10-07T22:53:25.123Z",
            "sessionId": "s1",
        }
    )
    items, base_id = parse_entry(0, line, "s1")
    assert items is not None, "Assistant with isMeta true and text should not be skipped"
    assert len(items) == 1, f"Expected 1 item, got {len(items)}"
    assert items[0]["metadata"]["kind"] == "message", f"Expected kind 'message', got {items[0]['metadata']['kind']}"
    assert items[0]["metadata"]["role"] == "assistant", f"Expected role 'assistant', got {items[0]['metadata']['role']}"
    assert items[0]["text"] == "assistant response", f"Expected text 'assistant response', got {items[0]['text']}"


@pytest.mark.parametrize(
    "label,line",
    [
        (
            "user_string_content",
            r'{"type":"user","message":{"role":"user","content":"\ud800"},"uuid":"u1","timestamp":"2026-10-07T22:53:25.123Z","sessionId":"s1"}',
        ),
        (
            "user_text_block",
            r'{"type":"user","message":{"role":"user","content":[{"type":"text","text":"\ud800"}]},"uuid":"u2","timestamp":"2026-10-07T22:53:25.123Z","sessionId":"s1"}',
        ),
        (
            "tool_result_string_content",
            r'{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"\ud800"}]},"uuid":"u3","timestamp":"2026-10-07T22:53:25.123Z","sessionId":"s1"}',
        ),
        (
            "assistant_text_block",
            r'{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"\ud800"}]},"uuid":"a1","timestamp":"2026-10-07T22:53:25.123Z","sessionId":"s1"}',
        ),
        (
            "uuid",
            r'{"type":"user","message":{"role":"user","content":"plain"},"uuid":"\ud800","timestamp":"2026-10-07T22:53:25.123Z","sessionId":"s1"}',
        ),
    ],
    ids=[
        "surrogate-user-string-content",
        "surrogate-user-text-block",
        "surrogate-tool-result-string-content",
        "surrogate-assistant-text-block",
        "surrogate-uuid",
    ],
)
def test_lone_surrogate_does_not_raise(label, line):
    """A lone UTF-16 surrogate escape in any text field must not raise.

    JSON parsed by json.loads can carry a lone surrogate from JS tools that
    slice strings mid-pair. Such a value must be sanitized before any UTF-8
    encoding, here or in the derived item id.
    """
    session_id = "s1"
    try:
        items, base_id = parse_entry(0, line, session_id)
    except Exception as exc:
        pytest.fail(f"{label}: parse_entry raised {type(exc).__name__}: {exc}")

    assert items is not None, f"{label}: expected one item, got None"
    assert len(items) == 1, f"{label}: expected 1 item, got {len(items)}"

    item = items[0]
    try:
        item["text"].encode("utf-8")
    except UnicodeEncodeError as exc:
        pytest.fail(f"{label}: item text still holds a lone surrogate: {exc}")
    try:
        item["id"].encode("utf-8")
    except UnicodeEncodeError as exc:
        pytest.fail(f"{label}: item id still holds a lone surrogate: {exc}")
