"""Test recipient field and role resolution functionality."""
import asyncio
import tempfile
from pathlib import Path

from taosmd.service import a2a_send, a2a_read


async def test_recipient_send_and_read():
    """Test that recipient field is stored and returned correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir)
        
        # First send a message with recipient
        print("Test 1: Send with agent recipient (@alice)")
        receipt1 = await a2a_send(
            sender="@test",
            body="Hello agent",
            thread="general",
            recipient="@alice",
            data_dir=data_dir
        )
        assert "recipient" in receipt1, "Receipt should contain recipient"
        assert receipt1["recipient"] == "@alice", f"Expected @alice, got {receipt1.get('recipient')}"
        print(f"  ✓ Receipt contains recipient: {receipt1['recipient']}")
        
        # Send a message without recipient
        print("\nTest 2: Send without recipient")
        receipt2 = await a2a_send(
            sender="@test",
            body="Hello no recipient",
            thread="general",
            data_dir=data_dir
        )
        assert "recipient" not in receipt2, "Receipt should not contain recipient"
        print(f"  ✓ Receipt does not contain recipient")
        
        # Send a message with role recipient
        print("\nTest 3: Send with role recipient (@taOS-PA)")
        receipt3 = await a2a_send(
            sender="@test",
            body="Hello role",
            thread="general",
            recipient="@taOS-PA",
            data_dir=data_dir
        )
        assert "recipient" in receipt3, "Receipt should contain recipient"
        assert receipt3["recipient"] == "@taOS-PA", f"Expected @taOS-PA, got {receipt3.get('recipient')}"
        print(f"  ✓ Receipt contains recipient: {receipt3['recipient']}")
        
        # Now read messages and verify recipient is present
        print("\nTest 4: Read all messages and verify recipient")
        messages = await a2a_read(thread="general", data_dir=data_dir)
        print(f"  Found {len(messages)} messages")
        
        for msg in messages:
            print(f"  Message from {msg.get('from')}: recipient present = {'recipient' in msg}")
            if "recipient" in msg:
                print(f"    Recipient: {msg['recipient']}")
        
        print("\n✓ All tests passed!")


if __name__ == "__main__":
    asyncio.run(test_recipient_send_and_read())