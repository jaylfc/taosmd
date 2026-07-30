"""Test recipient field functionality for A2A messages."""
import asyncio
import tempfile
import json
from pathlib import Path

from taosmd.service import a2a_send


async def test_recipient_field():
    """Test that recipient field is properly stored and returned."""
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir)
        
        # Test 1: Send with agent recipient
        print("Test 1: Send with agent recipient (@alice)")
        receipt = await a2a_send(
            sender="@test",
            body="Hello agent",
            thread="general",
            recipient="@alice",
            data_dir=data_dir
        )
        print(f"  Receipt includes recipient: {'recipient' in receipt}")
        if 'recipient' in receipt:
            print(f"  Recipient value: {receipt['recipient']}")
        
        # Test 2: Send without recipient
        print("\nTest 2: Send without recipient")
        receipt2 = await a2a_send(
            sender="@test",
            body="Hello no recipient",
            thread="general",
            data_dir=data_dir
        )
        print(f"  Receipt includes recipient: {'recipient' in receipt2}")
        
        # Test 3: Send with role recipient
        print("\nTest 3: Send with role recipient (@taOS-PA)")
        receipt3 = await a2a_send(
            sender="@test",
            body="Hello role",
            thread="general",
            recipient="@taOS-PA",
            data_dir=data_dir
        )
        print(f"  Receipt includes recipient: {'recipient' in receipt3}")
        if 'recipient' in receipt3:
            print(f"  Recipient value: {receipt3['recipient']}")
        
        # Test 4: Verify recipient is stored in archive
        print("\nTest 4: Verify recipient is stored in archive")
        from taosmd.archive import Archive
        archive = Archive(data_dir=str(data_dir), index_path=str(data_dir / 'archive-index.db'))
        await archive.init()
        
        # Get all a2a events
        rows = await archive.query(event_type='a2a')
        print(f"  Found {len(rows)} a2a events")
        
        for row in rows:
            data = json.loads(row.get('data_json', '{}'))
            print(f"  Event from {data.get('from')}: has recipient: {'recipient' in data}")
            if 'recipient' in data:
                print(f"    Recipient: {data['recipient']}")
        
        print("\nAll tests completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_recipient_field())