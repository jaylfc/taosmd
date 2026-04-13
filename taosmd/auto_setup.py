"""Auto-setup for taOSmd — creates data directories, initialises stores,
and optionally installs a daily maintenance cron job.

Usage:
    python -m taosmd.auto_setup          # Interactive setup
    python -m taosmd.auto_setup --yes    # Non-interactive, accept defaults
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_DATA_DIR = os.path.expanduser("~/.taosmd")


async def setup(data_dir: str = DEFAULT_DATA_DIR, interactive: bool = True):
    """Full taOSmd setup — creates all stores and verifies they work."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    print(f"taOSmd data directory: {data_path}")

    # 1. Knowledge Graph
    print("  Setting up Knowledge Graph...", end="", flush=True)
    from taosmd.knowledge_graph import TemporalKnowledgeGraph
    kg = TemporalKnowledgeGraph(db_path=data_path / "knowledge-graph.db")
    await kg.init()
    await kg.close()
    print(" ✓")

    # 2. Vector Memory
    print("  Setting up Vector Memory...", end="", flush=True)
    from taosmd.vector_memory import VectorMemory
    vmem = VectorMemory(db_path=data_path / "vector-memory.db")
    await vmem.init()
    await vmem.close()
    print(" ✓")

    # 3. Zero-Loss Archive
    print("  Setting up Zero-Loss Archive...", end="", flush=True)
    from taosmd.archive import ArchiveStore
    archive = ArchiveStore(
        archive_dir=data_path / "archive",
        index_path=data_path / "archive-index.db",
    )
    await archive.init()
    # Record the setup event as the first archive entry
    await archive.record(
        "system",
        {"event": "taosmd_setup", "data_dir": str(data_path)},
        summary="taOSmd initial setup completed",
    )
    await archive.close()
    print(" ✓")

    # 4. Browsing History
    print("  Setting up Browsing History...", end="", flush=True)
    from taosmd.browsing_history import BrowsingHistoryStore
    history = BrowsingHistoryStore(db_path=data_path / "browsing-history.db")
    await history.init()
    await history.close()
    print(" ✓")

    # 5. Daily maintenance cron
    setup_cron = True
    if interactive:
        resp = input("\n  Install daily maintenance cron job? (compresses old archives) [Y/n]: ").strip().lower()
        setup_cron = resp != "n"

    if setup_cron:
        _install_cron(data_dir)

    # 6. Create config file
    config = {
        "data_dir": str(data_path),
        "archive": {
            "enabled": True,
            "user_tracking": False,
            "retention_days": -1,  # Keep forever (zero-loss)
            "compress_after_days": 1,
        },
        "vector_memory": {
            "embed_mode": "onnx",
            "hybrid_search": True,
        },
        "extraction": {
            "regex_enabled": True,
            "llm_enabled": True,
            "llm_url": "http://localhost:8080",  # rkllama or ollama
        },
    }
    config_path = data_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n  Config saved to {config_path}")

    print(f"\n✓ taOSmd is ready at {data_path}")
    print(f"  Archive: {data_path}/archive/ (append-only daily JSONL)")
    print(f"  KG:      {data_path}/knowledge-graph.db")
    print(f"  Vectors: {data_path}/vector-memory.db")
    print(f"  Config:  {config_path}")


def _install_cron(data_dir: str):
    """Install a daily cron job for archive compression."""
    cron_cmd = f'0 3 * * * cd {data_dir} && python3 -c "import asyncio; from taosmd.archive import ArchiveStore; a = ArchiveStore(archive_dir=\\"{data_dir}/archive\\", index_path=\\"{data_dir}/archive-index.db\\"); asyncio.run(a.init()); asyncio.run(a.compress_old_files()); asyncio.run(a.close())" 2>/dev/null'

    try:
        # Check if cron job already exists
        existing = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        if "taosmd" in existing.stdout.lower() or "compress_old_files" in existing.stdout:
            print("  ✓ Daily cron already installed")
            return

        # Add cron job
        new_cron = existing.stdout.rstrip() + f"\n# taOSmd daily archive compression\n{cron_cmd}\n"
        proc = subprocess.run(["crontab", "-"], input=new_cron, capture_output=True, text=True)
        if proc.returncode == 0:
            print("  ✓ Daily cron installed (runs at 3 AM, compresses old archives)")
        else:
            print(f"  ⚠ Cron install failed: {proc.stderr.strip()}")
    except FileNotFoundError:
        print("  ⚠ crontab not available — skip daily compression")


if __name__ == "__main__":
    interactive = "--yes" not in sys.argv
    asyncio.run(setup(interactive=interactive))
