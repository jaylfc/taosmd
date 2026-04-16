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

    # 5. Session Catalog
    print("  Setting up Session Catalog...", end="", flush=True)
    from taosmd.session_catalog import SessionCatalog
    catalog = SessionCatalog(
        db_path=data_path / "session-catalog.db",
        archive_dir=data_path / "archive",
        sessions_dir=data_path / "sessions",
    )
    await catalog.init()
    await catalog.close()
    print(" ✓")

    # 6. Crystal Store
    print("  Setting up Crystal Store...", end="", flush=True)
    from taosmd.crystallize import CrystalStore
    cs = CrystalStore(db_path=data_path / "crystals.db")
    await cs.init()
    await cs.close()
    print(" ✓")

    # 7. Insight Store
    print("  Setting up Insight Store...", end="", flush=True)
    from taosmd.reflect import InsightStore
    insights = InsightStore(db_path=data_path / "insights.db")
    await insights.init()
    await insights.close()
    print(" ✓")

    # 8. Daily maintenance cron
    setup_cron = True
    if interactive:
        resp = input("\n  Install daily maintenance cron job? (compresses old archives) [Y/n]: ").strip().lower()
        setup_cron = resp != "n"

    if setup_cron:
        _install_cron(data_dir)

    # Create config file
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
            "llm_url": "http://localhost:11434",  # Ollama (GPU/CPU) or rkllama (NPU)
        },
        "pipeline": {
            "schedule": "0 3 * * *",
            "crystallize_enabled": True,
            "enricher_model": "auto",  # ResourceManager picks best available
        },
        "retrieval": {
            "default_strategy": "thorough",
            "secret_filter_mode": "redact",
        },
    }
    config_path = data_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n  Config saved to {config_path}")

    print(f"\n✓ taOSmd is ready at {data_path}")
    print(f"  Archive:  {data_path}/archive/ (append-only daily JSONL)")
    print(f"  KG:       {data_path}/knowledge-graph.db")
    print(f"  Vectors:  {data_path}/vector-memory.db")
    print(f"  Catalog:  {data_path}/session-catalog.db")
    print(f"  Crystals: {data_path}/crystals.db")
    print(f"  Config:   {config_path}")


def _install_cron(data_dir: str):
    """Install a daily cron job for archive compression."""
    cron_cmd = f'0 3 * * * cd {data_dir} && python3 -c "import asyncio; from taosmd.archive import ArchiveStore; from taosmd.catalog_pipeline import CatalogPipeline; a = ArchiveStore(archive_dir=\\"{data_dir}/archive\\", index_path=\\"{data_dir}/archive-index.db\\"); asyncio.run(a.init()); asyncio.run(a.compress_old_files()); asyncio.run(a.close()); p = CatalogPipeline(archive_dir=\\"{data_dir}/archive\\", sessions_dir=\\"{data_dir}/sessions\\", catalog_db=\\"{data_dir}/session-catalog.db\\", crystals_db=\\"{data_dir}/crystals.db\\", kg_db=\\"{data_dir}/knowledge-graph.db\\"); asyncio.run(p.init()); asyncio.run(p.index_yesterday()); asyncio.run(p.close())" 2>/dev/null'

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
