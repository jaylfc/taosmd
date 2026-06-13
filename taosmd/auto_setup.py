"""Auto-setup for taOSmd: creates data directories, initialises stores,
and optionally installs a daily maintenance cron job.

Usage:
    python -m taosmd.auto_setup          # Interactive setup
    python -m taosmd.auto_setup --yes    # Non-interactive, accept defaults
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_DATA_DIR = os.path.expanduser("~/.taosmd")


def _recommended_store_mode() -> dict:
    """Storage-format fields for the recipe recommended on this hardware.

    Returns the subset of ``config['vector_memory']`` that determines the
    store's format and embedding space (late_interaction, colbert_model,
    embed_model). Falls back to a plain dense MiniLM store on any error so
    setup never fails on the probe.
    """
    try:
        from taosmd import recipes as _recipes
        rec = _recipes.recommend(None)[0]
        rc = rec.retrieval
        return {
            "late_interaction": bool(rc.get("late_interaction", False)),
            "colbert_model": rc.get("colbert_model", "") or "",
            "embed_model": rc.get("embed_model", "minilm-onnx") or "minilm-onnx",
        }
    except Exception:  # noqa: BLE001 - never let setup crash on the probe
        return {"late_interaction": False, "colbert_model": "", "embed_model": "minilm-onnx"}


async def setup(data_dir: str = DEFAULT_DATA_DIR, interactive: bool = True):
    """Full taOSmd setup: creates all stores and verifies they work."""
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

    # 2. Vector Memory. Build it in the recommended storage mode so the store
    # marker matches the mode written to config below; otherwise the first
    # serve open would mode-mismatch a freshly dense-stamped store.
    print("  Setting up Vector Memory...", end="", flush=True)
    from taosmd.vector_memory import VectorMemory
    _mode = _recommended_store_mode()
    vmem = VectorMemory(
        db_path=data_path / "vector-memory.db",
        late_interaction=_mode["late_interaction"],
        colbert_model=_mode["colbert_model"],
    )
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

    # 8. Pending-decisions queue (safety net for low-confidence KG updates)
    print("  Setting up Pending-Decisions queue...", end="", flush=True)
    from taosmd.pending_decisions import PendingDecisionsStore
    pending = PendingDecisionsStore(db_path=data_path / "knowledge-graph.db")
    await pending.init()
    await pending.close()
    print(" ✓")

    # 9. Pre-flight: enricher-model availability
    _preflight_enricher_model(interactive=interactive)

    # 10. Daily nightly librarian cron
    setup_cron = True
    if interactive:
        print(
            "\n  The daily 3 AM cron runs the librarian end-to-end:\n"
            "    - catalogs yesterday's sessions from the archive\n"
            "    - crystallizes session digests + extracts lessons\n"
            "    - writes new triples into the knowledge graph\n"
            "    - defers low-confidence contradictions to the pending queue\n"
            "    - compresses old archive files to gzip"
        )
        resp = input("\n  Install daily librarian cron job? [Y/n]: ").strip().lower()
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
            # Seed the store's storage-format mode from the recipe recommended
            # for this hardware, so a fresh install actually builds the store
            # in the mode it will be told to use (e.g. an 8 GB GPU box gets the
            # late-interaction store the lateint recipe asks for, not a dense
            # store the recipe cannot retrofit).
            **_recommended_store_mode(),
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
    """Install the daily nightly-librarian cron job.

    The cron runs: archive compression -> CatalogPipeline.index_yesterday()
    (catalogs yesterday's archive files, crystallizes session digests +
    lessons, writes new KG triples, defers low-confidence contradictions
    to the pending-decisions queue).
    """
    # All path/user values are passed through shlex.quote so a data_dir
    # containing shell metacharacters (spaces, $, ;, &&, backticks, etc.)
    # cannot break out of the command line that /bin/sh runs for cron. The
    # quoted dir is assigned to a shell variable once and the Python snippet
    # reads it from the TAOSMD_DATA_DIR environment variable, so the path is
    # never re-interpolated inside the Python string literals.
    q_dir = shlex.quote(data_dir)
    py_snippet = (
        "import asyncio, os; "
        "from os.path import join as j; "
        "from taosmd.archive import ArchiveStore; "
        "from taosmd.catalog_pipeline import CatalogPipeline; "
        "d = os.environ['TAOSMD_DATA_DIR']; "
        "a = ArchiveStore(archive_dir=j(d, 'archive'), index_path=j(d, 'archive-index.db')); "
        "asyncio.run(a.init()); asyncio.run(a.compress_old_files()); asyncio.run(a.close()); "
        "p = CatalogPipeline(archive_dir=j(d, 'archive'), sessions_dir=j(d, 'sessions'), "
        "catalog_db=j(d, 'session-catalog.db'), crystals_db=j(d, 'crystals.db'), "
        "kg_db=j(d, 'knowledge-graph.db')); "
        "asyncio.run(p.init()); asyncio.run(p.index_yesterday()); asyncio.run(p.close())"
    )
    cron_cmd = (
        f"0 3 * * * cd {q_dir} && TAOSMD_DATA_DIR={q_dir} "
        f"python3 -c {shlex.quote(py_snippet)} 2>/dev/null"
    )

    try:
        existing = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        if "taosmd" in existing.stdout.lower() or "compress_old_files" in existing.stdout:
            print("  ✓ Daily librarian cron already installed")
            return

        new_cron = existing.stdout.rstrip() + f"\n# taOSmd nightly librarian (catalog + crystallize + KG + compress)\n{cron_cmd}\n"
        proc = subprocess.run(["crontab", "-"], input=new_cron, capture_output=True, text=True)
        if proc.returncode == 0:
            print("  ✓ Daily librarian cron installed (3 AM nightly)")
        else:
            print(f"  ⚠ Cron install failed: {proc.stderr.strip()}")
    except FileNotFoundError:
        print("  ⚠ crontab not available, skip nightly librarian install")


# Recommended enricher models per backend. The librarian / CatalogPipeline
# probes Ollama (11434) first, then rkllama/qmd on RK3588 NPU (7832 or 8080).
# Models are different formats per backend. Ollama uses GGUF via ``ollama
# pull``; rkllama uses RKLLM files downloaded from HuggingFace.
_RECOMMENDED_OLLAMA_MODELS = [
    "qwen3.5:9b",           # 12 GB GPU tier (production leader)
    "llama3.1:8b",          # 12 GB GPU tier (alt / EDU + KG extractor)
    "qwen3:4b",             # 4-8 GB tier (low-end CPU / small GPU)
    "gemma4:e2b",           # judge + small-tier generator
    "gemma4:e4b",           # mid-tier generator
]

# RK3588 NPU recipe per scripts/setup.sh. Model lives in
# ~/.rkllama/models/qwen3-4b-chat after huggingface-cli download.
_RECOMMENDED_RKLLAMA_MODEL = "qwen3-4b-chat (Qwen3-4B-rk3588-w8a8-opt-1-hybrid-ratio-0.0.rkllm)"
_RKLLAMA_PULL_CMD = (
    "huggingface-cli download dulimov/Qwen3-4B-rk3588-1.2.1-base "
    "Qwen3-4B-rk3588-w8a8-opt-1-hybrid-ratio-0.0.rkllm "
    "--local-dir ~/.rkllama/models/qwen3-4b-chat"
)


def _preflight_enricher_model(
    ollama_url: str = "http://localhost:11434",
    rkllama_urls: tuple[str, ...] = ("http://localhost:7832", "http://localhost:8080"),
    interactive: bool = True,
) -> None:
    """Check that at least one recommended enricher model is reachable.

    Probes Ollama first (GPU/CPU users, the common case), then rkllama/qmd
    on the standard RK3588 NPU ports (Orange Pi / Rock 5 / Radxa users).
    Reports which backend has which recommended models; warns with the
    right pull command per backend if nothing recommended is installed.

    Doesn't block setup, only warns. The librarian's ResourceManager will
    fall back to whatever IS available, but with no installed model the
    nightly crystallize step degrades to a no-op and the user wouldn't
    know without reading the cron logs.
    """
    import urllib.error
    import urllib.request

    print("  Checking enricher model availability...", end="", flush=True)

    ollama_recs: list[str] = []
    ollama_reachable = False
    try:
        with urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=3) as resp:
            data = json.load(resp)
        ollama_reachable = True
        installed_full = {m.get("name") for m in data.get("models", [])}
        ollama_recs = [r for r in _RECOMMENDED_OLLAMA_MODELS if r in installed_full]
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        ollama_reachable = False

    rkllama_url: str | None = None
    for url in rkllama_urls:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as resp:
                if resp.status == 200:
                    rkllama_url = url
                    break
        except (urllib.error.URLError, TimeoutError):
            continue

    if ollama_recs:
        print(f" ✓ Ollama: {len(ollama_recs)} recommended found "
              f"({', '.join(ollama_recs)})")
        if rkllama_url:
            print(f"  ✓ rkllama also reachable at {rkllama_url} "
                  "(NPU enrichment path available)")
        return
    if rkllama_url:
        print(f" ✓ rkllama reachable at {rkllama_url} "
              "(NPU enrichment path available)")
        if not ollama_reachable:
            print("    Note: Ollama is NOT reachable. The librarian will use "
                  "the NPU backend for enrichment.")
        return

    print()  # break from the "..." line
    print("  ⚠ No recommended enricher model is reachable.")
    if ollama_reachable:
        print(f"     Ollama IS reachable at {ollama_url} but has no recommended model installed.")
        print("     For GPU / CPU hardware, pick one matching your tier:")
        for m in _RECOMMENDED_OLLAMA_MODELS:
            print(f"       ollama pull {m}")
    else:
        print(f"     Ollama not reachable at {ollama_url}.")
        print("     For GPU / CPU: install Ollama (https://ollama.com/install)")
        print("     then pull one of:")
        for m in _RECOMMENDED_OLLAMA_MODELS:
            print(f"       ollama pull {m}")
    print()
    print("     For RK3588 NPU (Orange Pi / Rock 5 / Radxa): use rkllama instead.")
    print(f"     Recommended: {_RECOMMENDED_RKLLAMA_MODEL}")
    print(f"       {_RKLLAMA_PULL_CMD}")
    print()
    print("     Setup will continue. The librarian falls back to whatever")
    print("     IS available, but crystallize / contradiction-detect quality")
    print("     drops without a recommended model.")

    if interactive:
        resp = input("    Continue anyway? [Y/n]: ").strip().lower()
        if resp == "n":
            print("    Aborting setup. Re-run `python -m taosmd.auto_setup` after pulling a model.")
            sys.exit(0)


if __name__ == "__main__":
    interactive = "--yes" not in sys.argv
    asyncio.run(setup(interactive=interactive))
