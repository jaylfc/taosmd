"""Tests for catalog_session_id FK column in crystals table."""

import asyncio
import tempfile
from pathlib import Path

from taosmd.crystallize import CrystalStore


def _run(coro):
    return asyncio.run(coro)


def test_crystals_table_has_catalog_session_id():
    async def _check():
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "crystals.db"
            cs = CrystalStore(db_path)
            await cs.init()
            try:
                rows = cs._conn.execute("PRAGMA table_info(crystals)").fetchall()
                col_names = [row[1] for row in rows]
                assert "catalog_session_id" in col_names, (
                    f"catalog_session_id not found in crystals columns: {col_names}"
                )
            finally:
                await cs.close()

    _run(_check())
