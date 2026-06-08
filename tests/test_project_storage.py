"""Tests for project-scoped storage — project parameter on ingest/search."""
from __future__ import annotations

import pytest

from taosmd.api import ingest, search


@pytest.fixture
def data_dir(tmp_path):
    """Return a fresh taosmd data dir for each test."""
    return str(tmp_path / "taosmd-data")


# --- ingest with project -----------------------------------------------------


class TestIngestProject:
    @pytest.mark.asyncio
    async def test_ingest_stores_project_metadata(self, data_dir):
        result = await ingest(
            "The login page uses OAuth2",
            agent="claude",
            project="abc123",
            data_dir=data_dir,
        )
        assert result["project"] == "abc123"
        assert result["archived"] == 1

    @pytest.mark.asyncio
    async def test_ingest_without_project(self, data_dir):
        result = await ingest(
            "Some fact",
            agent="claude",
            data_dir=data_dir,
        )
        assert result["project"] is None

    @pytest.mark.asyncio
    async def test_ingest_different_agents_same_project(self, data_dir):
        await ingest("Fact from Claude", agent="claude", project="proj1", data_dir=data_dir)
        await ingest("Fact from Kilo", agent="kilo", project="proj1", data_dir=data_dir)
        # Both should succeed — no conflict
        hits = await search("Fact", agent="claude", project="proj1", data_dir=data_dir)
        assert len(hits) >= 1


# --- search with project -----------------------------------------------------


class TestSearchProject:
    @pytest.mark.asyncio
    async def test_search_scoped_to_project(self, data_dir):
        await ingest("OAuth2 on login page", agent="claude", project="proj-a", data_dir=data_dir)
        await ingest("OAuth2 on signup page", agent="claude", project="proj-b", data_dir=data_dir)

        hits_a = await search("OAuth2", agent="claude", project="proj-a", data_dir=data_dir)
        hits_b = await search("OAuth2", agent="claude", project="proj-b", data_dir=data_dir)

        # Each project should only see its own memories
        texts_a = [h["text"] for h in hits_a]
        texts_b = [h["text"] for h in hits_b]
        assert any("login" in t for t in texts_a)
        assert any("signup" in t for t in texts_b)

    @pytest.mark.asyncio
    async def test_search_without_project_sees_all(self, data_dir):
        await ingest("Fact A", agent="claude", project="proj-a", data_dir=data_dir)
        await ingest("Fact B", agent="claude", project="proj-b", data_dir=data_dir)

        hits = await search("Fact", agent="claude", data_dir=data_dir)
        assert len(hits) >= 2

    @pytest.mark.asyncio
    async def test_cross_agent_search_with_also_include(self, data_dir):
        await ingest("Database schema uses UUIDs", agent="claude", project="proj1", data_dir=data_dir)
        await ingest("API returns JSON:API format", agent="kilo", project="proj1", data_dir=data_dir)

        # Claude searches, including Kilo's memories
        hits = await search(
            "API format",
            agent="claude",
            project="proj1",
            also_include=["kilo"],
            data_dir=data_dir,
        )
        texts = [h["text"] for h in hits]
        assert any("JSON:API" in t for t in texts)

    @pytest.mark.asyncio
    async def test_cross_agent_excludes_other_projects(self, data_dir):
        await ingest("Claude fact in proj1", agent="claude", project="proj1", data_dir=data_dir)
        await ingest("Kilo fact in proj2", agent="kilo", project="proj2", data_dir=data_dir)

        hits = await search(
            "fact",
            agent="claude",
            project="proj1",
            also_include=["kilo"],
            data_dir=data_dir,
        )
        texts = [h["text"] for h in hits]
        # Should NOT see Kilo's proj2 fact
        assert not any("proj2" in t for t in texts)
