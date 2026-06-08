"""Tests for librarian discovery: list_projects and list_shelves."""
from __future__ import annotations

import pytest

from taosmd.api import ingest, list_projects, list_shelves


@pytest.fixture
def data_dir(tmp_path):
    return str(tmp_path / "taosmd-data")


class TestListProjects:
    @pytest.mark.asyncio
    async def test_empty_when_no_projects(self, data_dir):
        result = await list_projects(data_dir=data_dir)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_projects(self, data_dir):
        await ingest("fact A", agent="claude", project="proj1", data_dir=data_dir)
        await ingest("fact B", agent="kilo", project="proj2", data_dir=data_dir)

        result = await list_projects(data_dir=data_dir)
        ids = [p["project_id"] for p in result]
        assert "proj1" in ids
        assert "proj2" in ids

    @pytest.mark.asyncio
    async def test_groups_agents_per_project(self, data_dir):
        await ingest("fact from claude", agent="claude", project="proj1", data_dir=data_dir)
        await ingest("fact from kilo", agent="kilo", project="proj1", data_dir=data_dir)

        result = await list_projects(data_dir=data_dir)
        proj1 = next(p for p in result if p["project_id"] == "proj1")
        assert "claude" in proj1["agents"]
        assert "kilo" in proj1["agents"]

    @pytest.mark.asyncio
    async def test_excludes_non_project_memories(self, data_dir):
        await ingest("no project here", agent="claude", data_dir=data_dir)
        await ingest("has project", agent="kilo", project="proj1", data_dir=data_dir)

        result = await list_projects(data_dir=data_dir)
        assert len(result) == 1
        assert result[0]["project_id"] == "proj1"


class TestListShelves:
    @pytest.mark.asyncio
    async def test_empty_project(self, data_dir):
        result = await list_shelves(project="nonexistent", data_dir=data_dir)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_agents_in_project(self, data_dir):
        await ingest("fact from claude", agent="claude", project="proj1", data_dir=data_dir)
        await ingest("fact from kilo", agent="kilo", project="proj1", data_dir=data_dir)

        result = await list_shelves(project="proj1", data_dir=data_dir)
        agents = [s["agent"] for s in result]
        assert "claude" in agents
        assert "kilo" in agents

    @pytest.mark.asyncio
    async def test_fact_counts(self, data_dir):
        await ingest("fact 1", agent="claude", project="proj1", data_dir=data_dir)
        await ingest("fact 2", agent="claude", project="proj1", data_dir=data_dir)
        await ingest("fact 3", agent="kilo", project="proj1", data_dir=data_dir)

        result = await list_shelves(project="proj1", data_dir=data_dir)
        by_agent = {s["agent"]: s["facts"] for s in result}
        assert by_agent["claude"] == 2
        assert by_agent["kilo"] == 1

    @pytest.mark.asyncio
    async def test_excludes_other_projects(self, data_dir):
        await ingest("proj1 fact", agent="claude", project="proj1", data_dir=data_dir)
        await ingest("proj2 fact", agent="kilo", project="proj2", data_dir=data_dir)

        result = await list_shelves(project="proj1", data_dir=data_dir)
        agents = [s["agent"] for s in result]
        assert "claude" in agents
        assert "kilo" not in agents
