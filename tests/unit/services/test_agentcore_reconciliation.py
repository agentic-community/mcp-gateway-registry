"""
Unit tests for AgentCore federation reconciliation functions.

Tests cover:
- _build_expected_agentcore_paths: disabled vs enabled config
- _reconcile_agentcore_servers: stale removal, no stale, errors
- _reconcile_agentcore_agents: stale removal by tag+prefix filter
- _reconcile_agentcore_skills: stale removal by tag+prefix filter
- reconcile_agentcore_records: dry run, full run, None synced_paths
"""

from types import SimpleNamespace
from unittest.mock import (
    AsyncMock,
    patch,
)

import pytest

from registry.schemas.federation_schema import (
    AgentCoreFederationConfig,
    FederationConfig,
)
from registry.services.federation_reconciliation import (
    _build_expected_agentcore_paths,
    _reconcile_agentcore_agents,
    _reconcile_agentcore_servers,
    _reconcile_agentcore_skills,
    reconcile_agentcore_records,
)

# =============================================================================
# Helper: create mock agent/skill objects
# =============================================================================


def _make_agent(
    name: str,
    path: str,
    tags: list[str] | None = None,
) -> SimpleNamespace:
    """Create a mock agent object with name, path, tags."""
    return SimpleNamespace(name=name, path=path, tags=tags or [])


def _make_skill(
    name: str,
    path: str,
    tags: list[str] | None = None,
) -> SimpleNamespace:
    """Create a mock skill object with name, path, tags."""
    return SimpleNamespace(name=name, path=path, tags=tags or [])


# =============================================================================
# _build_expected_agentcore_paths Tests
# =============================================================================


@pytest.mark.unit
class TestBuildExpectedAgentcorePaths:
    """Tests for _build_expected_agentcore_paths."""

    def test_disabled_config_returns_empty_sets(self):
        """When agentcore is disabled, all sets should be empty."""
        config = FederationConfig(
            agentcore=AgentCoreFederationConfig(enabled=False),
        )
        synced = {
            "servers": {"/s1", "/s2"},
            "agents": {"/a1"},
            "skills": {"/sk1"},
        }
        result = _build_expected_agentcore_paths(config, synced)
        assert result["servers"] == set()
        assert result["agents"] == set()
        assert result["skills"] == set()

    def test_enabled_config_passes_through_synced_paths(self):
        """When agentcore is enabled, synced paths should be returned."""
        config = FederationConfig(
            agentcore=AgentCoreFederationConfig(enabled=True),
        )
        synced = {
            "servers": {"/server-a", "/server-b"},
            "agents": {"/agents/agentcore-x"},
            "skills": {"/skills/agentcore-y"},
        }
        result = _build_expected_agentcore_paths(config, synced)
        assert result["servers"] == {"/server-a", "/server-b"}
        assert result["agents"] == {"/agents/agentcore-x"}
        assert result["skills"] == {"/skills/agentcore-y"}

    def test_enabled_config_missing_keys_default_to_empty(self):
        """Missing keys in synced_paths should default to empty sets."""
        config = FederationConfig(
            agentcore=AgentCoreFederationConfig(enabled=True),
        )
        synced = {"servers": {"/s1"}}
        result = _build_expected_agentcore_paths(config, synced)
        assert result["servers"] == {"/s1"}
        assert result["agents"] == set()
        assert result["skills"] == set()


# =============================================================================
# _reconcile_agentcore_servers Tests
# =============================================================================


@pytest.mark.unit
class TestReconcileAgentcoreServers:
    """Tests for _reconcile_agentcore_servers."""

    @pytest.mark.asyncio
    async def test_no_stale_servers(self):
        """When all actual servers are expected, nothing is removed."""
        server_repo = AsyncMock()
        server_repo.list_by_source.return_value = {
            "/s1": {"server_name": "Server 1"},
        }
        server_service = AsyncMock()

        result = await _reconcile_agentcore_servers(
            expected_paths={"/s1"},
            server_service=server_service,
            server_repo=server_repo,
        )
        assert result["removed"] == []
        assert result["errors"] == []
        server_service.remove_server.assert_not_called()

    @pytest.mark.asyncio
    async def test_stale_servers_removed(self):
        """Stale servers (in DB but not expected) should be removed."""
        server_repo = AsyncMock()
        server_repo.list_by_source.return_value = {
            "/s1": {"server_name": "Server 1"},
            "/s2": {"server_name": "Server 2"},
            "/s3": {"server_name": "Server 3"},
        }
        server_service = AsyncMock()
        server_service.remove_server.return_value = True

        result = await _reconcile_agentcore_servers(
            expected_paths={"/s1"},
            server_service=server_service,
            server_repo=server_repo,
        )
        assert set(result["removed"]) == {"Server 2", "Server 3"}
        assert result["errors"] == []

    @pytest.mark.asyncio
    async def test_removal_failure_records_error(self):
        """When remove_server returns False, an error is recorded."""
        server_repo = AsyncMock()
        server_repo.list_by_source.return_value = {
            "/s1": {"server_name": "Server 1"},
        }
        server_service = AsyncMock()
        server_service.remove_server.return_value = False

        result = await _reconcile_agentcore_servers(
            expected_paths=set(),
            server_service=server_service,
            server_repo=server_repo,
        )
        assert result["removed"] == []
        assert len(result["errors"]) == 1

    @pytest.mark.asyncio
    async def test_removal_exception_records_error(self):
        """When remove_server raises an exception, an error is recorded."""
        server_repo = AsyncMock()
        server_repo.list_by_source.return_value = {
            "/s1": {"server_name": "Server 1"},
        }
        server_service = AsyncMock()
        server_service.remove_server.side_effect = RuntimeError("db failure")

        result = await _reconcile_agentcore_servers(
            expected_paths=set(),
            server_service=server_service,
            server_repo=server_repo,
        )
        assert result["removed"] == []
        assert len(result["errors"]) == 1
        assert "db failure" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_no_agentcore_servers_in_db(self):
        """When no agentcore servers exist in DB, nothing happens."""
        server_repo = AsyncMock()
        server_repo.list_by_source.return_value = {}
        server_service = AsyncMock()

        result = await _reconcile_agentcore_servers(
            expected_paths=set(),
            server_service=server_service,
            server_repo=server_repo,
        )
        assert result["removed"] == []
        assert result["errors"] == []


# =============================================================================
# _reconcile_agentcore_agents Tests
# =============================================================================


@pytest.mark.unit
class TestReconcileAgentcoreAgents:
    """Tests for _reconcile_agentcore_agents."""

    @pytest.mark.asyncio
    async def test_no_stale_agents(self):
        """When all agentcore agents are expected, nothing is removed."""
        agent_repo = AsyncMock()
        agent_repo.list_all.return_value = [
            _make_agent("Agent A", "/agents/agentcore-a", tags=["agentcore"]),
        ]

        result = await _reconcile_agentcore_agents(
            expected_paths={"/agents/agentcore-a"},
            agent_repo=agent_repo,
        )
        assert result["removed"] == []
        assert result["errors"] == []
        agent_repo.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_stale_agents_removed(self):
        """Stale agentcore agents should be removed."""
        agent_repo = AsyncMock()
        agent_repo.list_all.return_value = [
            _make_agent("Agent A", "/agents/agentcore-a", tags=["agentcore"]),
            _make_agent("Agent B", "/agents/agentcore-b", tags=["agentcore"]),
        ]
        agent_repo.delete.return_value = True

        result = await _reconcile_agentcore_agents(
            expected_paths={"/agents/agentcore-a"},
            agent_repo=agent_repo,
        )
        assert result["removed"] == ["Agent B"]
        assert result["errors"] == []
        agent_repo.delete.assert_called_once_with("/agents/agentcore-b")

    @pytest.mark.asyncio
    async def test_non_agentcore_agents_ignored(self):
        """Agents without 'agentcore' tag or wrong path prefix are ignored."""
        agent_repo = AsyncMock()
        agent_repo.list_all.return_value = [
            _make_agent("Normal Agent", "/agents/my-agent", tags=["production"]),
            _make_agent("Tagged Wrong Path", "/agents/other-agent", tags=["agentcore"]),
            _make_agent("Right Path No Tag", "/agents/agentcore-x", tags=["other"]),
        ]

        result = await _reconcile_agentcore_agents(
            expected_paths=set(),
            agent_repo=agent_repo,
        )
        assert result["removed"] == []
        agent_repo.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_failure_records_error(self):
        """When agent_repo.delete returns False, an error is recorded."""
        agent_repo = AsyncMock()
        agent_repo.list_all.return_value = [
            _make_agent("Agent A", "/agents/agentcore-a", tags=["agentcore"]),
        ]
        agent_repo.delete.return_value = False

        result = await _reconcile_agentcore_agents(
            expected_paths=set(),
            agent_repo=agent_repo,
        )
        assert result["removed"] == []
        assert len(result["errors"]) == 1


# =============================================================================
# _reconcile_agentcore_skills Tests
# =============================================================================


@pytest.mark.unit
class TestReconcileAgentcoreSkills:
    """Tests for _reconcile_agentcore_skills."""

    @pytest.mark.asyncio
    async def test_no_stale_skills(self):
        """When all agentcore skills are expected, nothing is removed."""
        skill_repo = AsyncMock()
        skill_repo.list_all.return_value = [
            _make_skill("Skill X", "/skills/agentcore-x", tags=["agentcore"]),
        ]

        result = await _reconcile_agentcore_skills(
            expected_paths={"/skills/agentcore-x"},
            skill_repo=skill_repo,
        )
        assert result["removed"] == []
        assert result["errors"] == []
        skill_repo.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_stale_skills_removed(self):
        """Stale agentcore skills should be removed."""
        skill_repo = AsyncMock()
        skill_repo.list_all.return_value = [
            _make_skill("Skill X", "/skills/agentcore-x", tags=["agentcore"]),
            _make_skill("Skill Y", "/skills/agentcore-y", tags=["agentcore"]),
        ]
        skill_repo.delete.return_value = True

        result = await _reconcile_agentcore_skills(
            expected_paths={"/skills/agentcore-x"},
            skill_repo=skill_repo,
        )
        assert result["removed"] == ["Skill Y"]
        assert result["errors"] == []

    @pytest.mark.asyncio
    async def test_non_agentcore_skills_ignored(self):
        """Skills without 'agentcore' tag or wrong path prefix are ignored."""
        skill_repo = AsyncMock()
        skill_repo.list_all.return_value = [
            _make_skill("Normal Skill", "/skills/my-skill", tags=["production"]),
            _make_skill("Tagged Wrong Path", "/skills/other", tags=["agentcore"]),
            _make_skill("Right Path No Tag", "/skills/agentcore-z", tags=["other"]),
        ]

        result = await _reconcile_agentcore_skills(
            expected_paths=set(),
            skill_repo=skill_repo,
        )
        assert result["removed"] == []
        skill_repo.delete.assert_not_called()


# =============================================================================
# reconcile_agentcore_records Tests
# =============================================================================


@pytest.mark.unit
class TestReconcileAgentcoreRecords:
    """Tests for reconcile_agentcore_records orchestrator."""

    @pytest.mark.asyncio
    async def test_dry_run_skips_removal(self):
        """dry_run=True should return without deleting anything."""
        config = FederationConfig(
            agentcore=AgentCoreFederationConfig(enabled=True),
        )
        result = await reconcile_agentcore_records(
            config=config,
            server_service=AsyncMock(),
            server_repo=AsyncMock(),
            agent_repo=AsyncMock(),
            skill_repo=AsyncMock(),
            synced_paths={"servers": set(), "agents": set(), "skills": set()},
            dry_run=True,
        )
        assert result["dry_run"] is True

    @pytest.mark.asyncio
    async def test_none_synced_paths_defaults_to_empty(self):
        """When synced_paths is None, it should default to empty sets."""
        config = FederationConfig(
            agentcore=AgentCoreFederationConfig(enabled=True),
        )
        server_repo = AsyncMock()
        server_repo.list_by_source.return_value = {}
        agent_repo = AsyncMock()
        agent_repo.list_all.return_value = []
        skill_repo = AsyncMock()
        skill_repo.list_all.return_value = []

        with patch("registry.services.federation_reconciliation._record_reconciliation_metrics"):
            result = await reconcile_agentcore_records(
                config=config,
                server_service=AsyncMock(),
                server_repo=server_repo,
                agent_repo=agent_repo,
                skill_repo=skill_repo,
                synced_paths=None,
                dry_run=False,
            )
        assert result["dry_run"] is False
        assert result["total_removed"] == 0

    @pytest.mark.asyncio
    async def test_full_run_removes_stale_records(self):
        """Full run should remove stale servers, agents, and skills."""
        config = FederationConfig(
            agentcore=AgentCoreFederationConfig(enabled=True),
        )

        # Server repo: one stale server
        server_repo = AsyncMock()
        server_repo.list_by_source.return_value = {
            "/stale-server": {"server_name": "Stale Server"},
        }
        server_service = AsyncMock()
        server_service.remove_server.return_value = True

        # Agent repo: one stale agent
        agent_repo = AsyncMock()
        agent_repo.list_all.return_value = [
            _make_agent("Stale Agent", "/agents/agentcore-old", tags=["agentcore"]),
        ]
        agent_repo.delete.return_value = True

        # Skill repo: one stale skill
        skill_repo = AsyncMock()
        skill_repo.list_all.return_value = [
            _make_skill("Stale Skill", "/skills/agentcore-old", tags=["agentcore"]),
        ]
        skill_repo.delete.return_value = True

        synced_paths = {
            "servers": set(),
            "agents": set(),
            "skills": set(),
        }

        with patch("registry.services.federation_reconciliation._record_reconciliation_metrics"):
            result = await reconcile_agentcore_records(
                config=config,
                server_service=server_service,
                server_repo=server_repo,
                agent_repo=agent_repo,
                skill_repo=skill_repo,
                synced_paths=synced_paths,
                dry_run=False,
            )

        assert result["dry_run"] is False
        assert result["total_removed"] == 3
        assert "Stale Server" in result["servers"]["removed"]
        assert "Stale Agent" in result["agents"]["removed"]
        assert "Stale Skill" in result["skills"]["removed"]

    @pytest.mark.asyncio
    async def test_disabled_agentcore_removes_all(self):
        """When agentcore is disabled, all agentcore records should be stale."""
        config = FederationConfig(
            agentcore=AgentCoreFederationConfig(enabled=False),
        )

        server_repo = AsyncMock()
        server_repo.list_by_source.return_value = {
            "/s1": {"server_name": "S1"},
        }
        server_service = AsyncMock()
        server_service.remove_server.return_value = True

        agent_repo = AsyncMock()
        agent_repo.list_all.return_value = [
            _make_agent("Agent X", "/agents/agentcore-x", tags=["agentcore"]),
        ]
        agent_repo.delete.return_value = True

        skill_repo = AsyncMock()
        skill_repo.list_all.return_value = []

        with patch("registry.services.federation_reconciliation._record_reconciliation_metrics"):
            result = await reconcile_agentcore_records(
                config=config,
                server_service=server_service,
                server_repo=server_repo,
                agent_repo=agent_repo,
                skill_repo=skill_repo,
                synced_paths={"servers": {"/s1"}, "agents": set(), "skills": set()},
                dry_run=False,
            )

        # Even though /s1 was in synced_paths, disabled config means expected is empty
        assert result["total_removed"] == 2
        assert "S1" in result["servers"]["removed"]
        assert "Agent X" in result["agents"]["removed"]
