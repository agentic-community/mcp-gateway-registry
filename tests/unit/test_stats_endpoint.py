"""
Unit tests for system stats endpoint and repository count methods.

Tests the new /api/stats endpoint and count() methods added to repositories.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


logger = logging.getLogger(__name__)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_repositories():
    """Mock repository instances with count() methods."""
    mock_server_repo = AsyncMock()
    mock_server_repo.count = AsyncMock(return_value=10)

    mock_agent_repo = AsyncMock()
    mock_agent_repo.count = AsyncMock(return_value=5)

    mock_skill_repo = AsyncMock()
    mock_skill_repo.count = AsyncMock(return_value=3)

    return {
        "server": mock_server_repo,
        "agent": mock_agent_repo,
        "skill": mock_skill_repo,
    }


@pytest.fixture
def mock_documentdb_client():
    """Mock DocumentDB client for database status check."""
    mock_db = AsyncMock()
    mock_db.command = AsyncMock(return_value={"ok": 1})
    return mock_db


# =============================================================================
# TEST: Repository count() Methods
# =============================================================================


@pytest.mark.unit
@pytest.mark.repositories
class TestRepositoryCountMethods:
    """Tests for count() methods in repositories."""

    @pytest.mark.asyncio
    async def test_file_server_repository_count(self):
        """Test FileServerRepository count() method."""
        from registry.repositories.file.server_repository import FileServerRepository

        with patch("registry.repositories.file.server_repository.settings") as mock_settings:
            # Setup mock settings
            mock_servers_dir = MagicMock()
            mock_servers_dir.mkdir = MagicMock()
            mock_state_path = MagicMock()
            mock_state_path.exists = MagicMock(return_value=False)

            mock_settings.servers_dir = mock_servers_dir
            mock_settings.state_file_path = mock_state_path

            # Create repository
            repo = FileServerRepository()

            # Add some test servers
            repo._servers = {
                "/server1": {"path": "/server1", "server_name": "Server 1"},
                "/server2": {"path": "/server2", "server_name": "Server 2"},
                "/server3": {"path": "/server3", "server_name": "Server 3"},
            }

            # Act
            count = await repo.count()

            # Assert
            assert count == 3

    @pytest.mark.asyncio
    async def test_file_agent_repository_count(self):
        """Test FileAgentRepository count() method."""
        from registry.repositories.file.agent_repository import FileAgentRepository
        from registry.schemas.agent_models import AgentCard

        with patch("registry.repositories.file.agent_repository.settings") as mock_settings:
            # Setup mock settings
            mock_agents_dir = MagicMock()
            mock_agents_dir.mkdir = MagicMock()
            mock_agents_dir.glob = MagicMock(return_value=[])
            mock_state_file = MagicMock()
            mock_state_file.exists = MagicMock(return_value=False)

            mock_settings.agents_dir = mock_agents_dir
            mock_settings.agent_state_file_path = mock_state_file

            # Create repository
            repo = FileAgentRepository()

            # Mock get_all to return test data
            with patch.object(repo, "get_all", new_callable=AsyncMock) as mock_get_all:
                mock_get_all.return_value = {
                    "/agent1": MagicMock(spec=AgentCard),
                    "/agent2": MagicMock(spec=AgentCard),
                }

                # Act
                count = await repo.count()

                # Assert
                assert count == 2


# =============================================================================
# TEST: Helper Functions
# =============================================================================


@pytest.mark.unit
class TestDetectDeploymentType:
    """Tests for _detect_deployment_type helper function."""

    def test_detect_kubernetes(self):
        """Test detection of Kubernetes environment."""
        from registry.api.system_routes import _detect_deployment_type

        with patch.dict("os.environ", {"KUBERNETES_SERVICE_HOST": "10.0.0.1"}):
            result = _detect_deployment_type()
            assert result == "Kubernetes"

    def test_detect_ecs(self):
        """Test detection of ECS environment."""
        from registry.api.system_routes import _detect_deployment_type

        with patch.dict(
            "os.environ",
            {"ECS_CONTAINER_METADATA_URI": "http://169.254.170.2/v3"},
            clear=True,
        ):
            result = _detect_deployment_type()
            assert result == "ECS"

    def test_detect_ecs_v4(self):
        """Test detection of ECS environment with v4 metadata."""
        from registry.api.system_routes import _detect_deployment_type

        with patch.dict(
            "os.environ",
            {"ECS_CONTAINER_METADATA_URI_V4": "http://169.254.170.2/v4"},
            clear=True,
        ):
            result = _detect_deployment_type()
            assert result == "ECS"

    def test_detect_ec2(self):
        """Test detection of EC2 environment."""
        from registry.api.system_routes import _detect_deployment_type

        with patch.dict("os.environ", {"AWS_EXECUTION_ENV": "AWS_ECS_EC2"}, clear=True):
            result = _detect_deployment_type()
            assert result == "EC2"

    def test_detect_local(self):
        """Test detection of local environment."""
        from registry.api.system_routes import _detect_deployment_type

        with patch.dict("os.environ", {}, clear=True):
            result = _detect_deployment_type()
            assert result == "Local"


@pytest.mark.unit
class TestGetRegistryStats:
    """Tests for _get_registry_stats function."""

    @pytest.mark.asyncio
    async def test_get_registry_stats_success(self, mock_repositories):
        """Test successful stats collection."""
        from registry.api.system_routes import _get_registry_stats

        with patch(
            "registry.repositories.factory.get_server_repository",
            return_value=mock_repositories["server"],
        ):
            with patch(
                "registry.repositories.factory.get_agent_repository",
                return_value=mock_repositories["agent"],
            ):
                with patch(
                    "registry.repositories.factory.get_skill_repository",
                    return_value=mock_repositories["skill"],
                ):
                    # Act
                    stats = await _get_registry_stats()

                    # Assert
                    assert stats["servers"] == 10
                    assert stats["agents"] == 5
                    assert stats["skills"] == 3

    @pytest.mark.asyncio
    async def test_get_registry_stats_error_handling(self):
        """Test error handling in stats collection."""
        from registry.api.system_routes import _get_registry_stats

        with patch(
            "registry.repositories.factory.get_server_repository", side_effect=Exception("DB error")
        ):
            # Act
            stats = await _get_registry_stats()

            # Assert - should return zeros on error
            assert stats["servers"] == 0
            assert stats["agents"] == 0
            assert stats["skills"] == 0


@pytest.mark.unit
class TestGetDatabaseStatus:
    """Tests for _get_database_status function."""

    @pytest.mark.asyncio
    async def test_database_status_file_backend(self):
        """Test database status with file backend."""
        from registry.api.system_routes import _get_database_status

        with patch("registry.api.system_routes.settings") as mock_settings:
            mock_settings.storage_backend = "file"

            # Act
            status = await _get_database_status()

            # Assert
            assert status["backend"] == "file"
            assert status["status"] == "N/A"
            assert status["host"] == "N/A"

    @pytest.mark.asyncio
    async def test_database_status_documentdb_healthy(self, mock_documentdb_client):
        """Test database status with healthy DocumentDB."""
        from registry.api.system_routes import _get_database_status

        with patch("registry.api.system_routes.settings") as mock_settings:
            mock_settings.storage_backend = "documentdb"
            mock_settings.documentdb_host = "localhost"
            mock_settings.documentdb_port = 27017

            with patch(
                "registry.repositories.documentdb.client.get_documentdb_client",
                new_callable=AsyncMock,
                return_value=mock_documentdb_client,
            ):
                # Act
                status = await _get_database_status()

                # Assert
                assert status["backend"] == "documentdb"
                assert status["status"] == "Healthy"
                assert status["host"] == "localhost:27017"

    @pytest.mark.asyncio
    async def test_database_status_documentdb_unhealthy(self):
        """Test database status with unhealthy DocumentDB."""
        from registry.api.system_routes import _get_database_status

        with patch("registry.api.system_routes.settings") as mock_settings:
            mock_settings.storage_backend = "documentdb"
            mock_settings.documentdb_host = "localhost"
            mock_settings.documentdb_port = 27017

            with patch(
                "registry.repositories.documentdb.client.get_documentdb_client",
                new_callable=AsyncMock,
                side_effect=Exception("Connection failed"),
            ):
                # Act
                status = await _get_database_status()

                # Assert
                assert status["backend"] == "documentdb"
                assert status["status"] == "Unhealthy"
                assert status["host"] == "localhost:27017"


@pytest.mark.unit
class TestGetCachedStats:
    """Tests for _get_cached_stats function."""

    @pytest.mark.asyncio
    async def test_cached_stats_cache_miss(self, mock_repositories):
        """Test stats collection on cache miss."""
        import registry.api.system_routes

        # Reset cache
        registry.api.system_routes._stats_cache = None
        registry.api.system_routes._stats_cache_time = None
        registry.api.system_routes._server_start_time = datetime.now(timezone.utc)

        with patch(
            "registry.repositories.factory.get_server_repository",
            return_value=mock_repositories["server"],
        ):
            with patch(
                "registry.repositories.factory.get_agent_repository",
                return_value=mock_repositories["agent"],
            ):
                with patch(
                    "registry.repositories.factory.get_skill_repository",
                    return_value=mock_repositories["skill"],
                ):
                    with patch("registry.api.system_routes.settings") as mock_settings:
                        mock_settings.storage_backend = "file"
                        mock_settings.deployment_mode.value = "standalone"

                        # Act
                        stats = await registry.api.system_routes._get_cached_stats()

                        # Assert
                        assert "uptime_seconds" in stats
                        assert "started_at" in stats
                        assert "version" in stats
                        assert "deployment_type" in stats
                        assert "deployment_mode" in stats
                        assert "registry_stats" in stats
                        assert stats["registry_stats"]["servers"] == 10
                        assert stats["registry_stats"]["agents"] == 5
                        assert stats["registry_stats"]["skills"] == 3


# =============================================================================
# TEST: Stats Endpoint
# =============================================================================


@pytest.mark.unit
class TestStatsEndpoint:
    """Tests for /api/stats endpoint."""

    @pytest.mark.asyncio
    async def test_stats_endpoint_success(self, mock_repositories):
        """Test successful stats endpoint call."""
        import registry.api.system_routes

        # Reset cache
        registry.api.system_routes._stats_cache = None
        registry.api.system_routes._stats_cache_time = None
        registry.api.system_routes._server_start_time = datetime.now(timezone.utc)

        with patch(
            "registry.repositories.factory.get_server_repository",
            return_value=mock_repositories["server"],
        ):
            with patch(
                "registry.repositories.factory.get_agent_repository",
                return_value=mock_repositories["agent"],
            ):
                with patch(
                    "registry.repositories.factory.get_skill_repository",
                    return_value=mock_repositories["skill"],
                ):
                    with patch("registry.api.system_routes.settings") as mock_settings:
                        mock_settings.storage_backend = "file"
                        mock_settings.deployment_mode.value = "standalone"

                        from registry.main import app

                        client = TestClient(app)

                        # Act
                        response = client.get("/api/stats")

                        # Assert
                        assert response.status_code == 200
                        data = response.json()
                        assert "uptime_seconds" in data
                        assert "started_at" in data
                        assert "version" in data
                        assert "deployment_type" in data
                        assert "registry_stats" in data
