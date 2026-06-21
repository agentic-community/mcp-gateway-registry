"""
Unit tests for system stats endpoint and repository count methods.

Tests the new /api/stats endpoint and count() methods added to repositories.
"""

import logging
from datetime import UTC, datetime
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
    async def test_database_status_documentdb_healthy(self, mock_documentdb_client):
        """Test database status with healthy DocumentDB."""
        from registry.api.system_routes import _get_database_status

        with patch("registry.api.system_routes.settings") as mock_settings:
            mock_settings.storage_backend = "documentdb"
            mock_settings.documentdb_host = "localhost"
            mock_settings.documentdb_port = 27017
            mock_settings.mongodb_connection_string = None

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
    async def test_database_status_documentdb_with_connection_string_override(
        self, mock_documentdb_client
    ):
        """Test database status masks credentials when a connection string override is set."""
        from registry.api.system_routes import _get_database_status

        connection_string = "mongodb+srv://user:pass@cluster.example.net/db"

        with patch("registry.api.system_routes.settings") as mock_settings:
            mock_settings.storage_backend = "documentdb"
            mock_settings.documentdb_host = "should-not-be-used"
            mock_settings.documentdb_port = 12345
            mock_settings.mongodb_connection_string = connection_string

            with patch(
                "registry.repositories.documentdb.client.get_documentdb_client",
                new_callable=AsyncMock,
                return_value=mock_documentdb_client,
            ):
                status = await _get_database_status()

                assert status["backend"] == "documentdb"
                assert status["status"] == "Healthy"
                assert status["host"] == "cluster.example.net"
                assert "user:pass" not in status["host"]
                assert "mongodb+srv://" not in status["host"]

    @pytest.mark.asyncio
    async def test_database_status_documentdb_unhealthy(self):
        """Test database status with unhealthy DocumentDB."""
        from registry.api.system_routes import _get_database_status

        with patch("registry.api.system_routes.settings") as mock_settings:
            mock_settings.storage_backend = "documentdb"
            mock_settings.documentdb_host = "localhost"
            mock_settings.documentdb_port = 27017
            mock_settings.mongodb_connection_string = None

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
        registry.api.system_routes._server_start_time = datetime.now(UTC)

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
                    with patch("registry.api.system_routes.settings") as mock_settings, \
                         patch("registry.api.system_routes._get_database_status", new_callable=AsyncMock, return_value={"backend": "mongodb-ce", "status": "Healthy"}):
                        mock_settings.storage_backend = "mongodb-ce"
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
    async def test_stats_endpoint_success_when_authenticated(self, mock_repositories):
        """Authenticated caller sees full stats payload."""
        import registry.api.system_routes
        from registry.auth.dependencies import nginx_proxied_auth

        # Reset cache
        registry.api.system_routes._stats_cache = None
        registry.api.system_routes._stats_cache_time = None
        registry.api.system_routes._server_start_time = datetime.now(UTC)

        user_context = {
            "username": "alice",
            "groups": ["mcp-registry-user"],
            "scopes": [],
            "auth_method": "oauth2",
            "provider": "keycloak",
            "accessible_servers": [],
            "accessible_services": [],
            "accessible_agents": [],
            "ui_permissions": {},
            "can_modify_servers": False,
            "is_admin": False,
        }

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
                    with patch("registry.api.system_routes.settings") as mock_settings, \
                         patch("registry.api.system_routes._get_database_status", new_callable=AsyncMock, return_value={"backend": "mongodb-ce", "status": "Healthy"}):
                        mock_settings.storage_backend = "mongodb-ce"
                        mock_settings.deployment_mode.value = "standalone"

                        from registry.main import app

                        app.dependency_overrides[nginx_proxied_auth] = lambda: user_context
                        try:
                            client = TestClient(app)
                            response = client.get("/api/stats")
                        finally:
                            app.dependency_overrides.pop(nginx_proxied_auth, None)

                        assert response.status_code == 200
                        data = response.json()
                        assert "uptime_seconds" in data
                        assert "started_at" in data
                        assert "version" in data
                        assert "deployment_type" in data
                        assert "registry_stats" in data

    def test_stats_endpoint_rejects_unauthenticated_callers(self):
        """Unauthenticated callers must get 401, not an anonymous-logged 200.

        Regression guard for the audit-anonymous bug: /api/stats previously
        had no auth dependency and logged authenticated users as anonymous.
        """
        from registry.main import app

        client = TestClient(app)
        response = client.get("/api/stats")

        assert response.status_code == 401

    def test_telemetry_detection_rejects_unauthenticated_callers(self):
        """Regression guard: /api/system/telemetry-detection now requires auth
        for the same audit-attribution reason as /api/stats."""
        from registry.main import app

        client = TestClient(app)
        response = client.get("/api/system/telemetry-detection")

        assert response.status_code == 401
