"""Unit tests for registry/api/log_routes.py - Application log retrieval API."""

import logging
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from registry.api import log_routes
from registry.api.log_routes import router

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _clear_rate_limit_cache():
    """Reset rate limit cache between tests."""
    log_routes._rate_limit_cache.clear()


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_admin_context() -> dict[str, Any]:
    return {
        "username": "admin-user",
        "groups": ["mcp-registry-admin"],
        "scopes": ["mcp-registry-admin"],
        "auth_method": "session",
        "provider": "local",
        "is_admin": True,
        "accessible_servers": ["all"],
        "accessible_services": ["all"],
        "accessible_agents": ["all"],
    }


@pytest.fixture
def mock_non_admin_context() -> dict[str, Any]:
    return {
        "username": "regular-user",
        "groups": ["mcp-registry-user"],
        "scopes": ["read:servers"],
        "auth_method": "session",
        "provider": "local",
        "is_admin": False,
        "accessible_servers": ["all"],
        "accessible_services": ["all"],
        "accessible_agents": ["all"],
    }


@pytest.fixture
def mock_app_log_repo():
    mock = AsyncMock()
    mock.query.return_value = ([], 0)
    mock.get_distinct_services.return_value = ["registry", "auth-server"]
    mock.get_distinct_hostnames.return_value = ["pod-abc123", "pod-def456"]
    return mock


@pytest.fixture
def sample_log_entries() -> list[dict[str, Any]]:
    return [
        {
            "timestamp": datetime(2026, 4, 24, 10, 0, 0, tzinfo=UTC),
            "hostname": "pod-abc123",
            "service": "registry",
            "level": "INFO",
            "level_no": 20,
            "logger": "registry.main",
            "filename": "main.py",
            "lineno": 42,
            "process": 130,
            "message": "Server started successfully",
        },
        {
            "timestamp": datetime(2026, 4, 24, 10, 0, 1, tzinfo=UTC),
            "hostname": "pod-abc123",
            "service": "registry",
            "level": "ERROR",
            "level_no": 40,
            "logger": "registry.api.server_routes",
            "filename": "server_routes.py",
            "lineno": 100,
            "process": 130,
            "message": "Failed to register server: timeout",
        },
    ]


@pytest.fixture
def admin_client(mock_admin_context, mock_app_log_repo):
    app = FastAPI()
    app.include_router(router, prefix="/api")

    from registry.auth.dependencies import nginx_proxied_auth

    app.dependency_overrides[nginx_proxied_auth] = lambda: mock_admin_context

    with patch(
        "registry.api.log_routes.get_app_log_repository",
        return_value=mock_app_log_repo,
    ):
        client = TestClient(app)
        yield client

    app.dependency_overrides.clear()


@pytest.fixture
def non_admin_client(mock_non_admin_context, mock_app_log_repo):
    app = FastAPI()
    app.include_router(router, prefix="/api")

    from registry.auth.dependencies import nginx_proxied_auth

    app.dependency_overrides[nginx_proxied_auth] = lambda: mock_non_admin_context

    with patch(
        "registry.api.log_routes.get_app_log_repository",
        return_value=mock_app_log_repo,
    ):
        client = TestClient(app)
        yield client

    app.dependency_overrides.clear()


@pytest.fixture
def no_mongo_client(mock_admin_context):
    """Client where MongoDB is not available (file backend)."""
    app = FastAPI()
    app.include_router(router, prefix="/api")

    from registry.auth.dependencies import nginx_proxied_auth

    app.dependency_overrides[nginx_proxied_auth] = lambda: mock_admin_context

    with patch(
        "registry.api.log_routes.get_app_log_repository",
        return_value=None,
    ):
        client = TestClient(app)
        yield client

    app.dependency_overrides.clear()


# =============================================================================
# ACCESS CONTROL TESTS
# =============================================================================


class TestLogRoutesAccessControl:
    """Test admin-only access enforcement."""

    def test_query_logs_requires_admin(self, non_admin_client):
        response = non_admin_client.get("/api/admin/logs")
        assert response.status_code == 403
        assert "Admin access required" in response.json()["detail"]

    def test_export_logs_requires_admin(self, non_admin_client):
        response = non_admin_client.get("/api/admin/logs/export")
        assert response.status_code == 403

    def test_metadata_requires_admin(self, non_admin_client):
        response = non_admin_client.get("/api/admin/logs/metadata")
        assert response.status_code == 403

    def test_admin_can_query_logs(self, admin_client):
        response = admin_client.get("/api/admin/logs")
        assert response.status_code == 200

    def test_admin_can_get_metadata(self, admin_client):
        response = admin_client.get("/api/admin/logs/metadata")
        assert response.status_code == 200


# =============================================================================
# QUERY LOGS TESTS
# =============================================================================


class TestQueryLogs:
    """Test GET /api/admin/logs endpoint."""

    def test_empty_response(self, admin_client):
        response = admin_client.get("/api/admin/logs")
        data = response.json()
        assert data["entries"] == []
        assert data["total_count"] == 0
        assert data["limit"] == 100
        assert data["offset"] == 0
        assert data["has_next"] is False

    def test_with_entries(self, admin_client, mock_app_log_repo, sample_log_entries):
        mock_app_log_repo.query.return_value = (sample_log_entries, 2)

        response = admin_client.get("/api/admin/logs")
        data = response.json()
        assert data["total_count"] == 2
        assert len(data["entries"]) == 2
        assert data["entries"][0]["service"] == "registry"
        assert data["entries"][0]["level"] == "INFO"

    def test_filter_by_service(self, admin_client, mock_app_log_repo):
        mock_app_log_repo.query.return_value = ([], 0)

        response = admin_client.get("/api/admin/logs?service=auth-server")
        assert response.status_code == 200
        mock_app_log_repo.query.assert_called_once()
        call_kwargs = mock_app_log_repo.query.call_args[1]
        assert call_kwargs["service"] == "auth-server"

    def test_filter_by_level(self, admin_client, mock_app_log_repo):
        mock_app_log_repo.query.return_value = ([], 0)

        response = admin_client.get("/api/admin/logs?level=ERROR")
        assert response.status_code == 200
        call_kwargs = mock_app_log_repo.query.call_args[1]
        assert call_kwargs["level_no"] == 40

    def test_filter_by_hostname(self, admin_client, mock_app_log_repo):
        mock_app_log_repo.query.return_value = ([], 0)

        response = admin_client.get("/api/admin/logs?hostname=pod-abc123")
        assert response.status_code == 200
        call_kwargs = mock_app_log_repo.query.call_args[1]
        assert call_kwargs["hostname"] == "pod-abc123"

    def test_filter_by_time_range(self, admin_client, mock_app_log_repo):
        mock_app_log_repo.query.return_value = ([], 0)

        response = admin_client.get(
            "/api/admin/logs?start=2026-04-24T00:00:00Z&end=2026-04-24T23:59:59Z"
        )
        assert response.status_code == 200
        call_kwargs = mock_app_log_repo.query.call_args[1]
        assert call_kwargs["start"] is not None
        assert call_kwargs["end"] is not None

    def test_search_in_message(self, admin_client, mock_app_log_repo):
        mock_app_log_repo.query.return_value = ([], 0)

        response = admin_client.get("/api/admin/logs?search=timeout")
        assert response.status_code == 200
        call_kwargs = mock_app_log_repo.query.call_args[1]
        assert call_kwargs["search"] == "timeout"

    def test_pagination_params(self, admin_client, mock_app_log_repo):
        mock_app_log_repo.query.return_value = ([], 0)

        response = admin_client.get("/api/admin/logs?limit=50&offset=100")
        assert response.status_code == 200
        call_kwargs = mock_app_log_repo.query.call_args[1]
        assert call_kwargs["limit"] == 50
        assert call_kwargs["skip"] == 100

    def test_has_next_true(self, admin_client, mock_app_log_repo, sample_log_entries):
        mock_app_log_repo.query.return_value = ([sample_log_entries[0]], 5)

        response = admin_client.get("/api/admin/logs?limit=1&offset=0")
        data = response.json()
        assert data["has_next"] is True
        assert data["total_count"] == 5

    def test_has_next_false_at_end(self, admin_client, mock_app_log_repo, sample_log_entries):
        mock_app_log_repo.query.return_value = ([sample_log_entries[0]], 5)

        response = admin_client.get("/api/admin/logs?limit=1&offset=4")
        data = response.json()
        assert data["has_next"] is False

    def test_limit_validation_too_low(self, admin_client):
        response = admin_client.get("/api/admin/logs?limit=0")
        assert response.status_code == 422

    def test_limit_validation_too_high(self, admin_client):
        response = admin_client.get("/api/admin/logs?limit=10001")
        assert response.status_code == 422

    def test_offset_validation_negative(self, admin_client):
        response = admin_client.get("/api/admin/logs?offset=-1")
        assert response.status_code == 422


# =============================================================================
# EXPORT LOGS TESTS
# =============================================================================


class TestExportLogs:
    """Test GET /api/admin/logs/export endpoint."""

    def test_export_empty(self, admin_client, mock_app_log_repo):
        mock_app_log_repo.query.return_value = ([], 0)

        response = admin_client.get("/api/admin/logs/export")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        assert response.text == ""

    def test_export_with_entries(self, admin_client, mock_app_log_repo, sample_log_entries):
        mock_app_log_repo.query.return_value = (sample_log_entries, 2)

        response = admin_client.get("/api/admin/logs/export")
        assert response.status_code == 200
        lines = response.text.strip().split("\n")
        assert len(lines) == 2

    def test_export_content_disposition(self, admin_client, mock_app_log_repo):
        mock_app_log_repo.query.return_value = ([], 0)

        response = admin_client.get("/api/admin/logs/export")
        disposition = response.headers.get("content-disposition", "")
        assert "logs-all-" in disposition
        assert ".jsonl" in disposition

    def test_export_with_filters(self, admin_client, mock_app_log_repo):
        mock_app_log_repo.query.return_value = ([], 0)

        response = admin_client.get("/api/admin/logs/export?service=registry&level=ERROR")
        assert response.status_code == 200
        call_kwargs = mock_app_log_repo.query.call_args[1]
        assert call_kwargs["service"] == "registry"
        assert call_kwargs["level_no"] == 40

    def test_export_limit_validation(self, admin_client):
        response = admin_client.get("/api/admin/logs/export?limit=50001")
        assert response.status_code == 422


# =============================================================================
# METADATA TESTS
# =============================================================================


class TestLogMetadata:
    """Test GET /api/admin/logs/metadata endpoint."""

    def test_metadata_returns_services_and_hostnames(self, admin_client, mock_app_log_repo):
        response = admin_client.get("/api/admin/logs/metadata")
        data = response.json()
        assert "registry" in data["services"]
        assert "auth-server" in data["services"]
        assert "pod-abc123" in data["hostnames"]
        assert "pod-def456" in data["hostnames"]
        assert "INFO" in data["levels"]
        assert "ERROR" in data["levels"]


# =============================================================================
# NO MONGODB BACKEND TESTS
# =============================================================================


class TestNoMongoDBBackend:
    """Test behavior when MongoDB backend is not available."""

    def test_query_returns_503(self, no_mongo_client):
        response = no_mongo_client.get("/api/admin/logs")
        assert response.status_code == 503
        assert "not available" in response.json()["detail"]

    def test_export_returns_503(self, no_mongo_client):
        response = no_mongo_client.get("/api/admin/logs/export")
        assert response.status_code == 503

    def test_metadata_returns_503(self, no_mongo_client):
        response = no_mongo_client.get("/api/admin/logs/metadata")
        assert response.status_code == 503


# =============================================================================
# RATE LIMITING TESTS
# =============================================================================


class TestRateLimiting:
    """Test per-user rate limiting on log API endpoints."""

    def test_rate_limit_exceeded(self, admin_client, mock_app_log_repo):
        mock_app_log_repo.query.return_value = ([], 0)

        for _ in range(10):
            response = admin_client.get("/api/admin/logs")
            assert response.status_code == 200

        response = admin_client.get("/api/admin/logs")
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]

    def test_rate_limit_applies_to_export(self, admin_client, mock_app_log_repo):
        mock_app_log_repo.query.return_value = ([], 0)

        for _ in range(10):
            admin_client.get("/api/admin/logs/export")

        response = admin_client.get("/api/admin/logs/export")
        assert response.status_code == 429


# =============================================================================
# SEARCH SANITIZATION TESTS
# =============================================================================


class TestSearchSanitization:
    """Test that regex metacharacters in search are properly escaped."""

    def test_regex_metacharacters_escaped(self, admin_client, mock_app_log_repo):
        mock_app_log_repo.query.return_value = ([], 0)

        response = admin_client.get("/api/admin/logs?search=error.*timeout")
        assert response.status_code == 200
        call_kwargs = mock_app_log_repo.query.call_args[1]
        assert call_kwargs["search"] == r"error\.\*timeout"

    def test_search_truncated_at_max_length(self, admin_client, mock_app_log_repo):
        mock_app_log_repo.query.return_value = ([], 0)

        long_search = "a" * 300
        response = admin_client.get(f"/api/admin/logs?search={long_search}")
        assert response.status_code == 200
        call_kwargs = mock_app_log_repo.query.call_args[1]
        assert len(call_kwargs["search"]) == 200

    def test_empty_search_returns_none(self, admin_client, mock_app_log_repo):
        mock_app_log_repo.query.return_value = ([], 0)

        response = admin_client.get("/api/admin/logs")
        assert response.status_code == 200
        call_kwargs = mock_app_log_repo.query.call_args[1]
        assert call_kwargs["search"] is None

    def test_level_no_mapping(self, admin_client, mock_app_log_repo):
        mock_app_log_repo.query.return_value = ([], 0)

        for level, expected_no in [
            ("DEBUG", 10),
            ("INFO", 20),
            ("WARNING", 30),
            ("ERROR", 40),
            ("CRITICAL", 50),
        ]:
            log_routes._rate_limit_cache.clear()
            response = admin_client.get(f"/api/admin/logs?level={level}")
            assert response.status_code == 200
            call_kwargs = mock_app_log_repo.query.call_args[1]
            assert call_kwargs["level_no"] == expected_no, f"{level} should map to {expected_no}"
