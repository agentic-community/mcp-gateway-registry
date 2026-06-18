"""Unit tests for embeddings admin routes.

Tests the GET /api/admin/embeddings/missing and
POST /api/admin/embeddings/reindex endpoints.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from registry.api.embeddings_admin_routes import router


@pytest.fixture
def app():
    """Create test app with the embeddings admin router."""
    test_app = FastAPI()
    test_app.include_router(router, prefix="/api")
    return test_app


@pytest.fixture
def admin_client(app):
    """Create test client with admin auth mocked."""
    with patch(
        "registry.api.embeddings_admin_routes.nginx_proxied_auth",
        return_value={"is_admin": True, "username": "admin"},
    ):
        app.dependency_overrides[
            __import__(
                "registry.auth.dependencies", fromlist=["nginx_proxied_auth"]
            ).nginx_proxied_auth
        ] = lambda: {"is_admin": True, "username": "admin"}
        yield TestClient(app)
        app.dependency_overrides.clear()


@pytest.fixture
def non_admin_client(app):
    """Create test client with non-admin auth mocked."""
    from registry.auth.dependencies import nginx_proxied_auth

    app.dependency_overrides[nginx_proxied_auth] = lambda: {
        "is_admin": False,
        "username": "user",
    }
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def mock_search_repo():
    """Mock the search repository."""
    mock_repo = AsyncMock()
    with patch(
        "registry.api.embeddings_admin_routes.get_search_repository",
        return_value=mock_repo,
    ):
        yield mock_repo


class TestGetMissingEmbeddings:
    """Tests for GET /api/admin/embeddings/missing."""

    def test_returns_missing_documents(self, admin_client, mock_search_repo):
        """Returns list of documents missing from embeddings."""
        mock_search_repo.find_missing_embeddings.return_value = {
            "missing": [
                {
                    "path": "/atlassian/",
                    "entity_type": "mcp_server",
                    "name": "Atlassian",
                    "is_enabled": True,
                },
                {
                    "path": "/some-agent",
                    "entity_type": "a2a_agent",
                    "name": "Some Agent",
                    "is_enabled": True,
                },
            ],
            "total_missing": 2,
            "total_indexed": 378,
            "total_source": 380,
        }

        response = admin_client.get("/api/admin/embeddings/missing")

        assert response.status_code == 200
        data = response.json()
        assert data["total_missing"] == 2
        assert data["total_indexed"] == 378
        assert data["total_source"] == 380
        assert len(data["missing"]) == 2
        assert data["missing"][0]["path"] == "/atlassian/"
        assert data["missing"][0]["entity_type"] == "mcp_server"

    def test_returns_empty_when_all_indexed(self, admin_client, mock_search_repo):
        """Returns empty list when no documents are missing."""
        mock_search_repo.find_missing_embeddings.return_value = {
            "missing": [],
            "total_missing": 0,
            "total_indexed": 378,
            "total_source": 378,
        }

        response = admin_client.get("/api/admin/embeddings/missing")

        assert response.status_code == 200
        data = response.json()
        assert data["total_missing"] == 0
        assert len(data["missing"]) == 0

    def test_non_admin_gets_403(self, non_admin_client, mock_search_repo):
        """Non-admin users get 403."""
        response = non_admin_client.get("/api/admin/embeddings/missing")
        assert response.status_code == 403


class TestReindexEmbeddings:
    """Tests for POST /api/admin/embeddings/reindex."""

    def test_reindex_success(self, admin_client, mock_search_repo):
        """Successfully re-indexes specified paths."""
        mock_search_repo.reindex_paths.return_value = {
            "success": 2,
            "failed": 0,
            "total": 2,
            "details": [
                {"path": "/server-1", "entity_type": "mcp_server", "status": "success", "error": None},
                {"path": "/agent-1", "entity_type": "a2a_agent", "status": "success", "error": None},
            ],
        }

        response = admin_client.post(
            "/api/admin/embeddings/reindex",
            json={"paths": ["/server-1", "/agent-1"]},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == 2
        assert data["failed"] == 0
        assert len(data["details"]) == 2

    def test_reindex_partial_failure(self, admin_client, mock_search_repo):
        """Reports partial failures correctly."""
        mock_search_repo.reindex_paths.return_value = {
            "success": 1,
            "failed": 1,
            "total": 2,
            "details": [
                {"path": "/good", "entity_type": "mcp_server", "status": "success", "error": None},
                {"path": "/bad", "entity_type": "unknown", "status": "failed", "error": "Not found"},
            ],
        }

        response = admin_client.post(
            "/api/admin/embeddings/reindex",
            json={"paths": ["/good", "/bad"]},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == 1
        assert data["failed"] == 1

    def test_reindex_empty_paths_rejected(self, admin_client, mock_search_repo):
        """Empty paths list returns 422."""
        response = admin_client.post(
            "/api/admin/embeddings/reindex",
            json={"paths": []},
        )
        assert response.status_code == 422

    def test_reindex_too_many_paths_rejected(self, admin_client, mock_search_repo):
        """More than 100 paths returns 422."""
        paths = [f"/path-{i}" for i in range(101)]
        response = admin_client.post(
            "/api/admin/embeddings/reindex",
            json={"paths": paths},
        )
        assert response.status_code == 422

    def test_non_admin_gets_403(self, non_admin_client, mock_search_repo):
        """Non-admin users get 403."""
        response = non_admin_client.post(
            "/api/admin/embeddings/reindex",
            json={"paths": ["/server-1"]},
        )
        assert response.status_code == 403


class TestGetStaleEmbeddings:
    """Tests for GET /api/admin/embeddings/stale."""

    def test_unsupported_backend_returns_501(self, app, admin_client):
        """Backends without stale scan support return 501."""
        unsupported_repo = object()
        with patch(
            "registry.api.embeddings_admin_routes.get_search_repository",
            return_value=unsupported_repo,
        ):
            response = admin_client.get("/api/admin/embeddings/stale")

        assert response.status_code == 501
        assert "not supported" in response.json()["detail"]

    def test_non_admin_gets_403(self, non_admin_client, mock_search_repo):
        """Non-admin users get 403."""
        response = non_admin_client.get("/api/admin/embeddings/stale")
        assert response.status_code == 403

    def test_returns_stale_documents(self, admin_client, mock_search_repo):
        """Returns orphaned embedding index entries."""
        mock_search_repo.find_stale_embeddings = AsyncMock(
            return_value={
                "stale": [
                    {
                        "path": "/ghost-server",
                        "entity_type": "mcp_server",
                        "name": "Ghost",
                        "is_enabled": True,
                    }
                ],
                "total_stale": 1,
                "total_indexed": 5,
                "total_source": 4,
            }
        )

        response = admin_client.get("/api/admin/embeddings/stale")

        assert response.status_code == 200
        data = response.json()
        assert data["total_stale"] == 1
        assert data["stale"][0]["path"] == "/ghost-server"


class TestCleanupStaleEmbeddings:
    """Tests for POST /api/admin/embeddings/stale/cleanup."""

    def test_unsupported_backend_returns_501(self, app, admin_client):
        """Backends without stale cleanup support return 501."""
        unsupported_repo = object()
        with patch(
            "registry.api.embeddings_admin_routes.get_search_repository",
            return_value=unsupported_repo,
        ):
            response = admin_client.post(
                "/api/admin/embeddings/stale/cleanup",
                json={"paths": ["/ghost-server"]},
            )

        assert response.status_code == 501
        assert "not supported" in response.json()["detail"]

    def test_non_admin_gets_403(self, non_admin_client, mock_search_repo):
        """Non-admin users get 403."""
        response = non_admin_client.post(
            "/api/admin/embeddings/stale/cleanup",
            json={"paths": ["/ghost-server"]},
        )
        assert response.status_code == 403

    def test_removes_stale_paths(self, admin_client, mock_search_repo):
        """Admin can remove orphaned embeddings by path."""
        mock_search_repo.remove_stale_embeddings = AsyncMock(
            return_value={
                "removed": 1,
                "not_found": 0,
                "failed": 0,
                "total": 1,
                "details": [{"path": "/ghost-server", "status": "removed", "error": None}],
            }
        )

        response = admin_client.post(
            "/api/admin/embeddings/stale/cleanup",
            json={"paths": ["/ghost-server"]},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["removed"] == 1
        assert data["not_found"] == 0
        assert data["failed"] == 0

    def test_reports_not_found_for_noop_path(self, admin_client, mock_search_repo):
        """A path that matched nothing is reported as not_found, not removed."""
        mock_search_repo.remove_stale_embeddings = AsyncMock(
            return_value={
                "removed": 0,
                "not_found": 1,
                "failed": 0,
                "total": 1,
                "details": [{"path": "/typo-path", "status": "not_found", "error": None}],
            }
        )

        response = admin_client.post(
            "/api/admin/embeddings/stale/cleanup",
            json={"paths": ["/typo-path"]},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["removed"] == 0
        assert data["not_found"] == 1

