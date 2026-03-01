"""Integration tests for virtual server API endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from registry.api.virtual_server_routes import _normalize_virtual_path
from registry.auth.dependencies import nginx_proxied_auth
from registry.main import app
from registry.schemas.virtual_server_models import VirtualServerConfig

# Sample virtual server data for testing
SAMPLE_VS_DATA = {
    "server_name": "Dev Essentials",
    "path": "/virtual/dev-essentials",
    "description": "Tools for everyday development",
    "tool_mappings": [
        {
            "tool_name": "search",
            "backend_server_path": "/github",
        },
    ],
    "required_scopes": [],
    "tags": ["dev", "productivity"],
}


ADMIN_CONTEXT = {
    "username": "admin",
    "groups": ["mcp-registry-admin"],
    "scopes": ["mcp-servers-unrestricted/read", "mcp-servers-unrestricted/execute"],
    "is_admin": True,
    "can_modify_servers": True,
}


USER_CONTEXT = {
    "username": "testuser",
    "groups": ["mcp-registry-user"],
    "scopes": ["mcp-servers-unrestricted/read"],
    "is_admin": False,
    "can_modify_servers": False,
}


@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_auth_admin(client):
    """Mock authentication returning admin user context."""
    app.dependency_overrides[nginx_proxied_auth] = lambda: ADMIN_CONTEXT
    yield ADMIN_CONTEXT
    app.dependency_overrides.pop(nginx_proxied_auth, None)


@pytest.fixture
def mock_auth_user(client):
    """Mock authentication returning regular user context."""
    app.dependency_overrides[nginx_proxied_auth] = lambda: USER_CONTEXT
    yield USER_CONTEXT
    app.dependency_overrides.pop(nginx_proxied_auth, None)


@pytest.fixture
def mock_vs_service():
    """Mock virtual server service."""
    mock = AsyncMock()
    mock.list_virtual_servers = AsyncMock(return_value=[])
    mock.get_virtual_server = AsyncMock(return_value=None)
    mock.create_virtual_server = AsyncMock()
    mock.update_virtual_server = AsyncMock()
    mock.delete_virtual_server = AsyncMock(return_value=True)
    mock.toggle_virtual_server = AsyncMock(return_value=True)
    mock.resolve_tools = AsyncMock(return_value=[])
    mock.rate_virtual_server = AsyncMock(
        return_value={
            "average_rating": 4.0,
            "is_new_rating": True,
            "total_ratings": 1,
        }
    )
    mock.get_virtual_server_rating = AsyncMock(
        return_value={
            "num_stars": 4.0,
            "rating_details": [{"user": "testuser", "rating": 4}],
        }
    )
    return mock


@pytest.fixture
def mock_catalog_service():
    """Mock tool catalog service."""
    mock = AsyncMock()
    mock.get_tool_catalog = AsyncMock(return_value=[])
    return mock


class TestListVirtualServers:
    """Tests for GET /api/virtual-servers."""

    def test_list_empty(self, client, mock_auth_admin, mock_vs_service):
        """Test listing virtual servers when none exist."""
        with patch(
            "registry.api.virtual_server_routes.get_virtual_server_service",
            return_value=mock_vs_service,
        ):
            response = client.get("/api/virtual-servers")

        assert response.status_code == 200
        data = response.json()
        assert data["virtual_servers"] == []
        assert data["total_count"] == 0

    def test_list_with_user_auth(self, client, mock_auth_user, mock_vs_service):
        """Test that regular users can list virtual servers."""
        with patch(
            "registry.api.virtual_server_routes.get_virtual_server_service",
            return_value=mock_vs_service,
        ):
            response = client.get("/api/virtual-servers")

        assert response.status_code == 200


class TestCreateVirtualServer:
    """Tests for POST /api/virtual-servers."""

    def test_create_success(self, client, mock_auth_admin, mock_vs_service):
        """Test creating a virtual server."""
        created_config = VirtualServerConfig(
            path="/virtual/dev-essentials",
            server_name="Dev Essentials",
            description="Tools for development",
        )
        mock_vs_service.create_virtual_server.return_value = created_config

        with patch(
            "registry.api.virtual_server_routes.get_virtual_server_service",
            return_value=mock_vs_service,
        ):
            response = client.post(
                "/api/virtual-servers",
                json=SAMPLE_VS_DATA,
            )

        assert response.status_code == 201
        data = response.json()
        assert data["path"] == "/virtual/dev-essentials"
        assert data["server_name"] == "Dev Essentials"

    def test_create_requires_admin(self, client, mock_auth_user, mock_vs_service):
        """Test that creating requires admin permissions."""
        with patch(
            "registry.api.virtual_server_routes.get_virtual_server_service",
            return_value=mock_vs_service,
        ):
            response = client.post(
                "/api/virtual-servers",
                json=SAMPLE_VS_DATA,
            )

        assert response.status_code == 403

    def test_create_validation_error(self, client, mock_auth_admin, mock_vs_service):
        """Test creating with invalid data returns 400."""
        from registry.exceptions import VirtualServerValidationError

        mock_vs_service.create_virtual_server.side_effect = VirtualServerValidationError(
            "Backend server '/nonexistent' does not exist"
        )

        with patch(
            "registry.api.virtual_server_routes.get_virtual_server_service",
            return_value=mock_vs_service,
        ):
            response = client.post(
                "/api/virtual-servers",
                json=SAMPLE_VS_DATA,
            )

        assert response.status_code == 400
        assert "does not exist" in response.json()["detail"]

    def test_create_duplicate_path_returns_409(self, client, mock_auth_admin, mock_vs_service):
        """Test creating virtual server with duplicate path returns 409."""
        from registry.exceptions import VirtualServerAlreadyExistsError

        mock_vs_service.create_virtual_server.side_effect = VirtualServerAlreadyExistsError(
            "/virtual/dev-essentials"
        )

        with patch(
            "registry.api.virtual_server_routes.get_virtual_server_service",
            return_value=mock_vs_service,
        ):
            response = client.post(
                "/api/virtual-servers",
                json=SAMPLE_VS_DATA,
            )

        assert response.status_code == 409


class TestGetVirtualServer:
    """Tests for GET /api/virtual-servers/{path}."""

    def test_get_existing(self, client, mock_auth_admin, mock_vs_service):
        """Test getting an existing virtual server."""
        config = VirtualServerConfig(
            path="/virtual/dev-essentials",
            server_name="Dev Essentials",
        )
        mock_vs_service.get_virtual_server.return_value = config

        with patch(
            "registry.api.virtual_server_routes.get_virtual_server_service",
            return_value=mock_vs_service,
        ):
            response = client.get("/api/virtual-servers/virtual/dev-essentials")

        assert response.status_code == 200
        assert response.json()["server_name"] == "Dev Essentials"

    def test_get_not_found(self, client, mock_auth_admin, mock_vs_service):
        """Test getting a nonexistent virtual server."""
        mock_vs_service.get_virtual_server.return_value = None

        with patch(
            "registry.api.virtual_server_routes.get_virtual_server_service",
            return_value=mock_vs_service,
        ):
            response = client.get("/api/virtual-servers/virtual/nonexistent")

        assert response.status_code == 404


class TestUpdateVirtualServer:
    """Tests for PUT /api/virtual-servers/{path}."""

    def test_update_success(self, client, mock_auth_admin, mock_vs_service):
        """Test updating a virtual server."""
        updated_config = VirtualServerConfig(
            path="/virtual/dev-essentials",
            server_name="Updated Name",
        )
        mock_vs_service.update_virtual_server.return_value = updated_config

        with patch(
            "registry.api.virtual_server_routes.get_virtual_server_service",
            return_value=mock_vs_service,
        ):
            response = client.put(
                "/api/virtual-servers/virtual/dev-essentials",
                json={"server_name": "Updated Name"},
            )

        assert response.status_code == 200
        assert response.json()["server_name"] == "Updated Name"

    def test_update_requires_admin(self, client, mock_auth_user, mock_vs_service):
        """Test that updating requires admin permissions."""
        with patch(
            "registry.api.virtual_server_routes.get_virtual_server_service",
            return_value=mock_vs_service,
        ):
            response = client.put(
                "/api/virtual-servers/virtual/dev-essentials",
                json={"server_name": "Updated"},
            )

        assert response.status_code == 403


class TestDeleteVirtualServer:
    """Tests for DELETE /api/virtual-servers/{path}."""

    def test_delete_success(self, client, mock_auth_admin, mock_vs_service):
        """Test deleting a virtual server."""
        mock_vs_service.delete_virtual_server.return_value = True

        with patch(
            "registry.api.virtual_server_routes.get_virtual_server_service",
            return_value=mock_vs_service,
        ):
            response = client.delete(
                "/api/virtual-servers/virtual/dev-essentials",
            )

        assert response.status_code == 204

    def test_delete_requires_admin(self, client, mock_auth_user, mock_vs_service):
        """Test that deleting requires admin permissions."""
        with patch(
            "registry.api.virtual_server_routes.get_virtual_server_service",
            return_value=mock_vs_service,
        ):
            response = client.delete(
                "/api/virtual-servers/virtual/dev-essentials",
            )

        assert response.status_code == 403

    def test_delete_not_found(self, client, mock_auth_admin, mock_vs_service):
        """Test deleting a nonexistent virtual server."""
        from registry.exceptions import VirtualServerNotFoundError

        mock_vs_service.delete_virtual_server.side_effect = VirtualServerNotFoundError(
            "/virtual/nonexistent"
        )

        with patch(
            "registry.api.virtual_server_routes.get_virtual_server_service",
            return_value=mock_vs_service,
        ):
            response = client.delete(
                "/api/virtual-servers/virtual/nonexistent",
            )

        assert response.status_code == 404


class TestToggleVirtualServer:
    """Tests for POST /api/virtual-servers/{path}/toggle."""

    def test_toggle_enable(self, client, mock_auth_admin, mock_vs_service):
        """Test enabling a virtual server."""
        mock_vs_service.toggle_virtual_server.return_value = True

        with patch(
            "registry.api.virtual_server_routes.get_virtual_server_service",
            return_value=mock_vs_service,
        ):
            response = client.post(
                "/api/virtual-servers/virtual/dev-essentials/toggle",
                json={"enabled": True},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["is_enabled"] is True

    def test_toggle_requires_admin(self, client, mock_auth_user, mock_vs_service):
        """Test that toggling requires admin permissions."""
        with patch(
            "registry.api.virtual_server_routes.get_virtual_server_service",
            return_value=mock_vs_service,
        ):
            response = client.post(
                "/api/virtual-servers/virtual/dev-essentials/toggle",
                json={"enabled": True},
            )

        assert response.status_code == 403

    def test_toggle_not_found(self, client, mock_auth_admin, mock_vs_service):
        """Test toggling a nonexistent virtual server returns 404."""
        from registry.exceptions import VirtualServerNotFoundError

        mock_vs_service.toggle_virtual_server.side_effect = VirtualServerNotFoundError(
            "/virtual/nonexistent"
        )

        with patch(
            "registry.api.virtual_server_routes.get_virtual_server_service",
            return_value=mock_vs_service,
        ):
            response = client.post(
                "/api/virtual-servers/virtual/nonexistent/toggle",
                json={"enabled": True},
            )

        assert response.status_code == 404

    def test_toggle_validation_error(self, client, mock_auth_admin, mock_vs_service):
        """Test toggling with validation error returns 400."""
        from registry.exceptions import VirtualServerValidationError

        mock_vs_service.toggle_virtual_server.side_effect = VirtualServerValidationError(
            "Cannot enable virtual server with no tool mappings"
        )

        with patch(
            "registry.api.virtual_server_routes.get_virtual_server_service",
            return_value=mock_vs_service,
        ):
            response = client.post(
                "/api/virtual-servers/virtual/empty/toggle",
                json={"enabled": True},
            )

        assert response.status_code == 400
        assert "no tool mappings" in response.json()["detail"]


class TestVirtualServerTools:
    """Tests for GET /api/virtual-servers/{path}/tools."""

    def test_get_tools(self, client, mock_auth_admin, mock_vs_service):
        """Test getting resolved tools for a virtual server."""
        from registry.schemas.virtual_server_models import ResolvedTool

        mock_vs_service.resolve_tools.return_value = [
            ResolvedTool(
                name="github_search",
                original_name="search",
                backend_server_path="/github",
                description="Search repos",
                input_schema={"type": "object"},
            ),
        ]

        with patch(
            "registry.api.virtual_server_routes.get_virtual_server_service",
            return_value=mock_vs_service,
        ):
            response = client.get(
                "/api/virtual-servers/virtual/dev-essentials/tools",
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 1
        assert data["tools"][0]["name"] == "github_search"

    def test_get_tools_not_found(self, client, mock_auth_admin, mock_vs_service):
        """Test getting tools for nonexistent server returns 404."""
        from registry.exceptions import VirtualServerNotFoundError

        mock_vs_service.resolve_tools.side_effect = VirtualServerNotFoundError(
            "/virtual/nonexistent"
        )

        with patch(
            "registry.api.virtual_server_routes.get_virtual_server_service",
            return_value=mock_vs_service,
        ):
            response = client.get(
                "/api/virtual-servers/virtual/nonexistent/tools",
            )

        assert response.status_code == 404


class TestUpdateVirtualServerErrors:
    """Additional tests for PUT /api/virtual-servers/{path} error cases."""

    @pytest.fixture
    def client(self):
        """Create FastAPI test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_auth_admin(self, client):
        """Mock authentication returning admin user context."""
        app.dependency_overrides[nginx_proxied_auth] = lambda: ADMIN_CONTEXT
        yield ADMIN_CONTEXT
        app.dependency_overrides.pop(nginx_proxied_auth, None)

    @pytest.fixture
    def mock_vs_service(self):
        """Mock virtual server service."""
        return AsyncMock()

    def test_update_not_found(self, client, mock_auth_admin, mock_vs_service):
        """Test updating a nonexistent virtual server returns 404."""
        from registry.exceptions import VirtualServerNotFoundError

        mock_vs_service.update_virtual_server.side_effect = VirtualServerNotFoundError(
            "/virtual/nonexistent"
        )

        with patch(
            "registry.api.virtual_server_routes.get_virtual_server_service",
            return_value=mock_vs_service,
        ):
            response = client.put(
                "/api/virtual-servers/virtual/nonexistent",
                json={"description": "Updated"},
            )

        assert response.status_code == 404

    def test_update_validation_error(self, client, mock_auth_admin, mock_vs_service):
        """Test updating with invalid data returns 400."""
        from registry.exceptions import VirtualServerValidationError

        mock_vs_service.update_virtual_server.side_effect = VirtualServerValidationError(
            "Tool mapping validation failed"
        )

        with patch(
            "registry.api.virtual_server_routes.get_virtual_server_service",
            return_value=mock_vs_service,
        ):
            response = client.put(
                "/api/virtual-servers/virtual/dev-essentials",
                json={"description": "Updated"},
            )

        assert response.status_code == 400


class TestToolCatalog:
    """Tests for GET /api/tool-catalog."""

    def test_get_catalog_empty(self, client, mock_auth_admin, mock_catalog_service):
        """Test getting tool catalog when empty."""
        with patch(
            "registry.api.virtual_server_routes.get_tool_catalog_service",
            return_value=mock_catalog_service,
        ):
            response = client.get("/api/tool-catalog")

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 0
        assert data["tools"] == []

    def test_get_catalog_with_filter(self, client, mock_auth_admin, mock_catalog_service):
        """Test getting tool catalog with server filter."""
        from registry.schemas.virtual_server_models import ToolCatalogEntry

        mock_catalog_service.get_tool_catalog.return_value = [
            ToolCatalogEntry(
                tool_name="search",
                server_path="/github",
                server_name="GitHub",
                description="Search repos",
            ),
        ]

        with patch(
            "registry.api.virtual_server_routes.get_tool_catalog_service",
            return_value=mock_catalog_service,
        ):
            response = client.get("/api/tool-catalog?server_path=/github")

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 1
        assert data["server_count"] == 1


class TestNormalizeVirtualPath:
    """Tests for _normalize_virtual_path edge cases."""

    def test_path_already_normalized(self):
        """Test that a fully qualified path is returned as-is."""
        assert _normalize_virtual_path("/virtual/dev-essentials") == "/virtual/dev-essentials"

    def test_path_without_leading_slash(self):
        """Test that virtual/... gets a leading slash prepended."""
        assert _normalize_virtual_path("virtual/dev-essentials") == "/virtual/dev-essentials"

    def test_bare_slug(self):
        """Test that a bare slug gets /virtual/ prefix."""
        assert _normalize_virtual_path("dev-essentials") == "/virtual/dev-essentials"

    def test_empty_path(self):
        """Test that an empty path is rejected as invalid."""
        with pytest.raises(HTTPException) as exc_info:
            _normalize_virtual_path("")
        assert exc_info.value.status_code == 400

    def test_path_with_double_dots(self):
        """Test path traversal attempt with '..' is rejected."""
        with pytest.raises(HTTPException) as exc_info:
            _normalize_virtual_path("../../etc/passwd")
        assert exc_info.value.status_code == 400
        assert "path traversal" in exc_info.value.detail

    def test_path_with_special_characters(self):
        """Test path with special characters is rejected."""
        with pytest.raises(HTTPException) as exc_info:
            _normalize_virtual_path("my-server_v2")
        assert exc_info.value.status_code == 400

    def test_path_that_is_just_virtual(self):
        """Test path that is just the word 'virtual'."""
        result = _normalize_virtual_path("virtual")
        assert result == "/virtual/virtual"

    def test_path_with_encoded_characters(self):
        """Test path with URL-encoded characters is rejected."""
        with pytest.raises(HTTPException) as exc_info:
            _normalize_virtual_path("my%20server")
        assert exc_info.value.status_code == 400

    def test_path_with_trailing_slash(self):
        """Test path with trailing slash is rejected."""
        with pytest.raises(HTTPException) as exc_info:
            _normalize_virtual_path("/virtual/dev-essentials/")
        assert exc_info.value.status_code == 400

    def test_path_with_nested_virtual(self):
        """Test path with sub-paths is rejected (only single slug allowed)."""
        with pytest.raises(HTTPException) as exc_info:
            _normalize_virtual_path("/virtual/sub/path")
        assert exc_info.value.status_code == 400


class TestRateVirtualServer:
    """Tests for POST /api/virtual-servers/{path}/rate."""

    def test_rate_success(self, client, mock_auth_user, mock_vs_service):
        """Test rating a virtual server successfully."""
        with patch(
            "registry.api.virtual_server_routes.get_virtual_server_service",
            return_value=mock_vs_service,
        ):
            response = client.post(
                "/api/virtual-servers/virtual/dev-essentials/rate",
                json={"rating": 4},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["average_rating"] == 4.0
        assert data["is_new_rating"] is True
        assert data["total_ratings"] == 1

    def test_rate_not_found(self, client, mock_auth_user, mock_vs_service):
        """Test rating a nonexistent virtual server returns 404."""
        from registry.exceptions import VirtualServerNotFoundError

        mock_vs_service.rate_virtual_server.side_effect = VirtualServerNotFoundError(
            "/virtual/nonexistent"
        )

        with patch(
            "registry.api.virtual_server_routes.get_virtual_server_service",
            return_value=mock_vs_service,
        ):
            response = client.post(
                "/api/virtual-servers/virtual/nonexistent/rate",
                json={"rating": 4},
            )

        assert response.status_code == 404

    def test_rate_invalid_rating(self, client, mock_auth_user, mock_vs_service):
        """Test rating with invalid value returns 400."""
        mock_vs_service.rate_virtual_server.side_effect = ValueError(
            "Rating must be between 1 and 5"
        )

        with patch(
            "registry.api.virtual_server_routes.get_virtual_server_service",
            return_value=mock_vs_service,
        ):
            response = client.post(
                "/api/virtual-servers/virtual/dev-essentials/rate",
                json={"rating": 10},
            )

        assert response.status_code == 400
        assert "between 1 and 5" in response.json()["detail"]

    def test_rate_update_existing(self, client, mock_auth_user, mock_vs_service):
        """Test updating an existing rating."""
        mock_vs_service.rate_virtual_server.return_value = {
            "average_rating": 5.0,
            "is_new_rating": False,
            "total_ratings": 1,
        }

        with patch(
            "registry.api.virtual_server_routes.get_virtual_server_service",
            return_value=mock_vs_service,
        ):
            response = client.post(
                "/api/virtual-servers/virtual/dev-essentials/rate",
                json={"rating": 5},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["is_new_rating"] is False


class TestGetVirtualServerRating:
    """Tests for GET /api/virtual-servers/{path}/rating."""

    def test_get_rating_success(self, client, mock_auth_user, mock_vs_service):
        """Test getting rating information successfully."""
        with patch(
            "registry.api.virtual_server_routes.get_virtual_server_service",
            return_value=mock_vs_service,
        ):
            response = client.get(
                "/api/virtual-servers/virtual/dev-essentials/rating",
            )

        assert response.status_code == 200
        data = response.json()
        assert data["num_stars"] == 4.0
        assert len(data["rating_details"]) == 1
        assert data["rating_details"][0]["user"] == "testuser"

    def test_get_rating_not_found(self, client, mock_auth_user, mock_vs_service):
        """Test getting rating for nonexistent virtual server returns 404."""
        from registry.exceptions import VirtualServerNotFoundError

        mock_vs_service.get_virtual_server_rating.side_effect = VirtualServerNotFoundError(
            "/virtual/nonexistent"
        )

        with patch(
            "registry.api.virtual_server_routes.get_virtual_server_service",
            return_value=mock_vs_service,
        ):
            response = client.get(
                "/api/virtual-servers/virtual/nonexistent/rating",
            )

        assert response.status_code == 404

    def test_get_rating_empty(self, client, mock_auth_user, mock_vs_service):
        """Test getting rating for server with no ratings."""
        mock_vs_service.get_virtual_server_rating.return_value = {
            "num_stars": 0.0,
            "rating_details": [],
        }

        with patch(
            "registry.api.virtual_server_routes.get_virtual_server_service",
            return_value=mock_vs_service,
        ):
            response = client.get(
                "/api/virtual-servers/virtual/dev-essentials/rating",
            )

        assert response.status_code == 200
        data = response.json()
        assert data["num_stars"] == 0.0
        assert data["rating_details"] == []
