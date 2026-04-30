"""
Unit tests for Federation Export API endpoints.

Tests the visibility-based access control, incremental sync, pagination,
and authentication requirements for federation endpoints.
"""

from typing import (
    Any,
)
from unittest.mock import (
    Mock,
    patch,
)

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from registry.api import federation_export_routes
from registry.main import app
from registry.services.agent_service import agent_service
from registry.services.server_service import server_service


@pytest.fixture
def mock_federation_auth():
    """Mock nginx_proxied_auth for federation peer with federation-service scope."""

    def _mock_auth(
        request=None, session=None, x_user=None, x_username=None, x_scopes=None, x_auth_method=None
    ):
        return {
            "username": "peer-registry-1",
            "groups": ["engineering", "finance"],
            "scopes": ["federation-service", "mcp-servers-restricted/read"],
            "auth_method": "oauth2",
            "provider": "keycloak",
            "accessible_servers": [],
            "accessible_services": ["all"],
            "can_modify_servers": False,
            "is_admin": False,
        }

    return _mock_auth


@pytest.fixture
def mock_federation_auth_no_groups():
    """Mock nginx_proxied_auth for federation peer with no groups."""

    def _mock_auth(
        request=None, session=None, x_user=None, x_username=None, x_scopes=None, x_auth_method=None
    ):
        return {
            "username": "peer-registry-public",
            "groups": [],
            "scopes": ["federation-service"],
            "auth_method": "oauth2",
            "provider": "keycloak",
            "accessible_servers": [],
            "accessible_services": ["all"],
            "can_modify_servers": False,
            "is_admin": False,
        }

    return _mock_auth


@pytest.fixture
def mock_federation_auth_missing_scope():
    """Mock nginx_proxied_auth for peer WITHOUT federation-service scope."""

    def _mock_auth(
        request=None, session=None, x_user=None, x_username=None, x_scopes=None, x_auth_method=None
    ):
        return {
            "username": "unauthorized-peer",
            "groups": ["engineering"],
            "scopes": ["mcp-servers-restricted/read"],
            "auth_method": "oauth2",
            "provider": "keycloak",
            "accessible_servers": [],
            "accessible_services": ["all"],
            "can_modify_servers": False,
            "is_admin": False,
        }

    return _mock_auth


@pytest.fixture
def sample_server_public() -> dict[str, Any]:
    """Create a public server for testing."""
    return {
        "path": "/public-server",
        "name": "Public Server",
        "description": "Public server available to all",
        "visibility": "public",
        "allowed_groups": [],
        "sync_metadata": {
            "sync_generation": 10,
            "last_synced_at": "2024-01-15T10:30:00Z",
        },
    }


@pytest.fixture
def sample_server_group_restricted() -> dict[str, Any]:
    """Create a group-restricted server for testing."""
    return {
        "path": "/finance-server",
        "name": "Finance Server",
        "description": "Finance team only",
        "visibility": "group-restricted",
        "allowed_groups": ["finance"],
        "sync_metadata": {
            "sync_generation": 15,
            "last_synced_at": "2024-01-15T10:30:00Z",
        },
    }


@pytest.fixture
def sample_server_internal() -> dict[str, Any]:
    """Create an internal server that should never be exported."""
    return {
        "path": "/internal-server",
        "name": "Internal Server",
        "description": "Internal only, never exported",
        "visibility": "internal",
        "allowed_groups": [],
        "sync_metadata": {
            "sync_generation": 20,
            "last_synced_at": "2024-01-15T10:30:00Z",
        },
    }


@pytest.fixture
def sample_agent_public() -> dict[str, Any]:
    """Create a public agent for testing."""
    return {
        "path": "/agents/public-agent",
        "name": "Public Agent",
        "description": "Public agent available to all",
        "visibility": "public",
        "allowed_groups": [],
        "sync_metadata": {
            "sync_generation": 5,
            "last_synced_at": "2024-01-15T10:30:00Z",
        },
    }


@pytest.fixture
def sample_agent_group_restricted() -> dict[str, Any]:
    """Create a group-restricted agent for testing."""
    return {
        "path": "/agents/engineering-agent",
        "name": "Engineering Agent",
        "description": "Engineering team only",
        "visibility": "group-restricted",
        "allowed_groups": ["engineering"],
        "sync_metadata": {
            "sync_generation": 8,
            "last_synced_at": "2024-01-15T10:30:00Z",
        },
    }


@pytest.mark.unit
class TestFederationHealth:
    """Test suite for GET /api/federation/health endpoint."""

    def test_health_returns_200(self) -> None:
        """Test health endpoint returns 200 when registry is healthy (2.SC9)."""
        client = TestClient(app)
        response = client.get("/api/federation/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["status"] == "healthy"
        assert "federation_api_version" in data
        assert "registry_id" in data

    def test_health_no_auth_required(self) -> None:
        """Test health endpoint does NOT require authentication."""
        # Health endpoint should work without any auth
        client = TestClient(app)
        response = client.get("/api/federation/health")

        assert response.status_code == status.HTTP_200_OK


@pytest.mark.unit
class TestFederationAuthRequirements:
    """Test suite for federation authentication requirements."""

    def test_export_servers_requires_auth(self) -> None:
        """Test unauthenticated requests to /api/federation/servers return 401 (2.SC1)."""
        from fastapi import HTTPException

        from registry.auth.dependencies import nginx_proxied_auth

        def _mock_no_auth(
            request=None,
            session=None,
            x_user=None,
            x_username=None,
            x_scopes=None,
            x_auth_method=None,
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
            )

        app.dependency_overrides[nginx_proxied_auth] = _mock_no_auth

        client = TestClient(app)
        response = client.get("/api/federation/servers")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

        app.dependency_overrides.clear()

    def test_export_agents_requires_auth(self) -> None:
        """Test unauthenticated requests to /api/federation/agents return 401 (2.SC1)."""
        from fastapi import HTTPException

        from registry.auth.dependencies import nginx_proxied_auth

        def _mock_no_auth(
            request=None,
            session=None,
            x_user=None,
            x_username=None,
            x_scopes=None,
            x_auth_method=None,
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
            )

        app.dependency_overrides[nginx_proxied_auth] = _mock_no_auth

        client = TestClient(app)
        response = client.get("/api/federation/agents")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

        app.dependency_overrides.clear()

    def test_missing_federation_scope_returns_403(
        self,
        mock_federation_auth_missing_scope: Any,
    ) -> None:
        """Test requests without federation-service scope return 403 (2.SC2)."""
        from registry.auth.dependencies import nginx_proxied_auth

        app.dependency_overrides[nginx_proxied_auth] = mock_federation_auth_missing_scope

        client = TestClient(app)
        response = client.get("/api/federation/servers")

        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert "federation-service" in response.json()["detail"]

        app.dependency_overrides.clear()


@pytest.mark.unit
class TestVisibilityFiltering:
    """Test suite for visibility-based filtering logic."""

    def test_public_items_returned_to_all_peers(
        self,
        mock_federation_auth_no_groups: Any,
        sample_server_public: dict[str, Any],
    ) -> None:
        """Test visibility=public items are returned to peers with no groups (2.SC3)."""
        from registry.auth.dependencies import nginx_proxied_auth

        app.dependency_overrides[nginx_proxied_auth] = mock_federation_auth_no_groups

        # Mock server service to return public server
        servers_dict = {sample_server_public["path"]: sample_server_public}

        with (
            patch.object(
                server_service,
                "get_all_servers",
                return_value=servers_dict,
            ),
            patch.object(
                server_service,
                "is_service_enabled",
                return_value=True,
            ),
        ):
            client = TestClient(app)
            response = client.get("/api/federation/servers")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert len(data["items"]) == 1
            assert data["items"][0]["path"] == "/public-server"

        app.dependency_overrides.clear()

    def test_group_restricted_returned_if_peer_in_group(
        self,
        mock_federation_auth: Any,
        sample_server_group_restricted: dict[str, Any],
    ) -> None:
        """Test group-restricted items returned only if peer is in allowed_groups (2.SC4)."""
        from registry.auth.dependencies import nginx_proxied_auth

        app.dependency_overrides[nginx_proxied_auth] = mock_federation_auth

        # Mock auth returns groups: ["engineering", "finance"]
        # Server has allowed_groups: ["finance"]
        servers_dict = {sample_server_group_restricted["path"]: sample_server_group_restricted}

        with (
            patch.object(
                server_service,
                "get_all_servers",
                return_value=servers_dict,
            ),
            patch.object(
                server_service,
                "is_service_enabled",
                return_value=True,
            ),
        ):
            client = TestClient(app)
            response = client.get("/api/federation/servers")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # Should be returned because peer is in "finance" group
            assert len(data["items"]) == 1
            assert data["items"][0]["path"] == "/finance-server"

        app.dependency_overrides.clear()

    def test_group_restricted_not_returned_if_peer_not_in_group(
        self,
        mock_federation_auth_no_groups: Any,
        sample_server_group_restricted: dict[str, Any],
    ) -> None:
        """Test group-restricted items NOT returned if peer is not in allowed_groups (2.SC4)."""
        from registry.auth.dependencies import nginx_proxied_auth

        app.dependency_overrides[nginx_proxied_auth] = mock_federation_auth_no_groups

        # Mock auth returns groups: []
        # Server has allowed_groups: ["finance"]
        servers_dict = {sample_server_group_restricted["path"]: sample_server_group_restricted}

        with (
            patch.object(
                server_service,
                "get_all_servers",
                return_value=servers_dict,
            ),
            patch.object(
                server_service,
                "is_service_enabled",
                return_value=True,
            ),
        ):
            client = TestClient(app)
            response = client.get("/api/federation/servers")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # Should NOT be returned because peer is not in "finance" group
            assert len(data["items"]) == 0

        app.dependency_overrides.clear()

    def test_internal_items_never_returned(
        self,
        mock_federation_auth: Any,
        sample_server_internal: dict[str, Any],
    ) -> None:
        """Test visibility=internal items are NEVER returned (2.SC5)."""
        from registry.auth.dependencies import nginx_proxied_auth

        app.dependency_overrides[nginx_proxied_auth] = mock_federation_auth

        servers_dict = {sample_server_internal["path"]: sample_server_internal}

        with (
            patch.object(
                server_service,
                "get_all_servers",
                return_value=servers_dict,
            ),
            patch.object(
                server_service,
                "is_service_enabled",
                return_value=True,
            ),
        ):
            client = TestClient(app)
            response = client.get("/api/federation/servers")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # Internal items should NEVER be returned
            assert len(data["items"]) == 0

        app.dependency_overrides.clear()

    def test_mixed_visibility_filtering(
        self,
        mock_federation_auth: Any,
        sample_server_public: dict[str, Any],
        sample_server_group_restricted: dict[str, Any],
        sample_server_internal: dict[str, Any],
    ) -> None:
        """Test filtering with mixed visibility items."""
        from registry.auth.dependencies import nginx_proxied_auth

        app.dependency_overrides[nginx_proxied_auth] = mock_federation_auth

        # Peer has groups: ["engineering", "finance"]
        servers_dict = {
            sample_server_public["path"]: sample_server_public,
            sample_server_group_restricted["path"]: sample_server_group_restricted,
            sample_server_internal["path"]: sample_server_internal,
        }

        with (
            patch.object(
                server_service,
                "get_all_servers",
                return_value=servers_dict,
            ),
            patch.object(
                server_service,
                "is_service_enabled",
                return_value=True,
            ),
        ):
            client = TestClient(app)
            response = client.get("/api/federation/servers")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # Should return public + group-restricted (peer in finance group)
            # Should NOT return internal
            assert len(data["items"]) == 2
            paths = [item["path"] for item in data["items"]]
            assert "/public-server" in paths
            assert "/finance-server" in paths
            assert "/internal-server" not in paths

        app.dependency_overrides.clear()


@pytest.mark.unit
class TestIncrementalSync:
    """Test suite for incremental sync with generation numbers."""

    def test_since_generation_filters_items(
        self,
        mock_federation_auth: Any,
        sample_server_public: dict[str, Any],
    ) -> None:
        """Test since_generation param returns only items with generation > param value (2.SC6)."""
        from registry.auth.dependencies import nginx_proxied_auth

        app.dependency_overrides[nginx_proxied_auth] = mock_federation_auth

        # Server has sync_generation: 10
        servers_dict = {sample_server_public["path"]: sample_server_public}

        with (
            patch.object(
                server_service,
                "get_all_servers",
                return_value=servers_dict,
            ),
            patch.object(
                server_service,
                "is_service_enabled",
                return_value=True,
            ),
        ):
            client = TestClient(app)

            # Request with since_generation=5 (should return server with gen 10)
            response = client.get("/api/federation/servers?since_generation=5")
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data["items"]) == 1

            # Request with since_generation=10 (should NOT return server with gen 10)
            response = client.get("/api/federation/servers?since_generation=10")
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data["items"]) == 0

            # Request with since_generation=15 (should NOT return server with gen 10)
            response = client.get("/api/federation/servers?since_generation=15")
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data["items"]) == 0

        app.dependency_overrides.clear()

    def test_since_generation_zero_returns_all(
        self,
        mock_federation_auth: Any,
        sample_server_public: dict[str, Any],
    ) -> None:
        """Test since_generation=0 returns all items (2.SC6)."""
        from registry.auth.dependencies import nginx_proxied_auth

        app.dependency_overrides[nginx_proxied_auth] = mock_federation_auth

        servers_dict = {sample_server_public["path"]: sample_server_public}

        with (
            patch.object(
                server_service,
                "get_all_servers",
                return_value=servers_dict,
            ),
            patch.object(
                server_service,
                "is_service_enabled",
                return_value=True,
            ),
        ):
            client = TestClient(app)
            response = client.get("/api/federation/servers?since_generation=0")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # Should return all items
            assert len(data["items"]) == 1

        app.dependency_overrides.clear()

    def test_response_includes_sync_generation(
        self,
        mock_federation_auth: Any,
        sample_server_public: dict[str, Any],
    ) -> None:
        """Test response includes sync_generation for incremental sync (2.SC8)."""
        from registry.auth.dependencies import nginx_proxied_auth

        app.dependency_overrides[nginx_proxied_auth] = mock_federation_auth

        servers_dict = {sample_server_public["path"]: sample_server_public}

        with (
            patch.object(
                server_service,
                "get_all_servers",
                return_value=servers_dict,
            ),
            patch.object(
                server_service,
                "is_service_enabled",
                return_value=True,
            ),
        ):
            client = TestClient(app)
            response = client.get("/api/federation/servers")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # Response must include sync_generation
            assert "sync_generation" in data
            assert isinstance(data["sync_generation"], int)

        app.dependency_overrides.clear()


@pytest.mark.unit
class TestPagination:
    """Test suite for pagination functionality."""

    def test_pagination_limit_offset(
        self,
        mock_federation_auth: Any,
    ) -> None:
        """Test pagination works correctly with limit and offset (2.SC7)."""
        from registry.auth.dependencies import nginx_proxied_auth

        app.dependency_overrides[nginx_proxied_auth] = mock_federation_auth

        # Create multiple servers for pagination testing
        servers_dict = {}
        for i in range(5):
            servers_dict[f"/server-{i}"] = {
                "path": f"/server-{i}",
                "name": f"Server {i}",
                "visibility": "public",
                "allowed_groups": [],
            }

        with (
            patch.object(
                server_service,
                "get_all_servers",
                return_value=servers_dict,
            ),
            patch.object(
                server_service,
                "is_service_enabled",
                return_value=True,
            ),
        ):
            client = TestClient(app)

            # Test limit
            response = client.get("/api/federation/servers?limit=2")
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data["items"]) == 2
            assert data["has_more"] is True

            # Test offset
            response = client.get("/api/federation/servers?limit=2&offset=2")
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data["items"]) == 2
            assert data["has_more"] is True

            # Test last page
            response = client.get("/api/federation/servers?limit=2&offset=4")
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data["items"]) == 1
            assert data["has_more"] is False

        app.dependency_overrides.clear()

    def test_limit_exceeds_max(
        self,
        mock_federation_auth: Any,
    ) -> None:
        """Test limit parameter is capped at max 1000."""
        from registry.auth.dependencies import nginx_proxied_auth

        app.dependency_overrides[nginx_proxied_auth] = mock_federation_auth

        with patch.object(
            server_service,
            "get_all_servers",
            return_value={},
        ):
            client = TestClient(app)
            # Requesting limit=2000 should be rejected by validation
            response = client.get("/api/federation/servers?limit=2000")

            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        app.dependency_overrides.clear()

    def test_pagination_metadata(
        self,
        mock_federation_auth: Any,
        sample_server_public: dict[str, Any],
    ) -> None:
        """Test pagination metadata in response."""
        from registry.auth.dependencies import nginx_proxied_auth

        app.dependency_overrides[nginx_proxied_auth] = mock_federation_auth

        servers_dict = {sample_server_public["path"]: sample_server_public}

        with (
            patch.object(
                server_service,
                "get_all_servers",
                return_value=servers_dict,
            ),
            patch.object(
                server_service,
                "is_service_enabled",
                return_value=True,
            ),
        ):
            client = TestClient(app)
            response = client.get("/api/federation/servers")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # Response must include pagination metadata
            assert "total_count" in data
            assert "has_more" in data
            assert data["total_count"] == 1
            assert data["has_more"] is False

        app.dependency_overrides.clear()


@pytest.mark.unit
class TestEmptyRegistry:
    """Test suite for empty registry edge case."""

    def test_empty_registry_returns_empty_list(
        self,
        mock_federation_auth: Any,
    ) -> None:
        """Test empty registry returns empty list, not error (2.SC11)."""
        from registry.auth.dependencies import nginx_proxied_auth

        app.dependency_overrides[nginx_proxied_auth] = mock_federation_auth

        with patch.object(
            server_service,
            "get_all_servers",
            return_value={},
        ):
            client = TestClient(app)
            response = client.get("/api/federation/servers")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert data["items"] == []
            assert data["total_count"] == 0
            assert data["has_more"] is False

        app.dependency_overrides.clear()

    def test_empty_agents_returns_empty_list(
        self,
        mock_federation_auth: Any,
    ) -> None:
        """Test empty agents registry returns empty list, not error (2.SC11)."""
        from registry.auth.dependencies import nginx_proxied_auth

        app.dependency_overrides[nginx_proxied_auth] = mock_federation_auth

        with patch.object(
            agent_service,
            "get_all_agents",
            return_value=[],
        ):
            client = TestClient(app)
            response = client.get("/api/federation/agents")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert data["items"] == []
            assert data["total_count"] == 0
            assert data["has_more"] is False

        app.dependency_overrides.clear()


@pytest.mark.unit
class TestAgentsEndpoint:
    """Test suite for GET /api/federation/agents endpoint."""

    def test_export_agents_success(
        self,
        mock_federation_auth: Any,
        sample_agent_public: dict[str, Any],
    ) -> None:
        """Test exporting agents with proper visibility filtering."""
        from registry.auth.dependencies import nginx_proxied_auth

        app.dependency_overrides[nginx_proxied_auth] = mock_federation_auth

        # Create mock agent objects (agents are objects, not dicts)
        mock_agent = Mock()
        mock_agent.path = sample_agent_public["path"]
        mock_agent.name = sample_agent_public["name"]
        mock_agent.visibility = sample_agent_public["visibility"]
        mock_agent.allowed_groups = sample_agent_public["allowed_groups"]
        mock_agent.sync_metadata = sample_agent_public["sync_metadata"]
        mock_agent.model_dump = Mock(return_value=sample_agent_public)

        with (
            patch.object(
                agent_service,
                "get_all_agents",
                return_value=[mock_agent],
            ),
            patch.object(
                agent_service,
                "get_all_agent_states",
                return_value={"/agents/public-agent": True},
            ),
        ):
            client = TestClient(app)
            response = client.get("/api/federation/agents")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert len(data["items"]) == 1
            assert data["items"][0]["path"] == "/agents/public-agent"

        app.dependency_overrides.clear()

    def test_export_agents_visibility_filtering(
        self,
        mock_federation_auth: Any,
        sample_agent_public: dict[str, Any],
        sample_agent_group_restricted: dict[str, Any],
    ) -> None:
        """Test agents visibility filtering works correctly."""
        from registry.auth.dependencies import nginx_proxied_auth

        app.dependency_overrides[nginx_proxied_auth] = mock_federation_auth

        # Create mock agent objects
        mock_agent_public = Mock()
        mock_agent_public.path = sample_agent_public["path"]
        mock_agent_public.visibility = sample_agent_public["visibility"]
        mock_agent_public.allowed_groups = sample_agent_public["allowed_groups"]
        mock_agent_public.sync_metadata = sample_agent_public["sync_metadata"]
        mock_agent_public.model_dump = Mock(return_value=sample_agent_public)

        mock_agent_restricted = Mock()
        mock_agent_restricted.path = sample_agent_group_restricted["path"]
        mock_agent_restricted.visibility = sample_agent_group_restricted["visibility"]
        mock_agent_restricted.allowed_groups = sample_agent_group_restricted["allowed_groups"]
        mock_agent_restricted.sync_metadata = sample_agent_group_restricted["sync_metadata"]
        mock_agent_restricted.model_dump = Mock(return_value=sample_agent_group_restricted)

        # Peer has groups: ["engineering", "finance"]
        with (
            patch.object(
                agent_service,
                "get_all_agents",
                return_value=[mock_agent_public, mock_agent_restricted],
            ),
            patch.object(
                agent_service,
                "get_all_agent_states",
                return_value={
                    "/agents/public-agent": True,
                    "/agents/engineering-agent": True,
                },
            ),
        ):
            client = TestClient(app)
            response = client.get("/api/federation/agents")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # Should return both: public + engineering-restricted (peer in engineering group)
            assert len(data["items"]) == 2
            paths = [item["path"] for item in data["items"]]
            assert "/agents/public-agent" in paths
            assert "/agents/engineering-agent" in paths

        app.dependency_overrides.clear()


@pytest.mark.unit
class TestHelperFunctions:
    """Test suite for internal helper functions."""

    def test_get_item_attr_dict(self) -> None:
        """Test _get_item_attr() with dict input."""
        item = {"name": "test", "value": 42}

        assert federation_export_routes._get_item_attr(item, "name") == "test"
        assert federation_export_routes._get_item_attr(item, "value") == 42
        assert federation_export_routes._get_item_attr(item, "missing", "default") == "default"

    def test_get_item_attr_object(self) -> None:
        """Test _get_item_attr() with object input."""
        mock_obj = Mock(spec=["name", "value"])
        mock_obj.name = "test"
        mock_obj.value = 42

        assert federation_export_routes._get_item_attr(mock_obj, "name") == "test"
        assert federation_export_routes._get_item_attr(mock_obj, "value") == 42
        assert federation_export_routes._get_item_attr(mock_obj, "missing", "default") == "default"

    def test_filter_by_visibility_public_only(self) -> None:
        """Test _filter_by_visibility() returns only public items to peers with no groups."""
        items = [
            {"visibility": "public", "path": "/public"},
            {
                "visibility": "group-restricted",
                "allowed_groups": ["finance"],
                "path": "/restricted",
            },
            {"visibility": "internal", "path": "/internal"},
        ]

        filtered = federation_export_routes._filter_by_visibility(items, [])

        assert len(filtered) == 1
        assert filtered[0]["path"] == "/public"

    def test_filter_by_visibility_group_match(self) -> None:
        """Test _filter_by_visibility() returns group-restricted if peer in group."""
        items = [
            {"visibility": "public", "path": "/public"},
            {"visibility": "group-restricted", "allowed_groups": ["finance"], "path": "/finance"},
            {
                "visibility": "group-restricted",
                "allowed_groups": ["engineering"],
                "path": "/engineering",
            },
        ]

        filtered = federation_export_routes._filter_by_visibility(items, ["finance"])

        assert len(filtered) == 2
        paths = [item["path"] for item in filtered]
        assert "/public" in paths
        assert "/finance" in paths
        assert "/engineering" not in paths

    def test_filter_by_visibility_multiple_groups(self) -> None:
        """Test peer with multiple groups gets union of allowed items."""
        items = [
            {"visibility": "group-restricted", "allowed_groups": ["finance"], "path": "/finance"},
            {
                "visibility": "group-restricted",
                "allowed_groups": ["engineering"],
                "path": "/engineering",
            },
            {"visibility": "group-restricted", "allowed_groups": ["hr"], "path": "/hr"},
        ]

        filtered = federation_export_routes._filter_by_visibility(items, ["finance", "engineering"])

        assert len(filtered) == 2
        paths = [item["path"] for item in filtered]
        assert "/finance" in paths
        assert "/engineering" in paths
        assert "/hr" not in paths

    def test_filter_by_visibility_empty_allowed_groups(self) -> None:
        """Test group-restricted with empty allowed_groups returns to no one."""
        items = [
            {"visibility": "group-restricted", "allowed_groups": [], "path": "/restricted"},
        ]

        # Even if peer has groups, empty allowed_groups means no match
        filtered = federation_export_routes._filter_by_visibility(items, ["finance", "engineering"])

        assert len(filtered) == 0

    def test_filter_by_visibility_no_visibility_field(self) -> None:
        """Test items with no visibility field default to public (backwards compatibility)."""
        items = [
            {"path": "/no-visibility"},
        ]

        filtered = federation_export_routes._filter_by_visibility(items, [])

        # Should default to public and be exported (backwards compatibility)
        assert len(filtered) == 1
        assert filtered[0]["path"] == "/no-visibility"

    def test_filter_by_generation_filters_correctly(self) -> None:
        """Test _filter_by_generation() filters items correctly."""
        items = [
            {"path": "/item1", "sync_metadata": {"sync_generation": 5}},
            {"path": "/item2", "sync_metadata": {"sync_generation": 10}},
            {"path": "/item3", "sync_metadata": {"sync_generation": 15}},
        ]

        # since_generation=10 should return only items with generation > 10
        filtered = federation_export_routes._filter_by_generation(items, 10)

        assert len(filtered) == 1
        assert filtered[0]["path"] == "/item3"

    def test_filter_by_generation_none_returns_all(self) -> None:
        """Test _filter_by_generation() with None returns all items."""
        items = [
            {"path": "/item1", "sync_metadata": {"sync_generation": 5}},
            {"path": "/item2", "sync_metadata": {"sync_generation": 10}},
        ]

        filtered = federation_export_routes._filter_by_generation(items, None)

        assert len(filtered) == 2

    def test_filter_by_generation_missing_metadata(self) -> None:
        """Test _filter_by_generation() includes items without sync_metadata.

        Items without sync_metadata are local items that have never been
        synced - they should always be included as they're "new" to the peer.
        """
        items = [
            {"path": "/item1"},  # No sync_metadata - local item
            {"path": "/item2", "sync_metadata": {"sync_generation": 10}},
        ]

        # Items without sync_metadata are always included (local items)
        filtered = federation_export_routes._filter_by_generation(items, 0)

        # Both should be returned: item1 (local) and item2 (generation 10 > 0)
        assert len(filtered) == 2
        paths = [item["path"] for item in filtered]
        assert "/item1" in paths
        assert "/item2" in paths

    def test_item_to_dict_dict(self) -> None:
        """Test _item_to_dict() with dict input."""
        item = {"path": "/test", "name": "Test"}

        result = federation_export_routes._item_to_dict(item)

        assert result == item

    def test_item_to_dict_pydantic(self) -> None:
        """Test _item_to_dict() with Pydantic model."""
        mock_model = Mock()
        mock_model.model_dump = Mock(return_value={"path": "/test", "name": "Test"})

        result = federation_export_routes._item_to_dict(mock_model)

        assert result == {"path": "/test", "name": "Test"}
        mock_model.model_dump.assert_called_once()

    def test_paginate_first_page(self) -> None:
        """Test _paginate() returns first page correctly."""
        items = [f"item{i}" for i in range(10)]

        paginated, has_more = federation_export_routes._paginate(items, limit=3, offset=0)

        assert len(paginated) == 3
        assert paginated == ["item0", "item1", "item2"]
        assert has_more is True

    def test_paginate_middle_page(self) -> None:
        """Test _paginate() returns middle page correctly."""
        items = [f"item{i}" for i in range(10)]

        paginated, has_more = federation_export_routes._paginate(items, limit=3, offset=3)

        assert len(paginated) == 3
        assert paginated == ["item3", "item4", "item5"]
        assert has_more is True

    def test_paginate_last_page(self) -> None:
        """Test _paginate() returns last page correctly."""
        items = [f"item{i}" for i in range(10)]

        paginated, has_more = federation_export_routes._paginate(items, limit=3, offset=9)

        assert len(paginated) == 1
        assert paginated == ["item9"]
        assert has_more is False

    def test_check_federation_scope_valid(self) -> None:
        """Test _check_federation_scope() passes with valid scope."""
        user_context = {
            "username": "test-peer",
            "scopes": ["federation-service", "other-scope"],
        }

        # Should not raise exception
        federation_export_routes._check_federation_scope(user_context)

    def test_check_federation_scope_invalid(self) -> None:
        """Test _check_federation_scope() raises 403 without scope."""
        from fastapi import HTTPException

        user_context = {
            "username": "test-peer",
            "scopes": ["other-scope"],
        }

        with pytest.raises(HTTPException) as exc_info:
            federation_export_routes._check_federation_scope(user_context)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "federation-service" in str(exc_info.value.detail)


@pytest.mark.unit
class TestDisabledItemsFiltering:
    """Test suite for filtering disabled servers and agents."""

    def test_disabled_servers_not_exported(
        self,
        mock_federation_auth: Any,
        sample_server_public: dict[str, Any],
    ) -> None:
        """Test disabled servers are never exported."""
        from registry.auth.dependencies import nginx_proxied_auth

        app.dependency_overrides[nginx_proxied_auth] = mock_federation_auth

        servers_dict = {sample_server_public["path"]: sample_server_public}

        with (
            patch.object(
                server_service,
                "get_all_servers",
                return_value=servers_dict,
            ),
            patch.object(
                server_service,
                "is_service_enabled",
                return_value=False,  # Server is disabled
            ),
        ):
            client = TestClient(app)
            response = client.get("/api/federation/servers")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # Disabled server should not be exported
            assert len(data["items"]) == 0

        app.dependency_overrides.clear()

    def test_disabled_agents_not_exported(
        self,
        mock_federation_auth: Any,
        sample_agent_public: dict[str, Any],
    ) -> None:
        """Test disabled agents are never exported."""
        from registry.auth.dependencies import nginx_proxied_auth

        app.dependency_overrides[nginx_proxied_auth] = mock_federation_auth

        mock_agent = Mock()
        mock_agent.path = sample_agent_public["path"]
        mock_agent.visibility = sample_agent_public["visibility"]
        mock_agent.allowed_groups = sample_agent_public["allowed_groups"]

        with (
            patch.object(
                agent_service,
                "get_all_agents",
                return_value=[mock_agent],
            ),
            patch.object(
                agent_service,
                "get_all_agent_states",
                return_value={"/agents/public-agent": False},
            ),
        ):
            client = TestClient(app)
            response = client.get("/api/federation/agents")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # Disabled agent should not be exported
            assert len(data["items"]) == 0

        app.dependency_overrides.clear()


@pytest.mark.unit
class TestChainPrevention:
    """Test suite for chain prevention (A->B->C scenario).

    When registry B syncs items from registry A, those items should NOT be
    re-exported from B to registry C. This prevents federation chains and
    ensures items only come from their original source.
    """

    def test_is_federated_item_with_dict(self) -> None:
        """Test _is_federated_item() detects federated dict items."""
        # Item synced from another peer
        federated_item = {
            "path": "/peer-a/server1",
            "sync_metadata": {
                "is_federated": True,
                "source_peer_id": "peer-a",
            },
        }

        assert federation_export_routes._is_federated_item(federated_item) is True

    def test_is_federated_item_with_object(self) -> None:
        """Test _is_federated_item() detects federated object items."""
        mock_item = Mock()
        mock_item.sync_metadata = Mock()
        mock_item.sync_metadata.is_federated = True

        assert federation_export_routes._is_federated_item(mock_item) is True

    def test_is_federated_item_local_item(self) -> None:
        """Test _is_federated_item() returns False for local items."""
        # Local item with no sync_metadata
        local_item = {
            "path": "/my-local-server",
        }

        assert federation_export_routes._is_federated_item(local_item) is False

    def test_is_federated_item_local_with_sync_metadata(self) -> None:
        """Test _is_federated_item() returns False for local items with sync_metadata."""
        # Local item that has sync_metadata but is_federated is False
        local_item = {
            "path": "/my-local-server",
            "sync_metadata": {
                "is_federated": False,
                "sync_generation": 5,
            },
        }

        assert federation_export_routes._is_federated_item(local_item) is False

    def test_is_federated_item_no_is_federated_field(self) -> None:
        """Test _is_federated_item() returns False when is_federated field missing."""
        item = {
            "path": "/server",
            "sync_metadata": {
                "sync_generation": 10,
            },
        }

        assert federation_export_routes._is_federated_item(item) is False

    def test_filter_by_visibility_excludes_federated_items(self) -> None:
        """Test _filter_by_visibility() excludes federated items (chain prevention)."""
        items = [
            # Local public server - should be exported
            {"path": "/local-public", "visibility": "public"},
            # Federated server from peer-a - should NOT be exported
            {
                "path": "/peer-a/server1",
                "visibility": "public",
                "sync_metadata": {
                    "is_federated": True,
                    "source_peer_id": "peer-a",
                },
            },
            # Another federated server - should NOT be exported
            {
                "path": "/peer-b/server2",
                "visibility": "public",
                "sync_metadata": {
                    "is_federated": True,
                    "source_peer_id": "peer-b",
                },
            },
        ]

        filtered = federation_export_routes._filter_by_visibility(items, [])

        # Only local item should be returned
        assert len(filtered) == 1
        assert filtered[0]["path"] == "/local-public"

    def test_filter_by_visibility_mixed_local_and_federated(self) -> None:
        """Test filtering with mix of local and federated items."""
        items = [
            # Local public
            {"path": "/local-public", "visibility": "public"},
            # Local group-restricted
            {
                "path": "/local-finance",
                "visibility": "group-restricted",
                "allowed_groups": ["finance"],
            },
            # Local internal
            {"path": "/local-internal", "visibility": "internal"},
            # Federated public
            {
                "path": "/peer-a/public",
                "visibility": "public",
                "sync_metadata": {"is_federated": True},
            },
            # Federated group-restricted
            {
                "path": "/peer-a/finance",
                "visibility": "group-restricted",
                "allowed_groups": ["finance"],
                "sync_metadata": {"is_federated": True},
            },
        ]

        # Peer has finance group
        filtered = federation_export_routes._filter_by_visibility(items, ["finance"])

        # Should return local public + local finance
        # Should NOT return: local internal, any federated items
        assert len(filtered) == 2
        paths = [item["path"] for item in filtered]
        assert "/local-public" in paths
        assert "/local-finance" in paths
        assert "/local-internal" not in paths
        assert "/peer-a/public" not in paths
        assert "/peer-a/finance" not in paths

    def test_export_servers_excludes_federated(
        self,
        mock_federation_auth: Any,
    ) -> None:
        """Test /api/federation/servers endpoint excludes federated servers."""
        from registry.auth.dependencies import nginx_proxied_auth

        app.dependency_overrides[nginx_proxied_auth] = mock_federation_auth

        servers_dict = {
            "/local-server": {
                "path": "/local-server",
                "name": "Local Server",
                "visibility": "public",
            },
            "/peer-a/synced-server": {
                "path": "/peer-a/synced-server",
                "name": "Synced from Peer A",
                "visibility": "public",
                "sync_metadata": {
                    "is_federated": True,
                    "source_peer_id": "peer-a",
                },
            },
        }

        with (
            patch.object(
                server_service,
                "get_all_servers",
                return_value=servers_dict,
            ),
            patch.object(
                server_service,
                "is_service_enabled",
                return_value=True,
            ),
        ):
            client = TestClient(app)
            response = client.get("/api/federation/servers")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # Only local server should be exported
            assert len(data["items"]) == 1
            assert data["items"][0]["path"] == "/local-server"

        app.dependency_overrides.clear()

    def test_export_agents_excludes_federated(
        self,
        mock_federation_auth: Any,
    ) -> None:
        """Test /api/federation/agents endpoint excludes federated agents."""
        from registry.auth.dependencies import nginx_proxied_auth

        app.dependency_overrides[nginx_proxied_auth] = mock_federation_auth

        # Create mock agents
        local_agent = Mock()
        local_agent.path = "/agents/local-agent"
        local_agent.visibility = "public"
        local_agent.allowed_groups = []
        local_agent.sync_metadata = None
        local_agent.model_dump = Mock(
            return_value={
                "path": "/agents/local-agent",
                "visibility": "public",
            }
        )

        federated_agent = Mock()
        federated_agent.path = "/agents/peer-a/synced-agent"
        federated_agent.visibility = "public"
        federated_agent.allowed_groups = []
        federated_agent.sync_metadata = {
            "is_federated": True,
            "source_peer_id": "peer-a",
        }
        federated_agent.model_dump = Mock(
            return_value={
                "path": "/agents/peer-a/synced-agent",
                "visibility": "public",
                "sync_metadata": {
                    "is_federated": True,
                    "source_peer_id": "peer-a",
                },
            }
        )

        with (
            patch.object(
                agent_service,
                "get_all_agents",
                return_value=[local_agent, federated_agent],
            ),
            patch.object(
                agent_service,
                "get_all_agent_states",
                return_value={
                    "/agents/local-agent": True,
                    "/agents/peer-a/synced-agent": True,
                },
            ),
        ):
            client = TestClient(app)
            response = client.get("/api/federation/agents")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # Only local agent should be exported
            assert len(data["items"]) == 1
            assert data["items"][0]["path"] == "/agents/local-agent"

        app.dependency_overrides.clear()
