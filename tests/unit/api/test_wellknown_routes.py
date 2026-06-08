"""
Unit tests for registry/api/wellknown_routes.py

Tests the well-known URL discovery endpoint including:
- GET /.well-known/mcp-servers - MCP server discovery
- Health status retrieval from health service
- Status normalization for client consumption
"""

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

logger = logging.getLogger(__name__)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_server_service():
    """Mock server_service dependency."""
    mock_service = MagicMock()
    mock_service.get_all_servers = AsyncMock(return_value={})
    mock_service.is_service_enabled = AsyncMock(return_value=True)
    return mock_service


@pytest.fixture
def mock_health_service():
    """Mock health_service dependency with server_health_status dict."""
    mock_service = MagicMock()
    mock_service.server_health_status = {}
    return mock_service


@pytest.fixture
def sample_server_info() -> dict[str, Any]:
    """Create sample server information for testing."""
    return {
        "path": "test-server",
        "server_name": "Test Server",
        "description": "A test MCP server",
        "transport": "streamable-http",
        "auth_type": "oauth",
        "auth_provider": "keycloak",
        "tool_list": [
            {"name": "get_data", "description": "Get data from source"},
            {"name": "process_data", "description": "Process data"},
        ],
        "proxy_pass_url": "http://localhost:8000",
        "is_enabled": True,
    }


# =============================================================================
# UNIT TESTS FOR _get_normalized_health_status
# =============================================================================


class TestGetNormalizedHealthStatus:
    """Tests for the _get_normalized_health_status helper function."""

    def test_healthy_status_normalized(self, mock_health_service, mock_settings):
        """Test that 'healthy' status is returned as 'healthy'."""
        mock_health_service.server_health_status = {"test-server": "healthy"}

        with patch("registry.api.wellknown_routes.health_service", mock_health_service):
            from registry.api.wellknown_routes import _get_normalized_health_status

            result = _get_normalized_health_status("test-server")
            assert result == "healthy"

    def test_healthy_auth_expired_normalized_to_healthy(self, mock_health_service, mock_settings):
        """Test that 'healthy-auth-expired' is normalized to 'healthy'."""
        mock_health_service.server_health_status = {"test-server": "healthy-auth-expired"}

        with patch("registry.api.wellknown_routes.health_service", mock_health_service):
            from registry.api.wellknown_routes import _get_normalized_health_status

            result = _get_normalized_health_status("test-server")
            assert result == "healthy"

    def test_unhealthy_timeout_normalized(self, mock_health_service, mock_settings):
        """Test that 'unhealthy: timeout' is normalized to 'unhealthy'."""
        mock_health_service.server_health_status = {"test-server": "unhealthy: timeout"}

        with patch("registry.api.wellknown_routes.health_service", mock_health_service):
            from registry.api.wellknown_routes import _get_normalized_health_status

            result = _get_normalized_health_status("test-server")
            assert result == "unhealthy"

    def test_unhealthy_connection_error_normalized(self, mock_health_service, mock_settings):
        """Test that 'unhealthy: connection error' is normalized to 'unhealthy'."""
        mock_health_service.server_health_status = {"test-server": "unhealthy: connection error"}

        with patch("registry.api.wellknown_routes.health_service", mock_health_service):
            from registry.api.wellknown_routes import _get_normalized_health_status

            result = _get_normalized_health_status("test-server")
            assert result == "unhealthy"

    def test_error_status_normalized_to_unhealthy(self, mock_health_service, mock_settings):
        """Test that error statuses are normalized to 'unhealthy'."""
        mock_health_service.server_health_status = {"test-server": "error: ConnectionError"}

        with patch("registry.api.wellknown_routes.health_service", mock_health_service):
            from registry.api.wellknown_routes import _get_normalized_health_status

            result = _get_normalized_health_status("test-server")
            assert result == "unhealthy"

    def test_disabled_status_normalized(self, mock_health_service, mock_settings):
        """Test that 'disabled' status is returned as 'disabled'."""
        mock_health_service.server_health_status = {"test-server": "disabled"}

        with patch("registry.api.wellknown_routes.health_service", mock_health_service):
            from registry.api.wellknown_routes import _get_normalized_health_status

            result = _get_normalized_health_status("test-server")
            assert result == "disabled"

    def test_checking_status_normalized_to_unknown(self, mock_health_service, mock_settings):
        """Test that 'checking' status is normalized to 'unknown'."""
        mock_health_service.server_health_status = {"test-server": "checking"}

        with patch("registry.api.wellknown_routes.health_service", mock_health_service):
            from registry.api.wellknown_routes import _get_normalized_health_status

            result = _get_normalized_health_status("test-server")
            assert result == "unknown"

    def test_unknown_server_returns_unknown(self, mock_health_service, mock_settings):
        """Test that unknown servers return 'unknown' status."""
        mock_health_service.server_health_status = {}

        with patch("registry.api.wellknown_routes.health_service", mock_health_service):
            from registry.api.wellknown_routes import _get_normalized_health_status

            result = _get_normalized_health_status("nonexistent-server")
            assert result == "unknown"


# =============================================================================
# UNIT TESTS FOR _format_server_discovery
# =============================================================================


class TestFormatServerDiscovery:
    """Tests for the _format_server_discovery function."""

    def test_format_includes_health_status(
        self, mock_health_service, mock_settings, sample_server_info
    ):
        """Test that formatted server includes actual health status."""
        mock_health_service.server_health_status = {"test-server": "healthy"}

        with patch("registry.api.wellknown_routes.health_service", mock_health_service):
            from registry.api.wellknown_routes import _format_server_discovery

            # Create a mock request
            mock_request = MagicMock()
            mock_request.headers = {"host": "localhost:7860"}
            mock_request.url.scheme = "http"

            result = _format_server_discovery(sample_server_info, mock_request)

            assert result["health_status"] == "healthy"
            assert result["name"] == "Test Server"
            assert result["description"] == "A test MCP server"

    def test_format_uses_unhealthy_status_from_health_service(
        self, mock_health_service, mock_settings, sample_server_info
    ):
        """Test that formatted server uses unhealthy status from health service."""
        mock_health_service.server_health_status = {"test-server": "unhealthy: timeout"}

        with patch("registry.api.wellknown_routes.health_service", mock_health_service):
            from registry.api.wellknown_routes import _format_server_discovery

            mock_request = MagicMock()
            mock_request.headers = {"host": "localhost:7860"}
            mock_request.url.scheme = "http"

            result = _format_server_discovery(sample_server_info, mock_request)

            # Should be normalized to 'unhealthy'
            assert result["health_status"] == "unhealthy"

    def test_format_unknown_server_has_unknown_status(self, mock_health_service, mock_settings):
        """Test that servers not in health service have 'unknown' status."""
        mock_health_service.server_health_status = {}

        server_info = {
            "path": "new-server",
            "server_name": "New Server",
            "description": "A new server",
        }

        with patch("registry.api.wellknown_routes.health_service", mock_health_service):
            from registry.api.wellknown_routes import _format_server_discovery

            mock_request = MagicMock()
            mock_request.headers = {"host": "localhost:7860"}
            mock_request.url.scheme = "http"

            result = _format_server_discovery(server_info, mock_request)

            assert result["health_status"] == "unknown"


# =============================================================================
# INTEGRATION TESTS FOR GET /.well-known/mcp-servers
# =============================================================================


class TestWellKnownMcpServersEndpoint:
    """Integration tests for the well-known MCP servers endpoint."""

    def test_endpoint_returns_actual_health_status(
        self,
        mock_server_service,
        mock_health_service,
        mock_settings,
        sample_server_info,
    ):
        """Test that the endpoint returns actual health status, not hardcoded."""
        # Set up mock data
        mock_server_service.get_all_servers = AsyncMock(
            return_value={"test-server": sample_server_info}
        )
        mock_server_service.is_service_enabled = AsyncMock(return_value=True)
        mock_health_service.server_health_status = {"test-server": "unhealthy: connection error"}

        # Patch settings to enable discovery
        mock_settings.enable_wellknown_discovery = True
        mock_settings.wellknown_cache_ttl = 300

        with (
            patch("registry.api.wellknown_routes.server_service", mock_server_service),
            patch("registry.api.wellknown_routes.health_service", mock_health_service),
            patch("registry.api.wellknown_routes.settings", mock_settings),
        ):
            from fastapi import FastAPI

            from registry.api.wellknown_routes import router

            app = FastAPI()
            app.include_router(router, prefix="/.well-known")

            client = TestClient(app)
            response = client.get("/.well-known/mcp-servers")

            assert response.status_code == 200
            data = response.json()
            assert len(data["servers"]) == 1
            # Verify health_status is normalized from "unhealthy: connection error" to "unhealthy"
            assert data["servers"][0]["health_status"] == "unhealthy"

    def test_endpoint_returns_healthy_status(
        self,
        mock_server_service,
        mock_health_service,
        mock_settings,
        sample_server_info,
    ):
        """Test that healthy servers show as healthy."""
        mock_server_service.get_all_servers = AsyncMock(
            return_value={"test-server": sample_server_info}
        )
        mock_server_service.is_service_enabled = AsyncMock(return_value=True)
        mock_health_service.server_health_status = {"test-server": "healthy"}

        mock_settings.enable_wellknown_discovery = True
        mock_settings.wellknown_cache_ttl = 300

        with (
            patch("registry.api.wellknown_routes.server_service", mock_server_service),
            patch("registry.api.wellknown_routes.health_service", mock_health_service),
            patch("registry.api.wellknown_routes.settings", mock_settings),
        ):
            from fastapi import FastAPI

            from registry.api.wellknown_routes import router

            app = FastAPI()
            app.include_router(router, prefix="/.well-known")

            client = TestClient(app)
            response = client.get("/.well-known/mcp-servers")

            assert response.status_code == 200
            data = response.json()
            assert data["servers"][0]["health_status"] == "healthy"

    def test_endpoint_returns_unknown_for_unchecked_servers(
        self,
        mock_server_service,
        mock_health_service,
        mock_settings,
        sample_server_info,
    ):
        """Test that servers not yet health-checked show as unknown."""
        mock_server_service.get_all_servers = AsyncMock(
            return_value={"test-server": sample_server_info}
        )
        mock_server_service.is_service_enabled = AsyncMock(return_value=True)
        # Empty health status dict means no health checks have run yet
        mock_health_service.server_health_status = {}

        mock_settings.enable_wellknown_discovery = True
        mock_settings.wellknown_cache_ttl = 300

        with (
            patch("registry.api.wellknown_routes.server_service", mock_server_service),
            patch("registry.api.wellknown_routes.health_service", mock_health_service),
            patch("registry.api.wellknown_routes.settings", mock_settings),
        ):
            from fastapi import FastAPI

            from registry.api.wellknown_routes import router

            app = FastAPI()
            app.include_router(router, prefix="/.well-known")

            client = TestClient(app)
            response = client.get("/.well-known/mcp-servers")

            assert response.status_code == 200
            data = response.json()
            assert data["servers"][0]["health_status"] == "unknown"

    def test_multiple_servers_with_different_health_statuses(
        self,
        mock_server_service,
        mock_health_service,
        mock_settings,
    ):
        """Test that multiple servers show their individual health statuses."""
        servers = {
            "healthy-server": {
                "path": "healthy-server",
                "server_name": "Healthy Server",
                "description": "A healthy server",
                "is_enabled": True,
            },
            "unhealthy-server": {
                "path": "unhealthy-server",
                "server_name": "Unhealthy Server",
                "description": "An unhealthy server",
                "is_enabled": True,
            },
            "unknown-server": {
                "path": "unknown-server",
                "server_name": "Unknown Server",
                "description": "A server with unknown status",
                "is_enabled": True,
            },
        }

        mock_server_service.get_all_servers = AsyncMock(return_value=servers)
        mock_server_service.is_service_enabled = AsyncMock(return_value=True)
        mock_health_service.server_health_status = {
            "healthy-server": "healthy",
            "unhealthy-server": "unhealthy: timeout",
            # unknown-server not in dict, should return "unknown"
        }

        mock_settings.enable_wellknown_discovery = True
        mock_settings.wellknown_cache_ttl = 300

        with (
            patch("registry.api.wellknown_routes.server_service", mock_server_service),
            patch("registry.api.wellknown_routes.health_service", mock_health_service),
            patch("registry.api.wellknown_routes.settings", mock_settings),
        ):
            from fastapi import FastAPI

            from registry.api.wellknown_routes import router

            app = FastAPI()
            app.include_router(router, prefix="/.well-known")

            client = TestClient(app)
            response = client.get("/.well-known/mcp-servers")

            assert response.status_code == 200
            data = response.json()
            assert len(data["servers"]) == 3

            # Create a dict for easier verification
            server_statuses = {s["name"]: s["health_status"] for s in data["servers"]}

            assert server_statuses["Healthy Server"] == "healthy"
            assert server_statuses["Unhealthy Server"] == "unhealthy"
            assert server_statuses["Unknown Server"] == "unknown"


# =============================================================================
# OAUTH DISCOVERY ENDPOINTS (RFC 9728 + RFC 8414)
# =============================================================================


def _make_oauth_discovery_app(mock_provider, mock_settings_obj=None):
    """Build a FastAPI app with the wellknown router and patched dependencies."""
    from fastapi import FastAPI

    from registry.api.wellknown_routes import router

    app = FastAPI()
    app.include_router(router, prefix="/.well-known")
    return app


@pytest.fixture
def fake_as_metadata():
    """A representative RFC 8414 document for tests."""
    return {
        "issuer": "https://idp.example.com/realms/test",
        "authorization_endpoint": "https://idp.example.com/realms/test/protocol/openid-connect/auth",
        "token_endpoint": "https://idp.example.com/realms/test/protocol/openid-connect/token",
        "jwks_uri": "https://idp.example.com/realms/test/protocol/openid-connect/certs",
        "response_types_supported": ["code"],
        "code_challenge_methods_supported": ["S256"],
    }


@pytest.fixture
def fake_provider(fake_as_metadata):
    """A MagicMock auth provider that returns the canned AS metadata."""
    from auth_server.providers.base import AuthProvider

    provider = MagicMock(spec=AuthProvider)
    provider.authorization_server_metadata.return_value = fake_as_metadata
    provider.authorization_server_issuer.return_value = fake_as_metadata["issuer"]
    # Use the real default protected_resource_metadata implementation by binding it
    provider.protected_resource_metadata.side_effect = (
        lambda resource,
        scopes_supported,
        resource_documentation=None: AuthProvider.protected_resource_metadata(
            provider, resource, scopes_supported, resource_documentation
        )
    )
    return provider


class TestOAuthProtectedResourceEndpoint:
    """Tests for GET /.well-known/oauth-protected-resource (RFC 9728)."""

    def test_returns_required_rfc9728_fields(self, mock_settings, fake_provider):
        """PRM document includes resource, authorization_servers, scopes_supported, bearer_methods_supported."""
        mock_settings.registry_url = "https://gw.example.com"
        mock_settings.mcp_https_required = True
        mock_settings.mcp_resource_documentation_url = None
        mock_settings.mcp_advertised_scopes = ""

        with (
            patch(
                "registry.api.wellknown_routes._get_active_auth_provider",
                return_value=fake_provider,
            ),
            patch(
                "registry.auth.oauth_metadata.reload_scopes_config",
                AsyncMock(
                    return_value={
                        "group_mappings": {"admins": ["mcp-admin"]},
                        "mcp-admin": [],
                    }
                ),
            ),
            patch("registry.auth.oauth_metadata.settings", mock_settings),
            patch("registry.api.wellknown_routes.settings", mock_settings),
        ):
            app = _make_oauth_discovery_app(fake_provider)
            client = TestClient(app)
            response = client.get("/.well-known/oauth-protected-resource")

            assert response.status_code == 200
            data = response.json()
            assert data["resource"] == "https://gw.example.com"
            assert data["authorization_servers"] == ["https://idp.example.com/realms/test"]
            assert data["scopes_supported"] == ["mcp-admin"]
            assert data["bearer_methods_supported"] == ["header"]
            assert data["resource_documentation"] == "https://gw.example.com/docs/oauth"

    def test_strips_trailing_slash_from_registry_url(self, mock_settings, fake_provider):
        """A trailing slash on registry_url must not survive into the `resource` field."""
        mock_settings.registry_url = "https://gw.example.com/"
        mock_settings.mcp_https_required = True
        mock_settings.mcp_resource_documentation_url = None

        with (
            patch(
                "registry.api.wellknown_routes._get_active_auth_provider",
                return_value=fake_provider,
            ),
            patch(
                "registry.auth.oauth_metadata.reload_scopes_config",
                AsyncMock(return_value={"group_mappings": {}}),
            ),
            patch("registry.auth.oauth_metadata.settings", mock_settings),
            patch("registry.api.wellknown_routes.settings", mock_settings),
        ):
            app = _make_oauth_discovery_app(fake_provider)
            client = TestClient(app)
            response = client.get("/.well-known/oauth-protected-resource")

            assert response.status_code == 200
            assert response.json()["resource"] == "https://gw.example.com"

    def test_https_enforcement_in_production(self, mock_settings, fake_provider):
        """An http registry_url with mcp_https_required=true must surface a 5xx."""
        mock_settings.registry_url = "http://gw.example.com"
        mock_settings.mcp_https_required = True
        mock_settings.mcp_resource_documentation_url = None

        with (
            patch(
                "registry.api.wellknown_routes._get_active_auth_provider",
                return_value=fake_provider,
            ),
            patch("registry.api.wellknown_routes.settings", mock_settings),
        ):
            app = _make_oauth_discovery_app(fake_provider)
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/.well-known/oauth-protected-resource")

            assert response.status_code == 500

    def test_local_dev_allows_http(self, mock_settings, fake_provider):
        """With mcp_https_required=false, http registry_url is permitted."""
        mock_settings.registry_url = "http://localhost:7860"
        mock_settings.mcp_https_required = False
        mock_settings.mcp_resource_documentation_url = None

        with (
            patch(
                "registry.api.wellknown_routes._get_active_auth_provider",
                return_value=fake_provider,
            ),
            patch(
                "registry.auth.oauth_metadata.reload_scopes_config",
                AsyncMock(return_value={"group_mappings": {}}),
            ),
            patch("registry.auth.oauth_metadata.settings", mock_settings),
            patch("registry.api.wellknown_routes.settings", mock_settings),
        ):
            app = _make_oauth_discovery_app(fake_provider)
            client = TestClient(app)
            response = client.get("/.well-known/oauth-protected-resource")

            assert response.status_code == 200
            assert response.json()["resource"] == "http://localhost:7860"

    def test_cache_control_header(self, mock_settings, fake_provider):
        """RFC 9728 docs are cacheable for 5 minutes."""
        mock_settings.registry_url = "https://gw.example.com"
        mock_settings.mcp_https_required = True
        mock_settings.mcp_resource_documentation_url = None

        with (
            patch(
                "registry.api.wellknown_routes._get_active_auth_provider",
                return_value=fake_provider,
            ),
            patch(
                "registry.auth.oauth_metadata.reload_scopes_config",
                AsyncMock(return_value={"group_mappings": {}}),
            ),
            patch("registry.auth.oauth_metadata.settings", mock_settings),
            patch("registry.api.wellknown_routes.settings", mock_settings),
        ):
            app = _make_oauth_discovery_app(fake_provider)
            client = TestClient(app)
            response = client.get("/.well-known/oauth-protected-resource")

            assert response.headers["cache-control"] == "public, max-age=300"

    def test_provider_not_implemented_returns_501(self, mock_settings):
        """A provider whose authorization_server_metadata() raises NotImplementedError surfaces as 501."""
        from auth_server.providers.base import AuthProvider

        mock_settings.registry_url = "https://gw.example.com"
        mock_settings.mcp_https_required = True
        mock_settings.mcp_resource_documentation_url = None

        provider = MagicMock(spec=AuthProvider)
        provider.authorization_server_metadata.side_effect = NotImplementedError("stub")
        provider.authorization_server_issuer.side_effect = NotImplementedError("stub")
        provider.protected_resource_metadata.side_effect = NotImplementedError("stub")

        with (
            patch(
                "registry.api.wellknown_routes._get_active_auth_provider",
                return_value=provider,
            ),
            patch(
                "registry.auth.oauth_metadata.reload_scopes_config",
                AsyncMock(return_value={"group_mappings": {}}),
            ),
            patch("registry.auth.oauth_metadata.settings", mock_settings),
            patch("registry.api.wellknown_routes.settings", mock_settings),
        ):
            app = _make_oauth_discovery_app(provider)
            client = TestClient(app)
            response = client.get("/.well-known/oauth-protected-resource")

            assert response.status_code == 501

    def test_upstream_idp_failure_returns_502(self, mock_settings):
        """If the provider can't fetch upstream metadata, surface as 502."""
        from auth_server.providers.base import AuthProvider

        mock_settings.registry_url = "https://gw.example.com"
        mock_settings.mcp_https_required = True
        mock_settings.mcp_resource_documentation_url = None

        provider = MagicMock(spec=AuthProvider)
        provider.protected_resource_metadata.side_effect = ValueError(
            "OpenID configuration retrieval failed"
        )

        with (
            patch(
                "registry.api.wellknown_routes._get_active_auth_provider",
                return_value=provider,
            ),
            patch(
                "registry.auth.oauth_metadata.reload_scopes_config",
                AsyncMock(return_value={"group_mappings": {}}),
            ),
            patch("registry.auth.oauth_metadata.settings", mock_settings),
            patch("registry.api.wellknown_routes.settings", mock_settings),
        ):
            app = _make_oauth_discovery_app(provider)
            client = TestClient(app)
            response = client.get("/.well-known/oauth-protected-resource")

            assert response.status_code == 502


class TestOAuthAuthorizationServerEndpoint:
    """Tests for GET /.well-known/oauth-authorization-server (RFC 8414)."""

    def test_returns_provider_metadata(self, mock_settings, fake_provider, fake_as_metadata):
        """The route returns whatever the provider's authorization_server_metadata() returns."""
        with patch(
            "registry.api.wellknown_routes._get_active_auth_provider",
            return_value=fake_provider,
        ):
            app = _make_oauth_discovery_app(fake_provider)
            client = TestClient(app)
            response = client.get("/.well-known/oauth-authorization-server")

            assert response.status_code == 200
            assert response.json() == fake_as_metadata

    def test_cache_control_header(self, mock_settings, fake_provider):
        """RFC 8414 docs are cacheable for 5 minutes."""
        with patch(
            "registry.api.wellknown_routes._get_active_auth_provider",
            return_value=fake_provider,
        ):
            app = _make_oauth_discovery_app(fake_provider)
            client = TestClient(app)
            response = client.get("/.well-known/oauth-authorization-server")

            assert response.headers["cache-control"] == "public, max-age=300"

    def test_provider_not_implemented_returns_501(self, mock_settings):
        """A stub provider returns 501 cleanly rather than 500."""
        from auth_server.providers.base import AuthProvider

        provider = MagicMock(spec=AuthProvider)
        provider.authorization_server_metadata.side_effect = NotImplementedError("stub")

        with patch(
            "registry.api.wellknown_routes._get_active_auth_provider",
            return_value=provider,
        ):
            app = _make_oauth_discovery_app(provider)
            client = TestClient(app)
            response = client.get("/.well-known/oauth-authorization-server")

            assert response.status_code == 501

    def test_upstream_failure_returns_502(self, mock_settings):
        """Network failures fetching IdP metadata surface as 502."""
        from auth_server.providers.base import AuthProvider

        provider = MagicMock(spec=AuthProvider)
        provider.authorization_server_metadata.side_effect = ValueError(
            "OpenID configuration retrieval failed"
        )

        with patch(
            "registry.api.wellknown_routes._get_active_auth_provider",
            return_value=provider,
        ):
            app = _make_oauth_discovery_app(provider)
            client = TestClient(app)
            response = client.get("/.well-known/oauth-authorization-server")

            assert response.status_code == 502


class TestPrmAndResourceMetadataMatchByteForByte:
    """Acceptance criterion: PRM `resource` field MUST equal the WWW-Authenticate
    `resource_metadata` URL byte-for-byte. This is the cross-cutting test that
    pins the contract."""

    def test_resource_field_equals_resource_metadata_url(self, mock_settings, fake_provider):
        from registry.auth.oauth_metadata import (
            build_canonical_resource_url,
            build_resource_metadata_url,
        )

        mock_settings.registry_url = "https://gw.example.com/"
        mock_settings.mcp_https_required = True
        mock_settings.mcp_resource_documentation_url = None

        with (
            patch(
                "registry.api.wellknown_routes._get_active_auth_provider",
                return_value=fake_provider,
            ),
            patch(
                "registry.auth.oauth_metadata.reload_scopes_config",
                AsyncMock(return_value={"group_mappings": {}}),
            ),
            patch("registry.auth.oauth_metadata.settings", mock_settings),
            patch("registry.api.wellknown_routes.settings", mock_settings),
        ):
            app = _make_oauth_discovery_app(fake_provider)
            client = TestClient(app)
            response = client.get("/.well-known/oauth-protected-resource")

            data = response.json()
            resource_field = data["resource"]

            # The URL the WWW-Authenticate middleware will embed in 401s
            expected_resource_metadata = build_resource_metadata_url(
                build_canonical_resource_url(mock_settings.registry_url)
            )

            # Must equal {resource}/.well-known/oauth-protected-resource exactly
            assert (
                expected_resource_metadata
                == f"{resource_field}/.well-known/oauth-protected-resource"
            )
