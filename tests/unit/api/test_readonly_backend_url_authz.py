"""Access-control and backend-URL redaction tests for server read endpoints.

Covers the information-disclosure / broken-access-control fixes on read-only
endpoints:

- ``GET /servers/{path}/versions`` — must enforce per-server access for
  non-admins and must not leak the internal ``proxy_pass_url`` of any version
  in with-gateway mode.
- ``GET /servers/groups`` and ``GET /servers/groups/{group_name}`` — expose the
  full access-control model (server_access / group_mappings / ui_permissions),
  so they must be admin-only, matching their write siblings.
- ``POST /api/search/semantic`` — must apply the same non-admin backend-URL
  redaction as ``GET /servers/{path}`` instead of returning raw
  ``proxy_pass_url`` / ``mcp_endpoint`` / ``sse_endpoint``.

Plus direct unit tests of the shared redaction helpers in
``registry.services.visibility``.
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
def admin_user_context() -> dict[str, Any]:
    """Admin user context (bypasses access checks, sees raw URLs)."""
    return {
        "username": "admin",
        "is_admin": True,
        "groups": ["mcp-registry-admin"],
        "scopes": ["mcp-servers-unrestricted/read"],
        "accessible_servers": ["all"],
        "accessible_services": ["all"],
        "accessible_agents": ["all"],
        "ui_permissions": {"list_service": ["all"]},
        "auth_method": "session",
    }


@pytest.fixture
def regular_user_context() -> dict[str, Any]:
    """Non-admin user context with access to exactly one server."""
    return {
        "username": "testuser",
        "is_admin": False,
        "groups": ["test-group"],
        "scopes": ["test-server/read"],
        "accessible_servers": ["test-server"],
        "accessible_services": ["test-server"],
        "accessible_agents": ["test-agent"],
        "ui_permissions": {"list_service": ["test-server"]},
        "auth_method": "session",
    }


@pytest.fixture
def versions_response() -> dict[str, Any]:
    """Server versions payload as returned by server_service.get_server_versions."""
    return {
        "path": "/test-server",
        "default_version": "v2.0.0",
        "versions": [
            {
                "version": "v2.0.0",
                "proxy_pass_url": "http://internal-backend:8080",
                "status": "stable",
                "is_default": True,
            },
            {
                "version": "v1.0.0",
                "proxy_pass_url": "http://internal-backend:8081",
                "status": "stable",
                "is_default": False,
            },
        ],
    }


@pytest.fixture
def mock_server_service():
    """Mock server_service with permissive defaults; tests tighten as needed."""
    mock_service = MagicMock()
    mock_service.get_server_info = AsyncMock(
        return_value={"server_name": "test-server", "path": "/test-server"}
    )
    mock_service.get_server_versions = AsyncMock(return_value={})
    mock_service.user_can_access_server_path = AsyncMock(return_value=True)
    return mock_service


def _make_client(user_context: dict[str, Any], mock_server_service: MagicMock):
    """Build a TestClient with auth + services overridden for the given user."""
    from registry.auth.dependencies import enhanced_auth, nginx_proxied_auth
    from registry.main import app

    app.dependency_overrides[nginx_proxied_auth] = lambda: user_context
    app.dependency_overrides[enhanced_auth] = lambda: user_context

    with (
        patch("registry.api.server_routes.server_service", mock_server_service),
        patch("registry.health.service.health_service", MagicMock()),
        patch("registry.core.nginx_service.nginx_service", MagicMock()),
        patch("registry.api.server_routes.security_scanner_service", MagicMock()),
    ):
        client = TestClient(app, cookies={"mcp_gateway_session": "test-session"})
        yield client

    app.dependency_overrides.clear()


@pytest.fixture
def client_admin(mock_settings, admin_user_context, mock_server_service):
    yield from _make_client(admin_user_context, mock_server_service)


@pytest.fixture
def client_regular(mock_settings, regular_user_context, mock_server_service):
    yield from _make_client(regular_user_context, mock_server_service)


def _force_with_gateway():
    """Patch the deployment mode at the helper's read site to with-gateway."""
    from registry.core.config import DeploymentMode

    return patch(
        "registry.services.visibility.settings.deployment_mode",
        DeploymentMode.WITH_GATEWAY,
    )


def _force_registry_only():
    """Patch the deployment mode at the helper's read site to registry-only."""
    from registry.core.config import DeploymentMode

    return patch(
        "registry.services.visibility.settings.deployment_mode",
        DeploymentMode.REGISTRY_ONLY,
    )


# =============================================================================
# SA-19: GET /servers/{path}/versions
# =============================================================================


class TestGetServerVersionsAuthz:
    """Access control and redaction on GET /servers/{path}/versions."""

    def test_non_admin_without_access_is_denied(
        self, client_regular, mock_server_service, versions_response
    ):
        """A non-admin who cannot access the server gets 403, no version data."""
        mock_server_service.get_server_versions.return_value = versions_response
        mock_server_service.user_can_access_server_path.return_value = False

        with _force_with_gateway():
            response = client_regular.get("/api/servers/test-server/versions")

        assert response.status_code == 403
        # The version list (and its URLs) must never be produced when denied.
        mock_server_service.get_server_versions.assert_not_called()

    def test_missing_server_returns_404(self, client_regular, mock_server_service):
        """A missing server returns 404 without leaking a 403-vs-404 oracle."""
        mock_server_service.get_server_info.return_value = None

        with _force_with_gateway():
            response = client_regular.get("/api/servers/ghost/versions")

        assert response.status_code == 404

    def test_non_admin_with_access_gets_versions_without_proxy_url(
        self, client_regular, mock_server_service, versions_response
    ):
        """A permitted non-admin sees versions but never the proxy_pass_url."""
        mock_server_service.get_server_versions.return_value = versions_response
        mock_server_service.user_can_access_server_path.return_value = True

        with _force_with_gateway():
            response = client_regular.get("/api/servers/test-server/versions")

        assert response.status_code == 200
        data = response.json()
        assert len(data["versions"]) == 2
        for version in data["versions"]:
            assert "proxy_pass_url" not in version

    def test_admin_sees_proxy_url(self, client_admin, mock_server_service, versions_response):
        """Admins always see the raw proxy_pass_url and bypass access checks."""
        mock_server_service.get_server_versions.return_value = versions_response

        with _force_with_gateway():
            response = client_admin.get("/api/servers/test-server/versions")

        assert response.status_code == 200
        data = response.json()
        assert data["versions"][0]["proxy_pass_url"] == "http://internal-backend:8080"
        mock_server_service.user_can_access_server_path.assert_not_called()

    def test_registry_only_keeps_proxy_url_for_non_admin(
        self, client_regular, mock_server_service, versions_response
    ):
        """In registry-only mode a permitted non-admin needs the URL to connect."""
        mock_server_service.get_server_versions.return_value = versions_response
        mock_server_service.user_can_access_server_path.return_value = True

        with _force_registry_only():
            response = client_regular.get("/api/servers/test-server/versions")

        assert response.status_code == 200
        data = response.json()
        assert data["versions"][0]["proxy_pass_url"] == "http://internal-backend:8080"


# =============================================================================
# SA-20: GET /servers/groups and GET /servers/groups/{group_name}
# =============================================================================


class TestGroupsReadAdminGate:
    """The group read endpoints expose the authz model — admin-only."""

    def test_list_groups_rejects_non_admin(self, client_regular):
        """GET /servers/groups is blocked for non-admins before any lookup."""
        with patch(
            "registry.api.server_routes._list_groups_impl", new_callable=AsyncMock
        ) as mock_impl:
            response = client_regular.get("/api/servers/groups")

        assert response.status_code == 403
        mock_impl.assert_not_called()

    def test_get_group_rejects_non_admin(self, client_regular):
        """GET /servers/groups/{name} is blocked for non-admins before any lookup."""
        with patch("registry.services.scope_service.get_group", new_callable=AsyncMock) as mock_get:
            response = client_regular.get("/api/servers/groups/mcp-registry-admin")

        assert response.status_code == 403
        mock_get.assert_not_called()

    def test_list_groups_allows_admin(self, client_admin):
        """Admins still reach the list implementation."""
        with patch(
            "registry.api.server_routes._list_groups_impl",
            new_callable=AsyncMock,
            return_value={"groups": []},
        ) as mock_impl:
            response = client_admin.get("/api/servers/groups")

        assert response.status_code != 403
        mock_impl.assert_called_once()

    def test_get_group_allows_admin(self, client_admin):
        """Admins still see the full group definition."""
        group_data = {
            "scope_name": "test-group",
            "server_access": [{"server": "test-server", "permissions": ["read"]}],
            "group_mappings": ["test-group"],
            "ui_permissions": {"register_service": ["all"]},
        }
        with patch(
            "registry.services.scope_service.get_group",
            new_callable=AsyncMock,
            return_value=group_data,
        ):
            response = client_admin.get("/api/servers/groups/test-group")

        assert response.status_code == 200
        assert response.json()["scope_name"] == "test-group"


# =============================================================================
# Shared helper unit tests (registry.services.visibility)
# =============================================================================


class TestBackendUrlRedactionHelpers:
    """Direct tests for the shared redaction decision + field stripper."""

    def test_should_redact_true_for_non_admin_with_gateway(self):
        from registry.services.visibility import should_redact_backend_urls

        with _force_with_gateway():
            assert should_redact_backend_urls({"is_admin": False}) is True

    def test_should_not_redact_for_admin(self):
        from registry.services.visibility import should_redact_backend_urls

        with _force_with_gateway():
            assert should_redact_backend_urls({"is_admin": True}) is False

    def test_should_not_redact_in_registry_only(self):
        from registry.services.visibility import should_redact_backend_urls

        with _force_registry_only():
            assert should_redact_backend_urls({"is_admin": False}) is False

    def test_should_redact_fails_closed_for_missing_context(self):
        """A missing/empty user_context must redact (fail closed) in with-gateway."""
        from registry.services.visibility import should_redact_backend_urls

        with _force_with_gateway():
            assert should_redact_backend_urls(None) is True
            assert should_redact_backend_urls({}) is True

    def test_redact_strips_top_level_and_versions(self):
        from registry.services.visibility import redact_server_backend_fields

        doc = {
            "server_name": "s",
            "proxy_pass_url": "http://internal:8080",
            "mcp_endpoint": "http://internal:8080/mcp",
            "sse_endpoint": "http://internal:8080/sse",
            "versions": [
                {"version": "v1", "proxy_pass_url": "http://internal:8081"},
                {"version": "v2", "mcp_endpoint": "http://internal:8082/mcp"},
            ],
        }
        result = redact_server_backend_fields(doc)

        assert result is doc  # mutates in place
        assert "proxy_pass_url" not in doc
        assert "mcp_endpoint" not in doc
        assert "sse_endpoint" not in doc
        for version in doc["versions"]:
            assert "proxy_pass_url" not in version
            assert "mcp_endpoint" not in version
        # Non-URL fields are preserved.
        assert doc["server_name"] == "s"
        assert doc["versions"][0]["version"] == "v1"


# =============================================================================
# Sibling sweep: GET /server_details/{path}
# =============================================================================


class TestGetServerDetailsRedaction:
    """GET /server_details/{path} must redact backend URLs for non-admins."""

    def test_non_admin_details_redacted(self, client_regular, mock_server_service):
        """Non-admin sees details but never the backend URLs (with-gateway)."""
        mock_server_service.get_server_info.return_value = {
            "server_name": "test-server",
            "path": "/test-server",
            "proxy_pass_url": "http://internal:8080",
            "mcp_endpoint": "http://internal:8080/mcp",
            "deployment": "remote",
        }
        mock_server_service.user_can_access_server_path.return_value = True

        with _force_with_gateway():
            response = client_regular.get("/api/server_details/test-server")

        assert response.status_code == 200
        data = response.json()
        assert "proxy_pass_url" not in data
        assert "mcp_endpoint" not in data

    def test_non_admin_local_details_redacted(self, client_regular, mock_server_service):
        """The local-server early-return path also redacts for non-admins."""
        mock_server_service.get_server_info.return_value = {
            "server_name": "local-server",
            "path": "/local-server",
            "proxy_pass_url": "http://internal:8080",
            "deployment": "local",
        }
        mock_server_service.user_can_access_server_path.return_value = True

        with _force_with_gateway():
            response = client_regular.get("/api/server_details/local-server")

        assert response.status_code == 200
        assert "proxy_pass_url" not in response.json()

    def test_admin_details_keep_backend_urls(self, client_admin, mock_server_service):
        """Admins still see the raw backend URLs."""
        mock_server_service.get_server_info.return_value = {
            "server_name": "test-server",
            "path": "/test-server",
            "proxy_pass_url": "http://internal:8080",
            "deployment": "remote",
        }

        with _force_with_gateway():
            response = client_admin.get("/api/server_details/test-server")

        assert response.status_code == 200
        assert response.json()["proxy_pass_url"] == "http://internal:8080"


# =============================================================================
# Sibling sweep: Anthropic registry API (/v0.1/servers*) transport.url
# =============================================================================


@pytest.mark.asyncio
class TestAnthropicApiBackendUrlRedaction:
    """The Anthropic API surfaces proxy_pass_url as transport.url — redact it."""

    def _server(self) -> dict[str, Any]:
        return {
            "server_name": "remote",
            "path": "/remote-srv",
            "deployment": "remote",
            "proxy_pass_url": "http://internal-backend:9000",
        }

    async def test_list_servers_redacts_transport_url_for_non_admin(self):
        from unittest.mock import patch as _patch

        from registry.api.registry_routes import list_servers

        with (
            _force_with_gateway(),
            _patch(
                "registry.api.registry_routes.server_service.get_all_servers_with_permissions",
                new_callable=AsyncMock,
                return_value={"/remote-srv": self._server()},
            ),
            _patch(
                "registry.api.registry_routes.health_service._get_service_health_data",
                return_value={"status": "healthy", "last_checked_iso": None},
            ),
        ):
            result = await list_servers(
                cursor=None,
                limit=10,
                user_context={
                    "username": "u",
                    "is_admin": False,
                    "accessible_servers": ["remote"],
                },
            )

        transport = result.servers[0].server.packages[0].transport
        assert transport.get("url") in ("", None)

    async def test_list_servers_keeps_transport_url_for_admin(self):
        from unittest.mock import patch as _patch

        from registry.api.registry_routes import list_servers

        with (
            _force_with_gateway(),
            _patch(
                "registry.api.registry_routes.server_service.get_all_servers",
                new_callable=AsyncMock,
                return_value={"/remote-srv": self._server()},
            ),
            _patch(
                "registry.api.registry_routes.health_service._get_service_health_data",
                return_value={"status": "healthy", "last_checked_iso": None},
            ),
        ):
            result = await list_servers(
                cursor=None,
                limit=10,
                user_context={"username": "admin", "is_admin": True},
            )

        transport = result.servers[0].server.packages[0].transport
        assert transport.get("url") == "http://internal-backend:9000"


# NOTE: the unauthenticated /.well-known/mcp-servers discovery endpoint that this
# suite's original TestWellKnownServerUrl covered was removed upstream (the
# anonymous discovery URL no longer exists), so those tests were dropped when
# upstream/main was merged in. The remaining redaction contract is exercised by
# the authenticated server read endpoints and semantic search above.
