"""
Unit tests for registry/api/wellknown_routes.py

Tests the well-known discovery endpoints:
- GET /.well-known/oauth-protected-resource (RFC 9728)
- GET /.well-known/oauth-authorization-server (RFC 8414)
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

logger = logging.getLogger(__name__)


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
        lambda resource, scopes_supported, resource_documentation=None: (
            AuthProvider.protected_resource_metadata(
                provider, resource, scopes_supported, resource_documentation
            )
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
            # With no mcp_advertised_scopes override, the PRM advertises the basic
            # IdP-universal OIDC scopes (NOT registry group names like 'mcp-admin').
            assert data["scopes_supported"] == ["openid", "email", "profile", "offline_access"]
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
