"""
Unit tests for registry/api/wellknown_routes.py

Tests the well-known discovery endpoints:
- GET /.well-known/oauth-protected-resource (RFC 9728)
- GET /.well-known/oauth-authorization-server (RFC 8414)
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

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

    def test_cache_control_header_is_no_store(self, mock_settings, fake_provider):
        """PRM metadata must not be cached by shared/CDN caches (poisoning risk)."""
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

            cache_control = response.headers["cache-control"]
            assert cache_control == "no-store"
            assert "public" not in cache_control
            assert "max-age" not in cache_control

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


class TestGlobalPrmEntraScopeFormat:
    """The global PRM must advertise scopes in the form the configured Entra
    version expects (issue #990). Uses a REAL EntraIdProvider so the route +
    provider.protected_resource_metadata() + format_advertised_scopes() are
    exercised end-to-end. Regression guard for AADSTS650053: v1 clients must
    receive api://<app-id>/<scope> so they send the correct /authorize scope.
    """

    def _entra_provider(self, monkeypatch, scope_format, application_id_uri=None):
        for var in (
            "ENTRA_LOGIN_BASE_URL",
            "ENTRA_GRAPH_BASE_URL",
            "ENTRA_SCOPE_FORMAT",
            "ENTRA_APPLICATION_ID_URI",
        ):
            monkeypatch.delenv(var, raising=False)
        from auth_server.providers.entra import EntraIdProvider

        return EntraIdProvider(
            tenant_id="tenant-abc",
            client_id="app-guid",
            client_secret="s",
            scope_format=scope_format,
            application_id_uri=application_id_uri,
        )

    def _run(self, provider, mock_settings, advertised_scopes):
        mock_settings.registry_url = "https://gw.example.com"
        mock_settings.mcp_https_required = True
        mock_settings.mcp_resource_documentation_url = None
        mock_settings.mcp_advertised_scopes = advertised_scopes
        with (
            patch(
                "registry.api.wellknown_routes._get_active_auth_provider",
                return_value=provider,
            ),
            patch("registry.auth.oauth_metadata.settings", mock_settings),
            patch("registry.api.wellknown_routes.settings", mock_settings),
        ):
            client = TestClient(_make_oauth_discovery_app(provider))
            return client.get("/.well-known/oauth-protected-resource")

    def test_v1_emits_api_prefixed_custom_scope(self, mock_settings, monkeypatch):
        provider = self._entra_provider(monkeypatch, scope_format="v1")
        resp = self._run(provider, mock_settings, "openid email mcp.read")
        assert resp.status_code == 200
        scopes = resp.json()["scopes_supported"]
        # OIDC scopes bare, custom scope api://-prefixed (AADSTS650053 fix).
        assert scopes == ["openid", "email", "api://app-guid/mcp.read"]

    def test_v1_uses_application_id_uri(self, mock_settings, monkeypatch):
        provider = self._entra_provider(
            monkeypatch, scope_format="v1", application_id_uri="api://gw-uri"
        )
        resp = self._run(provider, mock_settings, "mcp.read")
        assert resp.json()["scopes_supported"] == ["api://gw-uri/mcp.read"]

    def test_v2_emits_bare_custom_scope(self, mock_settings, monkeypatch):
        provider = self._entra_provider(monkeypatch, scope_format="v2")
        resp = self._run(provider, mock_settings, "openid mcp.read")
        assert resp.json()["scopes_supported"] == ["openid", "mcp.read"]


class TestGlobalPrmNonEntraUnchanged:
    """Non-Entra providers must emit scopes unchanged (no api:// rewrite).

    Keycloak/Okta/Cognito/Auth0 use the base protected_resource_metadata(),
    which preserves the caller-supplied scopes verbatim.
    """

    def test_non_entra_provider_emits_scopes_verbatim(self, mock_settings, fake_provider):
        # fake_provider is a MagicMock bound to the BASE implementation.
        mock_settings.registry_url = "https://gw.example.com"
        mock_settings.mcp_https_required = True
        mock_settings.mcp_resource_documentation_url = None
        mock_settings.mcp_advertised_scopes = "openid mcp.read mcp.write"
        with (
            patch(
                "registry.api.wellknown_routes._get_active_auth_provider",
                return_value=fake_provider,
            ),
            patch("registry.auth.oauth_metadata.settings", mock_settings),
            patch("registry.api.wellknown_routes.settings", mock_settings),
        ):
            client = TestClient(_make_oauth_discovery_app(fake_provider))
            resp = client.get("/.well-known/oauth-protected-resource")
        # No transformation: exactly what the operator configured.
        assert resp.json()["scopes_supported"] == ["openid", "mcp.read", "mcp.write"]


class TestServerNeedsPerServerPrm:
    """server_needs_per_server_prm() gate (issue #990).

    On Entra, EVERY server -- including plain ``none`` servers -- needs a
    per-server PRM, because the gateway-wide bare-origin resource is unmatchable
    to an Entra App ID URI (trailing slash) and fails token exchange with
    AADSTS9010010. On lenient IdPs, only the two egress modes need it; plain
    servers keep the gateway-wide origin PRM (unchanged behavior). This gate is
    the fix that unblocks Entra IDE login for plain servers.
    """

    def _gate(self, egress_auth_mode, auth_provider):
        from registry.api.wellknown_routes import server_needs_per_server_prm

        mock_settings = MagicMock()
        mock_settings.auth_provider = auth_provider
        with patch("registry.api.wellknown_routes.settings", mock_settings):
            return server_needs_per_server_prm(egress_auth_mode)

    # --- Entra: everything gets a per-server PRM ---
    def test_entra_plain_server_gets_per_server_prm(self):
        """The #990 fix: a plain (none) server on Entra needs the per-server PRM."""
        assert self._gate("none", "entra") is True

    def test_entra_unset_egress_mode_gets_per_server_prm(self):
        assert self._gate(None, "entra") is True

    def test_entra_obo_exchange_gets_per_server_prm(self):
        assert self._gate("obo_exchange", "entra") is True

    def test_entra_oauth_user_gets_per_server_prm(self):
        assert self._gate("oauth_user", "entra") is True

    # --- Non-Entra: only the egress modes; plain stays on the global PRM ---
    def test_keycloak_plain_server_uses_global_prm(self):
        """REGRESSION GUARD: a plain server on Keycloak must NOT get a per-server
        PRM -- the bare-origin global PRM works there and this is the path that
        has worked all along. Changing it would break lenient-IdP deployments."""
        assert self._gate("none", "keycloak") is False

    def test_keycloak_oauth_user_uses_global_prm(self):
        """Keycloak 3LO keeps using the gateway-wide root PRM (unchanged)."""
        assert self._gate("oauth_user", "keycloak") is False

    def test_cognito_plain_server_uses_global_prm(self):
        assert self._gate("none", "cognito") is False

    def test_okta_plain_server_uses_global_prm(self):
        assert self._gate(None, "okta") is False

    def test_obo_exchange_gets_per_server_prm_on_any_provider(self):
        """obo_exchange always needs the per-server PRM regardless of IdP."""
        assert self._gate("obo_exchange", "keycloak") is True

    def test_auth_provider_case_insensitive(self):
        assert self._gate("none", "Entra") is True


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

    def test_cache_control_header_is_no_store(self, mock_settings, fake_provider):
        """AS metadata must not be cached by shared/CDN caches (poisoning risk)."""
        with patch(
            "registry.api.wellknown_routes._get_active_auth_provider",
            return_value=fake_provider,
        ):
            app = _make_oauth_discovery_app(fake_provider)
            client = TestClient(app)
            response = client.get("/.well-known/oauth-authorization-server")

            cache_control = response.headers["cache-control"]
            assert cache_control == "no-store"
            assert "public" not in cache_control
            assert "max-age" not in cache_control

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


class TestPerServerOAuthProtectedResource:
    """Tests for GET /.well-known/oauth-protected-resource/{server_path} (per-server PRM).

    Serves a document ONLY for obo_exchange servers; advertises the SHARED
    gateway resource (one Entra App ID URI for all obo servers). Non-obo or
    unknown servers 404 so clients fall back to the global PRM.
    """

    def _settings(self, auth_provider="entra"):
        s = MagicMock()
        s.registry_url = "https://gw.example.com"
        s.mcp_https_required = True
        s.mcp_resource_documentation_url = None
        s.mcp_advertised_scopes = ""
        s.auth_provider = auth_provider
        return s

    def test_obo_server_returns_per_server_connection_url_resource(self, fake_provider):
        s = self._settings()
        obo_server = {"path": "/obo-echo", "egress_auth_mode": "obo_exchange"}
        with (
            patch(
                "registry.api.wellknown_routes._get_active_auth_provider",
                return_value=fake_provider,
            ),
            patch("registry.auth.oauth_metadata.settings", s),
            patch("registry.api.wellknown_routes.settings", s),
            patch(
                "registry.api.wellknown_routes.server_service.get_server_info",
                new=AsyncMock(return_value=obo_server),
            ),
        ):
            client = TestClient(_make_oauth_discovery_app(fake_provider))
            # path-aware discovery form the client uses (with /mcp suffix)
            resp = client.get("/.well-known/oauth-protected-resource/obo-echo/mcp")
            assert resp.status_code == 200
            data = resp.json()
            # Per-server connection URL (the only Entra-matchable + client-accepted
            # value): https://gw/<server>/mcp, no trailing slash.
            assert data["resource"] == "https://gw.example.com/obo-echo/mcp"
            assert data["scopes_supported"] == [
                "https://gw.example.com/obo-echo/mcp/user_impersonation"
            ]

    def test_append_mcp_false_server_omits_mcp_suffix(self, fake_provider):
        s = self._settings()
        obo_server = {
            "path": "/aws-knowledge",
            "egress_auth_mode": "obo_exchange",
            "append_mcp_path": False,
        }
        with (
            patch(
                "registry.api.wellknown_routes._get_active_auth_provider",
                return_value=fake_provider,
            ),
            patch("registry.auth.oauth_metadata.settings", s),
            patch("registry.api.wellknown_routes.settings", s),
            patch(
                "registry.api.wellknown_routes.server_service.get_server_info",
                new=AsyncMock(return_value=obo_server),
            ),
        ):
            client = TestClient(_make_oauth_discovery_app(fake_provider))
            resp = client.get("/.well-known/oauth-protected-resource/aws-knowledge")
            assert resp.status_code == 200
            assert resp.json()["resource"] == "https://gw.example.com/aws-knowledge"

    def test_oauth_user_server_returns_per_server_resource_on_entra(self, fake_provider):
        # 3LO (oauth_user) on ENTRA gets a per-server PRM: its ingress leg hits the
        # same strict resource/scope alignment constraint as obo_exchange.
        s = self._settings(auth_provider="entra")
        oauth_user_server = {"path": "/github", "egress_auth_mode": "oauth_user"}
        with (
            patch(
                "registry.api.wellknown_routes._get_active_auth_provider",
                return_value=fake_provider,
            ),
            patch("registry.auth.oauth_metadata.settings", s),
            patch("registry.api.wellknown_routes.settings", s),
            patch(
                "registry.api.wellknown_routes.server_service.get_server_info",
                new=AsyncMock(return_value=oauth_user_server),
            ),
        ):
            client = TestClient(_make_oauth_discovery_app(fake_provider))
            resp = client.get("/.well-known/oauth-protected-resource/github/mcp")
            assert resp.status_code == 200
            data = resp.json()
            assert data["resource"] == "https://gw.example.com/github/mcp"
            assert data["scopes_supported"] == [
                "https://gw.example.com/github/mcp/user_impersonation"
            ]

    def test_oauth_user_server_404s_on_keycloak(self, fake_provider):
        # REGRESSION GUARD: Keycloak 3LO works today via the gateway-wide root PRM.
        # oauth_user must NOT get a per-server PRM on Keycloak (only on Entra), or
        # we'd change that working path. 404 here -> client falls back to the global
        # PRM, exactly as on main.
        s = self._settings(auth_provider="keycloak")
        oauth_user_server = {"path": "/github", "egress_auth_mode": "oauth_user"}
        with (
            patch(
                "registry.api.wellknown_routes._get_active_auth_provider",
                return_value=fake_provider,
            ),
            patch("registry.auth.oauth_metadata.settings", s),
            patch("registry.api.wellknown_routes.settings", s),
            patch(
                "registry.api.wellknown_routes.server_service.get_server_info",
                new=AsyncMock(return_value=oauth_user_server),
            ),
        ):
            client = TestClient(_make_oauth_discovery_app(fake_provider))
            resp = client.get("/.well-known/oauth-protected-resource/github/mcp")
            assert resp.status_code == 404

    def test_non_egress_server_404s_on_lenient_idp(self, fake_provider):
        # On a lenient IdP (Keycloak), a plain server falls back to the global
        # PRM -- the bare-origin resource works there. 404 here is correct.
        s = self._settings(auth_provider="keycloak")
        plain_server = {"path": "/plain", "egress_auth_mode": "none"}
        with (
            patch(
                "registry.api.wellknown_routes._get_active_auth_provider",
                return_value=fake_provider,
            ),
            patch("registry.auth.oauth_metadata.settings", s),
            patch("registry.api.wellknown_routes.settings", s),
            patch(
                "registry.api.wellknown_routes.server_service.get_server_info",
                new=AsyncMock(return_value=plain_server),
            ),
        ):
            client = TestClient(_make_oauth_discovery_app(fake_provider))
            resp = client.get("/.well-known/oauth-protected-resource/plain/mcp")
            assert resp.status_code == 404

    def test_non_egress_server_gets_per_server_prm_on_entra(self, fake_provider):
        # issue #990: on Entra, a plain (none) server MUST get a per-server PRM
        # (resource = its connection URL), because the bare-origin global PRM is
        # unmatchable to an Entra App ID URI. This is the fix that unblocks Entra
        # IDE login for plain servers.
        s = self._settings(auth_provider="entra")
        plain_server = {"path": "/plain", "egress_auth_mode": "none"}
        with (
            patch(
                "registry.api.wellknown_routes._get_active_auth_provider",
                return_value=fake_provider,
            ),
            patch("registry.auth.oauth_metadata.settings", s),
            patch("registry.api.wellknown_routes.settings", s),
            patch(
                "registry.api.wellknown_routes.server_service.get_server_info",
                new=AsyncMock(return_value=plain_server),
            ),
        ):
            client = TestClient(_make_oauth_discovery_app(fake_provider))
            resp = client.get("/.well-known/oauth-protected-resource/plain/mcp")
            assert resp.status_code == 200
            data = resp.json()
            assert data["resource"] == "https://gw.example.com/plain/mcp"
            assert data["scopes_supported"] == [
                "https://gw.example.com/plain/mcp/user_impersonation"
            ]

    def test_unknown_server_404s(self, fake_provider):
        s = self._settings()
        with (
            patch(
                "registry.api.wellknown_routes._get_active_auth_provider",
                return_value=fake_provider,
            ),
            patch("registry.auth.oauth_metadata.settings", s),
            patch("registry.api.wellknown_routes.settings", s),
            patch(
                "registry.api.wellknown_routes.server_service.get_server_info",
                new=AsyncMock(return_value=None),
            ),
        ):
            client = TestClient(_make_oauth_discovery_app(fake_provider))
            resp = client.get("/.well-known/oauth-protected-resource/nope/mcp")
            assert resp.status_code == 404

    def test_server_lookup_failure_returns_502_not_500(self, fake_provider):
        """A repository/backend error during the server lookup on this
        UNAUTHENTICATED endpoint must fail closed to a generic 502 (logged), not
        surface as an unhandled 500 with a traceback."""
        s = self._settings()
        with (
            patch(
                "registry.api.wellknown_routes._get_active_auth_provider",
                return_value=fake_provider,
            ),
            patch("registry.auth.oauth_metadata.settings", s),
            patch("registry.api.wellknown_routes.settings", s),
            patch(
                "registry.api.wellknown_routes.server_service.get_server_info",
                new=AsyncMock(side_effect=RuntimeError("documentdb down")),
            ),
        ):
            client = TestClient(
                _make_oauth_discovery_app(fake_provider), raise_server_exceptions=False
            )
            resp = client.get("/.well-known/oauth-protected-resource/obo-echo/mcp")
            assert resp.status_code == 502
            # No internal exception detail leaked on the public endpoint.
            assert "documentdb" not in resp.text.lower()

    def test_path_normalization_strips_mcp_suffix(self, fake_provider):
        """The handler must look up '/obo-echo', not '/obo-echo/mcp'."""
        s = self._settings()
        seen = {}

        async def _capture(path, *a, **k):
            seen["path"] = path
            return {"path": "/obo-echo", "egress_auth_mode": "obo_exchange"}

        with (
            patch(
                "registry.api.wellknown_routes._get_active_auth_provider",
                return_value=fake_provider,
            ),
            patch("registry.auth.oauth_metadata.settings", s),
            patch("registry.api.wellknown_routes.settings", s),
            patch("registry.api.wellknown_routes.server_service.get_server_info", new=_capture),
        ):
            client = TestClient(_make_oauth_discovery_app(fake_provider))
            client.get("/.well-known/oauth-protected-resource/obo-echo/mcp")
            assert seen["path"] == "/obo-echo"
