"""End-to-end integration test for the MCP 2025-06-18 OAuth discovery chain.

This is the spec-faithful integration test mandated by issue #989's acceptance
criteria: it drives the full discovery sequence that a real MCP client (Claude
Code, Claude.ai connector, Cursor) would perform against the gateway, and
asserts each step lands on the right URL with the right shape.

The test uses a fake AuthProvider in place of a live IdP so the chain runs
in-process (no docker-compose, no network). The shape and URLs match what
Keycloak / Cognito / etc. would return; only the HTTP transport to the IdP
is short-circuited.

Sequence covered (per MCP 2025-06-18 §authorization-flow-steps):

    1. Client sends MCP request without token  -> 401 + WWW-Authenticate
    2. Client GETs the resource_metadata URL   -> RFC 9728 PRM document
    3. Client GETs the AS metadata URL         -> RFC 8414 AS metadata
    4. Client extracts authorization_endpoint  -> ready for PKCE flow

Steps 5+ (browser auth, code exchange, token use) belong to the IdP and are
out of scope for #989.
"""

import re
from urllib.parse import urlparse

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from registry.api.wellknown_routes import router as wellknown_router
from registry.middleware.mcp_www_authenticate import WWWAuthenticateMiddleware

pytestmark = [pytest.mark.integration]


# Canonical gateway URL for this test. Production deployments would supply
# this via REGISTRY_URL; we set it directly on the patched settings.
TEST_GATEWAY_URL: str = "https://mcpgateway.test"


# Fake IdP shape, matching what Keycloak/Auth0/Okta would return for RFC 8414.
FAKE_AS_METADATA: dict = {
    "issuer": "https://idp.test/realms/mcp-gateway",
    "authorization_endpoint": "https://idp.test/realms/mcp-gateway/protocol/openid-connect/auth",
    "token_endpoint": "https://idp.test/realms/mcp-gateway/protocol/openid-connect/token",
    "jwks_uri": "https://idp.test/realms/mcp-gateway/protocol/openid-connect/certs",
    "userinfo_endpoint": "https://idp.test/realms/mcp-gateway/protocol/openid-connect/userinfo",
    "end_session_endpoint": "https://idp.test/realms/mcp-gateway/protocol/openid-connect/logout",
    "registration_endpoint": "https://idp.test/realms/mcp-gateway/clients-registrations/openid-connect",
    "response_types_supported": ["code"],
    "grant_types_supported": ["authorization_code", "refresh_token"],
    "code_challenge_methods_supported": ["S256"],
}


class FakeProvider:
    """Minimal stand-in for an AuthProvider during the discovery chain."""

    def authorization_server_metadata(self) -> dict:
        return dict(FAKE_AS_METADATA)

    def authorization_server_issuer(self) -> str:
        return FAKE_AS_METADATA["issuer"]

    def protected_resource_metadata(
        self,
        resource: str,
        scopes_supported: list,
        resource_documentation: str | None = None,
    ) -> dict:
        from auth_server.providers.base import AuthProvider

        return AuthProvider.protected_resource_metadata(
            self, resource, scopes_supported, resource_documentation
        )


@pytest.fixture
def discovery_app(monkeypatch):
    """Build a FastAPI app with PRM + AS-metadata routes + WWW-Authenticate middleware,
    plus a stub /<server>/mcp endpoint that returns 401 (simulating an unauthenticated
    request to a protected MCP server)."""
    from registry.api import wellknown_routes as wkr
    from registry.auth import oauth_metadata as om

    # Patch settings used by both the routes and the helpers
    class _StubSettings:
        registry_url = TEST_GATEWAY_URL
        mcp_https_required = True
        mcp_resource_documentation_url = None

    stub_settings = _StubSettings()
    monkeypatch.setattr(wkr, "settings", stub_settings)
    monkeypatch.setattr(om, "settings", stub_settings)

    # derive_supported_scopes() returns the basic default OIDC scopes when
    # mcp_advertised_scopes is unset; it no longer reads the scopes DB, so no
    # scopes-loader patching is needed here.

    # Patch the provider factory used by the route
    fake_provider = FakeProvider()
    monkeypatch.setattr(wkr, "_get_active_auth_provider", lambda: fake_provider)

    app = FastAPI()
    # Middleware first so it wraps the protected route too
    app.add_middleware(
        WWWAuthenticateMiddleware,
        resource_metadata_url=f"{TEST_GATEWAY_URL}/.well-known/oauth-protected-resource",
    )
    app.include_router(wellknown_router, prefix="/.well-known")

    @app.get("/airegistry-tools/mcp")
    async def airegistry_tools_mcp_protected():
        # Real gateway has nginx auth_request enforce this; in-process we
        # simulate the 401 the client would see.
        raise HTTPException(status_code=401, detail="auth required")

    return app


class TestDiscoveryChainStep1Unauthenticated401:
    """Step 1: Client sends MCP request without token; gets 401 + WWW-Authenticate."""

    def test_401_includes_www_authenticate(self, discovery_app):
        client = TestClient(discovery_app)
        response = client.get("/airegistry-tools/mcp")

        assert response.status_code == 401
        assert "www-authenticate" in {k.lower() for k in response.headers.keys()}

    def test_www_authenticate_points_to_prm_url(self, discovery_app):
        client = TestClient(discovery_app)
        response = client.get("/airegistry-tools/mcp")

        header = response.headers["www-authenticate"]
        match = re.search(r'resource_metadata="([^"]+)"', header)
        assert match is not None, f"Could not extract resource_metadata from {header!r}"

        resource_metadata_url = match.group(1)
        assert resource_metadata_url == f"{TEST_GATEWAY_URL}/.well-known/oauth-protected-resource"


class TestDiscoveryChainStep2FetchPRM:
    """Step 2: Client follows resource_metadata URL to get the RFC 9728 PRM doc."""

    def test_prm_returns_rfc9728_required_fields(self, discovery_app):
        client = TestClient(discovery_app)
        response = client.get("/.well-known/oauth-protected-resource")

        assert response.status_code == 200
        prm = response.json()

        # All RFC 9728-required fields present
        assert prm["resource"] == TEST_GATEWAY_URL
        assert prm["authorization_servers"] == [FAKE_AS_METADATA["issuer"]]
        assert prm["bearer_methods_supported"] == ["header"]
        assert "scopes_supported" in prm
        assert "resource_documentation" in prm

    def test_prm_resource_field_matches_401_resource_metadata_url(self, discovery_app):
        """Acceptance criterion: byte-for-byte match between PRM `resource` field
        and the WWW-Authenticate `resource_metadata` URL."""
        client = TestClient(discovery_app)

        # Get the URL the WWW-Authenticate header points at
        unauth_response = client.get("/airegistry-tools/mcp")
        header = unauth_response.headers["www-authenticate"]
        resource_metadata_url = re.search(r'resource_metadata="([^"]+)"', header).group(1)

        # Get the PRM doc and check its resource field
        prm_response = client.get("/.well-known/oauth-protected-resource")
        prm = prm_response.json()

        # Must equal exactly: <resource_field>/.well-known/oauth-protected-resource
        assert resource_metadata_url == f"{prm['resource']}/.well-known/oauth-protected-resource"


class TestDiscoveryChainStep3FetchASMetadata:
    """Step 3: Client follows authorization_servers entry to get RFC 8414 AS metadata."""

    def test_as_metadata_returns_rfc8414_required_fields(self, discovery_app):
        client = TestClient(discovery_app)
        response = client.get("/.well-known/oauth-authorization-server")

        assert response.status_code == 200
        metadata = response.json()

        # RFC 8414 required fields
        assert metadata["issuer"] == FAKE_AS_METADATA["issuer"]
        assert metadata["authorization_endpoint"]
        assert metadata["token_endpoint"]
        assert metadata["jwks_uri"]
        # PKCE per OAuth 2.1
        assert "S256" in metadata["code_challenge_methods_supported"]


class TestDiscoveryChainStep4ClientExtractsAuthorizeURL:
    """Step 4: Client extracts authorization_endpoint from AS metadata; ready for PKCE."""

    def test_full_chain_lands_on_authorize_endpoint(self, discovery_app):
        """Drive the entire discovery sequence end-to-end and confirm the client
        knows where to send the user for browser-based auth."""
        client = TestClient(discovery_app)

        # Step 1: get the WWW-Authenticate
        unauth = client.get("/airegistry-tools/mcp")
        assert unauth.status_code == 401
        prm_url = re.search(
            r'resource_metadata="([^"]+)"', unauth.headers["www-authenticate"]
        ).group(1)

        # Step 2: GET the PRM doc
        prm_response = client.get(urlparse(prm_url).path)
        assert prm_response.status_code == 200
        prm = prm_response.json()

        # Step 3: GET the AS metadata
        # In production a client follows authorization_servers[0] directly to
        # the IdP. The gateway also exposes a normalized passthrough at
        # /.well-known/oauth-authorization-server. Test the passthrough since
        # it's the one we own.
        as_response = client.get("/.well-known/oauth-authorization-server")
        assert as_response.status_code == 200
        as_metadata = as_response.json()

        # Step 4: client now has authorization_endpoint to redirect the user to
        authorize_url = as_metadata["authorization_endpoint"]
        parsed = urlparse(authorize_url)
        assert parsed.scheme == "https"
        assert parsed.netloc
        assert parsed.path.endswith("/auth") or parsed.path.endswith("/authorize")

        # And the AS issuer matches the one PRM advertised, closing the loop.
        assert as_metadata["issuer"] in prm["authorization_servers"]
