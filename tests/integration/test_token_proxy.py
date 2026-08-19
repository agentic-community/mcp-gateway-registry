"""AS-metadata token_endpoint repoint to the gateway proxy.

Extends the discovery-chain style (in-process fake IdP, no docker/network):
when MCP_TOKEN_PROXY_ENABLED is set, the gateway's RFC 8414 AS-metadata document
advertises its OWN /oauth/token as the token_endpoint, while keeping the IdP's
authorization_endpoint, issuer, and jwks_uri so clients still validate IdP-signed
tokens against the IdP JWKS. When the flag is off, the document is an unchanged
passthrough of the IdP's endpoints.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from registry.api.wellknown_routes import router as wellknown_router

pytestmark = [pytest.mark.integration]

TEST_GATEWAY_URL = "https://mcpgateway.test"

FAKE_AS_METADATA = {
    "issuer": "https://idp.test/realms/mcp-gateway",
    "authorization_endpoint": "https://idp.test/realms/mcp-gateway/protocol/openid-connect/auth",
    "token_endpoint": "https://idp.test/realms/mcp-gateway/protocol/openid-connect/token",
    "jwks_uri": "https://idp.test/realms/mcp-gateway/protocol/openid-connect/certs",
    "response_types_supported": ["code"],
    "grant_types_supported": ["authorization_code", "refresh_token"],
    "code_challenge_methods_supported": ["S256"],
}


class FakeProvider:
    def authorization_server_metadata(self) -> dict:
        return dict(FAKE_AS_METADATA)


def _app(monkeypatch, *, proxy_enabled: bool) -> FastAPI:
    from registry.api import wellknown_routes as wkr
    from registry.auth import oauth_metadata as om

    class _StubSettings:
        registry_url = TEST_GATEWAY_URL
        mcp_https_required = True
        mcp_resource_documentation_url = None
        mcp_token_proxy_enabled = proxy_enabled

    stub = _StubSettings()
    monkeypatch.setattr(wkr, "settings", stub)
    monkeypatch.setattr(om, "settings", stub)
    monkeypatch.setattr(wkr, "_get_active_auth_provider", lambda: FakeProvider())

    app = FastAPI()
    app.include_router(wellknown_router, prefix="/.well-known")
    return app


class TestAsMetadataRepoint:
    def test_token_endpoint_is_idp_when_proxy_disabled(self, monkeypatch):
        client = TestClient(_app(monkeypatch, proxy_enabled=False))
        doc = client.get("/.well-known/oauth-authorization-server").json()
        assert doc["token_endpoint"] == FAKE_AS_METADATA["token_endpoint"]

    def test_token_endpoint_repoints_to_gateway_when_proxy_enabled(self, monkeypatch):
        client = TestClient(_app(monkeypatch, proxy_enabled=True))
        doc = client.get("/.well-known/oauth-authorization-server").json()

        # token_endpoint now the gateway's own proxy...
        assert doc["token_endpoint"] == f"{TEST_GATEWAY_URL}/oauth/token"
        # ...while authorization_endpoint / issuer / jwks_uri stay the IdP's, so
        # clients still run the auth-code flow at the IdP and validate against
        # the IdP JWKS.
        assert doc["authorization_endpoint"] == FAKE_AS_METADATA["authorization_endpoint"]
        assert doc["issuer"] == FAKE_AS_METADATA["issuer"]
        assert doc["jwks_uri"] == FAKE_AS_METADATA["jwks_uri"]
