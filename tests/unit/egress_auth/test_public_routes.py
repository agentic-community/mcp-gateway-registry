"""Public egress-auth endpoint tests.

TestClient over the router with dependencies stubbed: nginx_proxied_auth (user
context), CSRF (no-op), server_service (async-mocked), and the EgressAuthService.
"""

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import registry.api.egress_auth_routes as routes

ADMIN = {"username": "admin", "is_admin": True, "auth_method": "oauth2", "client_id": "web"}
USER = {"username": "alice", "is_admin": False, "auth_method": "oauth2", "client_id": "web"}
STATIC = {"username": "ci-bot", "is_admin": False, "auth_method": "network-trusted"}


class _Conn:
    def __init__(self, provider, server_path):
        self.provider = provider
        self.server_path = server_path

    def model_dump(self):
        return {"provider": self.provider, "server_path": self.server_path, "status": "active"}


@pytest.fixture
def client(monkeypatch):
    def _build(user_context, *, server=None, svc=None, enabled=True):
        monkeypatch.setattr(routes.settings, "egress_auth_enabled", enabled)
        monkeypatch.setattr(routes.settings, "egress_oauth_callback_base_url", "https://gw.example")
        # server_service is a module-level singleton; patch its methods.
        monkeypatch.setattr(
            routes.server_service, "get_server_info", AsyncMock(return_value=server)
        )
        monkeypatch.setattr(routes.server_service, "update_server", AsyncMock(return_value=True))
        if svc is not None:
            monkeypatch.setattr(routes, "get_egress_auth_service", lambda: svc)

        app = FastAPI()
        app.include_router(routes.router, prefix="/api")
        app.dependency_overrides[routes.nginx_proxied_auth] = lambda: user_context
        app.dependency_overrides[routes.verify_csrf_token_flexible] = lambda: None
        return TestClient(app)

    return _build


def _server(**over):
    base = {
        "path": "/github-mcp",
        "proxy_pass_url": "https://api.githubcopilot.com/mcp",
        "egress_auth_mode": "oauth_user",
        "egress_oauth": {
            "provider": "github",
            "client_id": "Iv1.x",
            "client_secret_encrypted": "enc",
            "scopes": ["repo"],
        },
    }
    base.update(over)
    return base


@pytest.mark.unit
class TestConfigure:
    def test_configure_admin_ok_secret_stripped(self, client, monkeypatch):
        monkeypatch.setattr(routes, "encrypt_credential", lambda s: "ENC")
        c = client(ADMIN, server=_server(egress_oauth=None, egress_auth_mode="none"))
        r = c.post(
            "/api/servers/github-mcp/egress-auth",
            json={
                "egress_auth_mode": "oauth_user",
                "egress_provider": "github",
                "client_id": "Iv1.x",
                "client_secret": "ghs_secret",
                "scopes": ["repo"],
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["callback_url"] == "https://gw.example/oauth2/egress/callback"
        assert "client_secret" not in str(body)
        assert "client_secret_encrypted" not in str(body)

    def test_configure_non_admin_403(self, client):
        c = client(USER, server=_server())
        r = c.post(
            "/api/servers/github-mcp/egress-auth",
            json={
                "egress_auth_mode": "oauth_user",
                "egress_provider": "github",
                "client_id": "x",
                "client_secret": "y",
            },
        )
        assert r.status_code == 403

    def test_configure_unknown_provider_400(self, client):
        c = client(ADMIN, server=_server())
        r = c.post(
            "/api/servers/github-mcp/egress-auth",
            json={
                "egress_auth_mode": "oauth_user",
                "egress_provider": "bogus",
                "client_id": "x",
                "client_secret": "y",
            },
        )
        assert r.status_code == 400

    def test_configure_custom_missing_urls_400(self, client, monkeypatch):
        monkeypatch.setattr(routes, "encrypt_credential", lambda s: "ENC")
        c = client(ADMIN, server=_server())
        r = c.post(
            "/api/servers/github-mcp/egress-auth",
            json={
                "egress_auth_mode": "oauth_user",
                "egress_provider": "custom",
                "client_id": "x",
                "client_secret": "y",
            },
        )
        assert r.status_code == 400

    def test_feature_disabled_404(self, client):
        c = client(ADMIN, server=_server(), enabled=False)
        r = c.post("/api/servers/github-mcp/egress-auth", json={"egress_auth_mode": "none"})
        assert r.status_code == 404


@pytest.mark.unit
class TestReadConfig:
    def test_read_strips_secret(self, client):
        c = client(ADMIN, server=_server())
        r = c.get("/api/servers/github-mcp/egress-auth")
        assert r.status_code == 200
        assert "client_secret_encrypted" not in str(r.json())
        assert r.json()["egress_provider"] == "github"

    def test_read_non_admin_403(self, client):
        c = client(USER, server=_server())
        assert c.get("/api/servers/github-mcp/egress-auth").status_code == 403


@pytest.mark.unit
class TestInitiate:
    def test_initiate_returns_authorize_url(self, client):
        svc = AsyncMock()
        svc.build_consent_url = lambda **kw: "https://github.com/login/oauth/authorize?x=1"
        c = client(USER, server=_server(), svc=svc)
        r = c.post("/api/egress-auth/initiate", json={"server_path": "/github-mcp"})
        assert r.status_code == 200
        assert r.json()["authorize_url"].startswith("https://github.com/")

    def test_initiate_non_per_user_403(self, client):
        c = client(STATIC, server=_server())
        r = c.post("/api/egress-auth/initiate", json={"server_path": "/github-mcp"})
        assert r.status_code == 403

    def test_initiate_unconfigured_server_400(self, client):
        c = client(USER, server=_server(egress_auth_mode="none", egress_oauth=None))
        r = c.post("/api/egress-auth/initiate", json={"server_path": "/github-mcp"})
        assert r.status_code == 400


@pytest.mark.unit
class TestConnectionsAndDisconnect:
    def test_list_connections(self, client):
        svc = AsyncMock()
        svc.list_connections = AsyncMock(return_value=[_Conn("github", "/github-mcp")])
        c = client(USER, svc=svc)
        r = c.get("/api/egress-auth/connections")
        assert r.status_code == 200
        assert r.json()[0]["provider"] == "github"
        assert "access_token" not in str(r.json())

    def test_disconnect(self, client):
        svc = AsyncMock()
        svc.disconnect = AsyncMock(return_value=None)
        c = client(USER, svc=svc)
        r = c.request("DELETE", "/api/egress-auth/connections/github/github-mcp")
        assert r.status_code == 200
        assert r.json()["status"] == "revoked"


@pytest.mark.unit
class TestCallback:
    def test_callback_missing_params_400(self, client):
        c = client(USER, server=_server())
        assert c.get("/api/oauth2/egress/callback").status_code == 400

    def test_callback_bad_state_400(self, client):
        svc = AsyncMock()
        c = client(USER, server=_server(), svc=svc)
        # decode_state is called inside the route on the real codec -> garbage 400
        r = c.get("/api/oauth2/egress/callback", params={"code": "c", "state": "garbage"})
        assert r.status_code == 400

    def test_callback_happy_path_stores_and_succeeds(self, client, monkeypatch):
        # Mint a real signed+encrypted state so the route's decode_state succeeds,
        # then stub handle_callback to confirm the success page renders.
        monkeypatch.setenv("SECRET_KEY", "test-secret-key-for-testing-only-do-not-use")
        from registry.egress_auth import state_codec
        from registry.egress_auth.schemas import EgressConnection, OAuthState

        state_codec.reset_cipher_for_tests()
        blob = state_codec.encode_state(
            OAuthState(
                user_id="alice",
                auth_method="oauth2",
                provider="github",
                server_path="/github-mcp",
                nonce="n1",
                issued_at="2026-06-19T00:00:00+00:00",
            )
        )
        svc = AsyncMock()
        svc.handle_callback = AsyncMock(
            return_value=EgressConnection(provider="github", server_path="/github-mcp")
        )
        c = client(USER, server=_server(), svc=svc)
        r = c.get("/api/oauth2/egress/callback", params={"code": "the-code", "state": blob})
        assert r.status_code == 200
        assert "Connected github" in r.text
        assert svc.handle_callback.await_count == 1
        state_codec.reset_cipher_for_tests()

    def test_callback_store_write_failure_returns_503_not_success(self, client, monkeypatch):
        # The code exchange succeeds but persisting the token to the secret store
        # (Vault/OpenBao) fails even after the store's own transient retries. The
        # route MUST surface a retryable 503 "Connection not saved" page and MUST
        # NOT fall through to the success page -- otherwise the user is told
        # "Connected" while no token was vaulted, i.e. a silent "0 tools".
        monkeypatch.setenv("SECRET_KEY", "test-secret-key-for-testing-only-do-not-use")
        from registry.egress_auth import state_codec
        from registry.egress_auth.schemas import OAuthState
        from registry.secrets.interfaces import SecretStoreError

        state_codec.reset_cipher_for_tests()
        blob = state_codec.encode_state(
            OAuthState(
                user_id="alice",
                auth_method="oauth2",
                provider="github",
                server_path="/github-mcp",
                nonce="n1",
                issued_at="2026-06-19T00:00:00+00:00",
            )
        )
        svc = AsyncMock()
        svc.handle_callback = AsyncMock(side_effect=SecretStoreError("vault write failed"))
        c = client(USER, server=_server(), svc=svc)
        r = c.get("/api/oauth2/egress/callback", params={"code": "the-code", "state": blob})
        assert r.status_code == 503
        assert "Connection not saved" in r.text
        assert "Connected github" not in r.text
        assert svc.handle_callback.await_count == 1
        state_codec.reset_cipher_for_tests()

    def test_callback_success_page_escapes_server_path(self, client, monkeypatch):
        # server_path is admin-registrant-supplied; validate_server_path blocks
        # nginx metacharacters but not '<'/'>', so the success HTML must escape it
        # or a crafted path executes script in the connecting user's browser.
        monkeypatch.setenv("SECRET_KEY", "test-secret-key-for-testing-only-do-not-use")
        from registry.egress_auth import state_codec
        from registry.egress_auth.schemas import EgressConnection, OAuthState

        state_codec.reset_cipher_for_tests()
        evil_path = "/<svg/onload=alert(1)>"
        blob = state_codec.encode_state(
            OAuthState(
                user_id="alice",
                auth_method="oauth2",
                provider="github",
                server_path=evil_path,
                nonce="n1",
                issued_at="2026-06-19T00:00:00+00:00",
            )
        )
        svc = AsyncMock()
        svc.handle_callback = AsyncMock(
            return_value=EgressConnection(provider="<b>github</b>", server_path=evil_path)
        )
        c = client(USER, server=_server(), svc=svc)
        r = c.get("/api/oauth2/egress/callback", params={"code": "the-code", "state": blob})
        assert r.status_code == 200
        # The raw script-bearing markup must NOT appear; the escaped form must.
        assert "<svg/onload=alert(1)>" not in r.text
        assert "<b>github</b>" not in r.text
        assert "&lt;svg/onload=alert(1)&gt;" in r.text
        state_codec.reset_cipher_for_tests()
