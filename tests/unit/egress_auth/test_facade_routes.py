"""Route-level tests for the egress consent route (egress_oauth_facade_routes.py).

TestClient over the router with the session read, server_service, and the
EgressAuthService stubbed. Verifies the param-free /oauth2/egress/connect front
door (session-gated, including the no-session -> Keycloak login bounce), the
session-cookie forwarding regression, and the resource-param parsing.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import registry.api.egress_oauth_facade_routes as facade

REGISTRY_URL = "https://gw.example.com"
USER = {
    "username": "alice",
    "auth_method": "oauth2",
    "client_id": "web",
    "groups": ["mcp-registry-user"],
    "scopes": ["openid", "email"],
}
STATIC = {"username": "ci-bot", "auth_method": "network-trusted", "groups": [], "scopes": []}


def _server(**over) -> dict:
    base = {
        "path": "/github",
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


@pytest.fixture
def client(monkeypatch):
    def _build(
        user_context=USER,
        *,
        server=None,
        consent_url="https://github.com/login/oauth/authorize?x=1",
        enabled=True,
        registry_url=REGISTRY_URL,
    ):
        monkeypatch.setattr(facade.settings, "egress_auth_enabled", enabled)
        monkeypatch.setattr(facade.settings, "registry_url", registry_url)

        async def _get_server_info(path, include_credentials=False):
            return server

        monkeypatch.setattr(facade.server_service, "get_server_info", _get_server_info)

        # /connect reads the session via _optional_session -> nginx_proxied_auth
        # (called with session=<cookie>). None user_context simulates no gateway
        # session (login bounce).
        async def _session(request, session=None):
            if user_context is None:
                raise RuntimeError("no session")
            return user_context

        monkeypatch.setattr(facade, "nginx_proxied_auth", _session)

        class _Svc:
            def build_consent_url(self, **kwargs):
                _Svc.last_session_id = kwargs.get("session_id")
                return consent_url

        monkeypatch.setattr(facade, "get_egress_auth_service", lambda: _Svc())

        app = FastAPI()
        app.include_router(facade.router)
        c = TestClient(app, follow_redirects=False)
        c._svc = _Svc
        return c

    return _build


@pytest.mark.unit
class TestOptionalSession:
    """Regression: _optional_session MUST read the session cookie off the request
    and pass it explicitly to nginx_proxied_auth. nginx_proxied_auth's `session`
    is a FastAPI Cookie(...) param only populated by dependency injection; calling
    it directly without passing the cookie always sees session=None -> the login
    loop (the live bug)."""

    async def test_passes_request_cookie_to_nginx_proxied_auth(self, monkeypatch):
        monkeypatch.setattr(facade.settings, "session_cookie_name", "mcp_gateway_session")
        seen = {}

        async def _fake(request, session=None):
            seen["session"] = session
            return {"username": "alice", "auth_method": "oauth2"}

        monkeypatch.setattr(facade, "nginx_proxied_auth", _fake)

        class _Req:
            cookies = {"mcp_gateway_session": "the-cookie-value"}

        ctx = await facade._optional_session(_Req())
        assert ctx["username"] == "alice"
        assert seen["session"] == "the-cookie-value"  # extracted + forwarded

    async def test_returns_none_when_auth_raises(self, monkeypatch):
        async def _raise(request, session=None):
            raise RuntimeError("no session")

        monkeypatch.setattr(facade, "nginx_proxied_auth", _raise)

        class _Req:
            cookies = {}

        assert await facade._optional_session(_Req()) is None


@pytest.mark.unit
class TestResourceParam:
    """resource-param parsing: a client may pass the canonical server identifier
    or the PRM document URL form; recover the server path from either."""

    def test_parses_canonical_resource_identifier(self, monkeypatch):
        monkeypatch.setattr(facade.settings, "registry_url", REGISTRY_URL)
        assert facade._server_path_from_resource(f"{REGISTRY_URL}/github") == "/github"

    def test_parses_prm_document_url_form(self, monkeypatch):
        monkeypatch.setattr(facade.settings, "registry_url", REGISTRY_URL)
        url = f"{REGISTRY_URL}/.well-known/oauth-protected-resource/github"
        assert facade._server_path_from_resource(url) == "/github"

    def test_bare_path_accepted(self, monkeypatch):
        monkeypatch.setattr(facade.settings, "registry_url", REGISTRY_URL)
        assert facade._server_path_from_resource("/github") == "/github"

    def test_unrelated_resource_rejected(self, monkeypatch):
        monkeypatch.setattr(facade.settings, "registry_url", REGISTRY_URL)
        assert facade._server_path_from_resource("https://evil.com/github") == ""

    def test_empty_resource(self, monkeypatch):
        monkeypatch.setattr(facade.settings, "registry_url", REGISTRY_URL)
        assert facade._server_path_from_resource("") == ""


@pytest.mark.unit
class TestConnectRoute:
    """The param-free /oauth2/egress/connect front door for MCP URL-mode
    elicitation. It takes NO client OAuth params (no redirect_uri/PKCE/DCR), so it
    works with providers (Entra) that lack DCR. It session-verifies the opener
    (anti-phishing) then 302s to provider consent using the user's real
    session_id (web Connected-Accounts callback path)."""

    def test_connect_redirects_to_provider(self, client):
        c = client(server=_server())
        r = c.get("/oauth2/egress/connect", params={"server": "/github"})
        assert r.status_code == 302
        assert r.headers["location"].startswith("https://github.com/login/oauth/authorize")
        # Uses the real session_id, never a facade-marked one (no leg-1 resume).
        assert not (c._svc.last_session_id or "").startswith("facade:")

    def test_connect_no_session_bounces_to_keycloak_login(self, client):
        c = client(user_context=None, server=_server())
        r = c.get("/oauth2/egress/connect", params={"server": "/github"})
        assert r.status_code == 302
        loc = r.headers["location"]
        assert loc.startswith(f"{REGISTRY_URL}/oauth2/login/keycloak")
        # returns to /connect with its query preserved (anti-phishing: the opener
        # must authenticate before consent starts)
        assert "oauth2%2Fegress%2Fconnect" in loc

    def test_connect_non_per_user_denied(self, client):
        c = client(user_context=STATIC, server=_server())
        r = c.get("/oauth2/egress/connect", params={"server": "/github"})
        assert r.status_code == 403

    def test_connect_missing_server_400(self, client):
        c = client(server=_server())
        r = c.get("/oauth2/egress/connect")
        assert r.status_code == 400

    def test_connect_unconfigured_server_400(self, client):
        c = client(server=_server(egress_auth_mode="none", egress_oauth=None))
        r = c.get("/oauth2/egress/connect", params={"server": "/github"})
        assert r.status_code == 400

    def test_connect_feature_disabled_404(self, client):
        c = client(server=_server(), enabled=False)
        r = c.get("/oauth2/egress/connect", params={"server": "/github"})
        assert r.status_code == 404

    def test_connect_accepts_resource_param_form(self, client):
        # A client may pass the resource identifier instead of ?server.
        c = client(server=_server())
        r = c.get(
            "/oauth2/egress/connect",
            params={"resource": f"{REGISTRY_URL}/github"},
        )
        assert r.status_code == 302
        assert r.headers["location"].startswith("https://github.com/login/oauth/authorize")
