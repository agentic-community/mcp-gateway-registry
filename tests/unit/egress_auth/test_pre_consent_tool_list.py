"""Registry side of listing stored tools before egress consent (issue #1986).

An admin can opt an ``oauth_user`` or ``pat`` server in so that a user who has
not connected yet sees the server's stored tools on a direct-route
``tools/list``, the way a virtual route already shows them. Covers:
    - ``_pre_consent_tool_list``: when it returns a list, which tool_list it
      reads, blocked tools, the size cap.
    - ``POST /internal/egress-token``: the list rides only on a consent miss,
      only when asked, only on a direct route.
    - ``POST /servers/{path}/egress-auth``: the opt-in is stored, reported, and
      refused for modes with no pre-consent state.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import registry.api.egress_auth_routes as routes

pytestmark = pytest.mark.unit

UPSTREAM: str = "https://api.githubcopilot.com/mcp"
STORED_TOOLS: list[dict] = [
    {"name": "get_me", "description": "Who am I", "inputSchema": {"type": "object"}},
    {
        "name": "list_repos",
        "parsed_description": {"main": "List repositories"},
        "schema": {"properties": {"org": {"type": "string"}}},
    },
]


def _server(**over) -> dict:
    base = {
        "path": "/github-mcp",
        "egress_auth_mode": "oauth_user",
        "egress_oauth": {"provider": "github", "client_id": "Iv1.x"},
        "proxy_pass_url": UPSTREAM,
        "versions": [],
        "tool_list": STORED_TOOLS,
        "egress_list_tools_before_consent": True,
    }
    base.update(over)
    return base


def _claims(**over) -> dict:
    base = {
        "sub": "alice",
        "server": "github-mcp",
        "version_id": "",
        "virtual_backend": False,
        "egress_user": "alice",
        "auth_method": "oauth2",
        "upstream_url": UPSTREAM,
    }
    base.update(over)
    return base


class _StubRepo:
    def __init__(self, server: dict, versions: dict | None = None):
        self._server = server
        self._versions = versions or {}

    async def get(self, path):
        if path == "/github-mcp":
            return self._server
        return self._versions.get(path)


class _StubEgressService:
    def __init__(self, token: str | None):
        self._token = token

    async def get_valid_token(self, **kwargs):
        return self._token

    async def get_pat(self, **kwargs):
        return self._token

    async def build_consent_url(self, **kwargs):
        return "https://github.com/login/oauth/authorize?from=miss"


@pytest.fixture
def vend_client(monkeypatch):
    """Build a client for POST /internal/egress-token with stubbed collaborators."""

    def _build(claims: dict, server: dict, token: str | None = None, versions=None):
        monkeypatch.setattr(routes.settings, "egress_auth_enabled", True)
        monkeypatch.setattr(routes, "verify_mcp_proxy_token", lambda tok: claims)
        repo = _StubRepo(server, versions)
        monkeypatch.setattr(routes, "get_server_repository", lambda: repo)
        monkeypatch.setattr("registry.repositories.factory.get_server_repository", lambda: repo)
        monkeypatch.setattr(routes, "get_egress_auth_service", lambda: _StubEgressService(token))
        app = FastAPI()
        app.include_router(routes.router)
        app.dependency_overrides[routes.validate_internal_auth] = lambda: "auth-server"
        return TestClient(app)

    return _build


def _vend(client: TestClient, include: bool = True) -> dict:
    response = client.post(
        "/internal/egress-token",
        json={"server_path": "/github-mcp", "include_pre_consent_tools": include},
        headers={"X-Internal-Token": "proxy-token"},
    )
    assert response.status_code == 200, response.text
    return response.json()


class TestPreConsentToolListHelper:
    """The decision and conversion, on documents the vend already holds."""

    def test_returns_none_when_not_requested(self):
        server = _server()
        assert routes._pre_consent_tool_list(server, server, True, False) is None

    def test_returns_none_off_a_direct_route(self):
        server = _server()
        assert routes._pre_consent_tool_list(server, server, False, True) is None

    def test_returns_none_when_server_not_opted_in(self):
        server = _server(egress_list_tools_before_consent=False)
        assert routes._pre_consent_tool_list(server, server, True, True) is None

    def test_missing_opt_in_field_reads_as_off(self):
        server = _server()
        del server["egress_list_tools_before_consent"]
        assert routes._pre_consent_tool_list(server, server, True, True) is None

    def test_converts_both_stored_shapes_to_mcp_entries(self):
        server = _server()
        tools = routes._pre_consent_tool_list(server, server, True, True)
        assert tools == [
            {"name": "get_me", "description": "Who am I", "inputSchema": {"type": "object"}},
            {
                "name": "list_repos",
                "description": "List repositories",
                "inputSchema": {"type": "object", "properties": {"org": {"type": "string"}}},
            },
        ]

    def test_uses_the_selected_version_tool_list(self):
        server = _server()
        selected = {"tool_list": [{"name": "v2_only"}]}
        tools = routes._pre_consent_tool_list(server, selected, True, True)
        assert [t["name"] for t in tools] == ["v2_only"]

    def test_linked_version_without_tool_list_borrows_the_active_one(self):
        server = _server()
        tools = routes._pre_consent_tool_list(server, {"path": "/github-mcp:v2"}, True, True)
        assert [t["name"] for t in tools] == ["get_me", "list_repos"]

    def test_drops_blocked_tools(self):
        server = _server(tool_overrides={"get_me": {"blocked": True}, "x": {"blocked": False}})
        tools = routes._pre_consent_tool_list(server, server, True, True)
        assert [t["name"] for t in tools] == ["list_repos"]

    def test_skips_malformed_entries(self):
        server = _server(tool_list=["not-a-dict", {"description": "no name"}, {"name": "ok"}])
        tools = routes._pre_consent_tool_list(server, server, True, True)
        assert [t["name"] for t in tools] == ["ok"]

    def test_caps_the_list(self, monkeypatch):
        monkeypatch.setattr(routes, "MAX_PRE_CONSENT_TOOLS", 3)
        server = _server(tool_list=[{"name": f"t{i}"} for i in range(10)])
        tools = routes._pre_consent_tool_list(server, server, True, True)
        assert len(tools) == 3

    def test_empty_tool_list_keeps_todays_response_and_warns(self, caplog):
        server = _server(tool_list=[])
        with caplog.at_level("WARNING"):
            assert routes._pre_consent_tool_list(server, server, True, True) is None
        assert "has no stored tool_list" in caplog.text


class TestVendEndpoint:
    """The list rides only on a consent miss that asked for it."""

    def test_oauth_miss_carries_the_stored_tools(self, vend_client):
        body = _vend(vend_client(_claims(), _server()))
        assert body["consent_required"] is True
        assert [t["name"] for t in body["pre_consent_tools"]] == ["get_me", "list_repos"]

    def test_oauth_hit_never_carries_the_list(self, vend_client):
        body = _vend(vend_client(_claims(), _server(), token="at_vended"))
        assert body["access_token"] == "at_vended"
        assert body["pre_consent_tools"] is None

    def test_pat_miss_carries_the_stored_tools(self, vend_client):
        server = _server(egress_auth_mode="pat", egress_oauth={"provider": "github"})
        body = _vend(vend_client(_claims(), server))
        assert body["mode"] == "pat"
        assert [t["name"] for t in body["pre_consent_tools"]] == ["get_me", "list_repos"]

    def test_request_without_the_flag_gets_no_list(self, vend_client):
        """An auth server from before #1986 never sends the flag."""
        client = vend_client(_claims(), _server())
        response = client.post(
            "/internal/egress-token",
            json={"server_path": "/github-mcp"},
            headers={"X-Internal-Token": "proxy-token"},
        )
        assert response.json()["pre_consent_tools"] is None

    def test_virtual_backend_token_gets_no_list(self, vend_client):
        body = _vend(vend_client(_claims(virtual_backend=True), _server()))
        assert body["consent_required"] is True
        assert body["pre_consent_tools"] is None

    def test_token_without_route_claims_gets_no_list(self, vend_client):
        """A hop token minted before the route claims is not treated as direct."""
        claims = _claims()
        del claims["version_id"]
        del claims["virtual_backend"]
        body = _vend(vend_client(claims, _server()))
        assert body["consent_required"] is True
        assert body["pre_consent_tools"] is None

    def test_obo_exchange_never_carries_the_list(self, vend_client, monkeypatch):
        monkeypatch.setattr(routes, "_validate_obo_egress_config", lambda aud, scopes: None)
        server = _server(
            egress_auth_mode="obo_exchange",
            egress_oauth={"target_audience": "api://internal", "scopes": []},
        )
        body = _vend(vend_client(_claims(), server))
        assert body["mode"] == "obo_exchange"
        assert body["pre_consent_tools"] is None


class _StubServerService:
    def __init__(self, server: dict):
        self._server = server
        self.updated_with: dict | None = None

    async def get_server_info(self, path, include_credentials=False):
        return dict(self._server)

    async def update_server(self, path, server):
        self.updated_with = server
        return True


@pytest.fixture
def config_client(monkeypatch):
    """Build an admin client for POST/GET /servers/{path}/egress-auth."""

    def _build(server: dict | None = None):
        monkeypatch.setattr(routes.settings, "egress_auth_enabled", True)
        svc = _StubServerService(server or {"path": "/gh", "egress_oauth": None})
        monkeypatch.setattr(routes, "server_service", svc)
        monkeypatch.setattr(routes, "encrypt_credential", lambda s: f"enc:{s}")
        app = FastAPI()
        app.include_router(routes.router)
        app.dependency_overrides[routes.nginx_proxied_auth] = lambda: {
            "username": "admin",
            "is_admin": True,
            "auth_method": "keycloak",
        }
        app.dependency_overrides[routes.verify_csrf_token_flexible] = lambda: None
        client = TestClient(app)
        client.svc = svc
        return client

    return _build


class TestEgressConfigOptIn:
    """The admin config route stores and reports the opt-in."""

    def test_pat_opt_in_is_stored_and_reported(self, config_client):
        client = config_client()
        response = client.post(
            "/servers/gh/egress-auth",
            json={
                "egress_auth_mode": "pat",
                "egress_provider": "github",
                "list_tools_before_consent": True,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["list_tools_before_consent"] is True
        assert client.svc.updated_with["egress_list_tools_before_consent"] is True

    def test_omitting_the_field_turns_it_off(self, config_client):
        client = config_client({"path": "/gh", "egress_list_tools_before_consent": True})
        response = client.post(
            "/servers/gh/egress-auth",
            json={"egress_auth_mode": "pat", "egress_provider": "github"},
        )
        assert response.status_code == 200, response.text
        assert client.svc.updated_with["egress_list_tools_before_consent"] is False

    @pytest.mark.parametrize("mode", ["none", "obo_exchange"])
    def test_opt_in_is_refused_without_a_pre_consent_state(self, config_client, mode, monkeypatch):
        monkeypatch.setattr(routes, "_validate_obo_egress_config", lambda aud, scopes: None)
        client = config_client()
        response = client.post(
            "/servers/gh/egress-auth",
            json={
                "egress_auth_mode": mode,
                "target_audience": "api://internal",
                "list_tools_before_consent": True,
            },
        )
        assert response.status_code == 400
        assert "oauth_user or pat" in response.json()["detail"]
        assert client.svc.updated_with is None

    def test_config_view_reports_the_stored_value(self, config_client):
        client = config_client(
            {"path": "/gh", "egress_auth_mode": "pat", "egress_list_tools_before_consent": True}
        )
        response = client.get("/servers/gh/egress-auth")
        assert response.status_code == 200, response.text
        assert response.json()["list_tools_before_consent"] is True
