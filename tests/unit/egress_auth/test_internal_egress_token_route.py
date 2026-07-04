"""Authz tests for POST /internal/egress-token.

Drives each security branch with the dependencies stubbed:
- validate_internal_auth overridden (caller already authenticated).
- verify_mcp_proxy_token monkeypatched to return controlled claims (this is
  covered separately in test_verify_mcp_proxy_token.py).
- get_server_repository / get_egress_auth_service stubbed.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import registry.api.egress_auth_routes as routes


class _StubRepo:
    def __init__(self, server):
        self._server = server
        self.queried_paths: list[str] = []

    async def get(self, path):
        self.queried_paths.append(path)
        return self._server


class _StubService:
    def __init__(self, token):
        self._token = token
        self.called = False

    async def get_valid_token(self, **kwargs):
        self.called = True
        return self._token

    def build_consent_url(self, **kwargs):
        return "https://github.com/login/oauth/authorize?from=miss"


def _server(**over):
    base = {
        "egress_auth_mode": "oauth_user",
        "egress_oauth": {"provider": "github", "client_id": "Iv1.x"},
        "proxy_pass_url": "https://api.githubcopilot.com/mcp",
        "versions": [],
    }
    base.update(over)
    return base


@pytest.fixture
def make_client(monkeypatch):
    """Factory: build a TestClient with controllable claims/server/token."""

    def _build(claims, server, vended_token="at_vended", enabled=True):
        monkeypatch.setattr(routes.settings, "egress_auth_enabled", enabled)
        monkeypatch.setattr(routes, "verify_mcp_proxy_token", lambda tok: claims)
        repo = _StubRepo(server)
        monkeypatch.setattr(routes, "get_server_repository", lambda: repo)
        svc = _StubService(vended_token)
        monkeypatch.setattr(routes, "get_egress_auth_service", lambda: svc)

        app = FastAPI()
        app.include_router(routes.router)
        app.dependency_overrides[routes.validate_internal_auth] = lambda: "auth-server"
        client = TestClient(app)
        client._svc = svc  # expose for assertions
        client._repo = repo
        return client

    return _build


def _claims(**over):
    base = {
        "sub": "alice",
        "auth_method": "oauth2",
        "upstream_url": "https://api.githubcopilot.com/mcp",
    }
    base.update(over)
    return base


def _post(client, token="proxy-token", server_path="/github-mcp"):
    return client.post(
        "/internal/egress-token",
        json={"server_path": server_path},
        headers={"X-Internal-Token": token},
    )


@pytest.mark.unit
class TestInternalEgressTokenRoute:
    def test_happy_path_vends(self, make_client):
        client = make_client(_claims(), _server())
        r = _post(client)
        assert r.status_code == 200
        assert r.json()["access_token"] == "at_vended"
        assert client._svc.called

    def test_no_leading_slash_is_normalized(self, make_client):
        # mcp_proxy sends the bare first path segment ("github"); the endpoint must
        # normalize to "/github" so the server lookup + vault key + consent agree.
        client = make_client(_claims(), _server())
        r = _post(client, server_path="github-mcp")  # no leading slash
        assert r.status_code == 200
        assert r.json()["access_token"] == "at_vended"
        assert client._repo.queried_paths == ["/github-mcp"]  # normalized before lookup

    def test_feature_disabled_404(self, make_client):
        client = make_client(_claims(), _server(), enabled=False)
        assert _post(client).status_code == 404

    def test_missing_internal_token_401(self, make_client):
        client = make_client(_claims(), _server())
        r = client.post("/internal/egress-token", json={"server_path": "/github-mcp"})
        assert r.status_code == 401

    def test_non_per_user_auth_method_consent_no_vend(self, make_client):
        # Network-trusted/federation callers never vend.
        client = make_client(_claims(auth_method="network-trusted"), _server())
        r = _post(client)
        assert r.status_code == 200
        assert r.json()["consent_required"] is True
        assert r.json()["access_token"] is None
        assert not client._svc.called

    def test_server_not_oauth_user_consent(self, make_client):
        client = make_client(_claims(), _server(egress_auth_mode="none", egress_oauth=None))
        r = _post(client)
        assert r.json()["consent_required"] is True
        assert not client._svc.called

    def test_unknown_server_consent(self, make_client):
        client = make_client(_claims(), None)
        assert _post(client).json()["consent_required"] is True

    def test_upstream_mismatch_403(self, make_client):
        # Forged upstream not in the registered set -> refuse.
        client = make_client(_claims(upstream_url="https://attacker.example/mcp"), _server())
        r = _post(client)
        assert r.status_code == 403
        assert not client._svc.called

    def test_multi_version_upstream_accepted(self, make_client):
        # Union: a versioned upstream (not the base proxy_pass_url) is legal.
        srv = _server(
            versions=[{"version": "v2", "proxy_pass_url": "https://v2.githubcopilot.com/mcp"}]
        )
        client = make_client(_claims(upstream_url="https://v2.githubcopilot.com/mcp/sub"), srv)
        # note: base-URL comparison ignores the sub-path; v2 host matches the union
        r = _post(client)
        assert r.status_code == 200
        assert client._svc.called

    def test_vend_miss_returns_consent_url(self, make_client):
        # On a miss for an egress-configured server, the vend builds + returns the
        # authorize_url so mcp_proxy can hand it to the user (auto-consent trigger).
        client = make_client(_claims(), _server(), vended_token=None)
        r = _post(client)
        body = r.json()
        assert body["consent_required"] is True
        assert body["access_token"] is None
        assert body["authorize_url"] == "https://github.com/login/oauth/authorize?from=miss"

    def test_vend_miss_returns_connect_url_and_request_state(self, make_client):
        # The URL-mode elicitation switch adds a session-verified connect_url
        # (the gateway front door the MCP client opens, no DCR), an AEAD
        # request_state blob for the MRTR retry, and the provider key.
        client = make_client(_claims(), _server(), vended_token=None)
        body = _post(client, server_path="/github-mcp").json()
        assert body["consent_required"] is True
        # connect_url points at the gateway's own /oauth2/egress/connect with the
        # server path, NOT the provider-direct authorize_url.
        assert "/oauth2/egress/connect" in body["connect_url"]
        assert "server=" in body["connect_url"]
        assert "github" in body["connect_url"]
        # request_state is an opaque, non-empty AEAD blob (decodable by the codec).
        assert body["request_state"]
        from registry.egress_auth.state_codec import decode_state

        st = decode_state(body["request_state"])
        assert st.user_id == "alice"
        assert st.auth_method == "oauth2"
        assert st.provider == "github"
        assert st.server_path == "/github-mcp"
        assert body["provider"] == "github"

    def test_non_per_user_miss_has_no_authorize_url(self, make_client):
        # A non-per-user caller has nothing to connect -> no authorize_url and no
        # connect_url (mcp_proxy then falls through to normal forwarding).
        client = make_client(_claims(auth_method="network-trusted"), _server())
        body = _post(client).json()
        assert body["consent_required"] is True
        assert body.get("authorize_url") is None
        assert body.get("connect_url") is None
        assert body.get("request_state") is None
