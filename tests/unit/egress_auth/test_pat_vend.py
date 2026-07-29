"""Vend-path tests for the pat branch of POST /internal/egress-token.

The pat branch runs BEFORE the oauth_user vault lookup and never routes a pat
server through get_valid_token (the OAuth refresh path). A HIT returns
{access_token}; a miss (never submitted OR expired) returns {mode:"pat"} with no
connect_url/authorize_url. The per-user + upstream cross-checks apply exactly as
for the other modes.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import registry.api.egress_auth_routes as routes


class _StubRepo:
    def __init__(self, server):
        self._server = server

    async def get(self, path):
        return self._server


class _StubService:
    """Records which vend path was taken so we can assert get_valid_token is
    never called for a pat server."""

    def __init__(self, pat_token):
        self._pat_token = pat_token
        self.get_pat_called = False
        self.get_valid_token_called = False
        self.get_pat_kwargs = None

    async def get_pat(self, **kwargs):
        self.get_pat_called = True
        self.get_pat_kwargs = kwargs
        return self._pat_token

    async def get_valid_token(self, **kwargs):
        self.get_valid_token_called = True
        return "SHOULD-NOT-BE-USED"

    def build_consent_url(self, **kwargs):
        return "https://example.com/authorize"


def _server(**over):
    base = {
        "egress_auth_mode": "pat",
        "egress_oauth": {"provider": "github"},
        "proxy_pass_url": "https://api.githubcopilot.com/mcp",
        "versions": [],
    }
    base.update(over)
    return base


@pytest.fixture
def make_client(monkeypatch):
    def _build(claims, server, pat_token="ghp_vended", enabled=True):
        monkeypatch.setattr(routes.settings, "egress_auth_enabled", enabled)
        monkeypatch.setattr(routes, "verify_mcp_proxy_token", lambda tok: claims)
        monkeypatch.setattr(routes, "get_server_repository", lambda: _StubRepo(server))
        svc = _StubService(pat_token)
        monkeypatch.setattr(routes, "get_egress_auth_service", lambda: svc)

        app = FastAPI()
        app.include_router(routes.router)
        app.dependency_overrides[routes.validate_internal_auth] = lambda: "auth-server"
        client = TestClient(app)
        client._svc = svc
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


def _post(client, token="proxy-token", server_path="/github"):
    return client.post(
        "/internal/egress-token",
        json={"server_path": server_path},
        headers={"X-Internal-Token": token},
    )


@pytest.mark.unit
class TestPatVend:
    def test_hit_injects_access_token(self, make_client):
        client = make_client(_claims(), _server(), pat_token="ghp_vended")
        r = _post(client)
        assert r.status_code == 200
        body = r.json()
        assert body["access_token"] == "ghp_vended"
        assert body["consent_required"] is False
        # pat runs via get_pat, NEVER the OAuth refresh path.
        assert client._svc.get_pat_called
        assert not client._svc.get_valid_token_called
        # With no Backend Auth scheme set, the inject header defaults to
        # Authorization: Bearer so mcp_proxy can still build the header.
        assert body["pat_header_name"] == "Authorization"
        assert body["pat_value_prefix"] == "Bearer "

    def test_hit_inherits_inject_header_from_backend_auth(self, make_client):
        # The inject header is derived from the server's Backend Auth scheme:
        # api_key -> "<header>: <bare-PAT>" (no prefix). No per-mode config.
        server = _server(
            egress_oauth={"provider": "gitlab"},
            auth_scheme="api_key",
            auth_header_name="PRIVATE-TOKEN",
        )
        client = make_client(_claims(), server, pat_token="glpat_vended")
        r = _post(client)
        assert r.status_code == 200
        body = r.json()
        assert body["access_token"] == "glpat_vended"
        assert body["pat_header_name"] == "PRIVATE-TOKEN"
        assert body["pat_value_prefix"] == ""

    def test_get_pat_receives_verified_identity(self, make_client):
        client = make_client(_claims(), _server())
        _post(client)
        kwargs = client._svc.get_pat_kwargs
        assert kwargs["auth_method"] == "oauth2"
        assert kwargs["user_id"] == "alice"
        assert kwargs["provider"] == "github"
        assert kwargs["server_path"] == "/github"

    def test_miss_returns_mode_pat_no_connect_url(self, make_client):
        # An expired/absent PAT is a miss: get_pat returns None -> mode=pat, no URL.
        client = make_client(_claims(), _server(), pat_token=None)
        r = _post(client)
        assert r.status_code == 200
        body = r.json()
        assert body["consent_required"] is True
        assert body["mode"] == "pat"
        assert body["access_token"] is None
        assert body.get("connect_url") is None
        assert body.get("authorize_url") is None
        assert not client._svc.get_valid_token_called

    def test_egress_user_claim_preferred_over_sub(self, make_client):
        client = make_client(_claims(egress_user="alice-oidc-sub"), _server())
        _post(client)
        assert client._svc.get_pat_kwargs["user_id"] == "alice-oidc-sub"

    def test_upstream_mismatch_403(self, make_client):
        # A forged upstream not in the registered set -> refuse before vend.
        client = make_client(_claims(upstream_url="https://attacker.example/mcp"), _server())
        r = _post(client)
        assert r.status_code == 403
        assert not client._svc.get_pat_called

    def test_non_per_user_consent_no_vend(self, make_client):
        # Network-trusted/federation callers never vend a PAT.
        client = make_client(_claims(auth_method="network-trusted"), _server())
        r = _post(client)
        assert r.status_code == 200
        assert r.json()["consent_required"] is True
        assert not client._svc.get_pat_called

    def test_unknown_server_consent(self, make_client):
        client = make_client(_claims(), None)
        r = _post(client)
        assert r.json()["consent_required"] is True
        assert not client._svc.get_pat_called

    def test_server_without_egress_oauth_consent(self, make_client):
        client = make_client(_claims(), _server(egress_oauth=None))
        r = _post(client)
        assert r.json()["consent_required"] is True
        assert not client._svc.get_pat_called


# ---------------------------------------------------------------------------- #
# Direct EgressAuthService.get_pat tests (expiry + client_id=None handling).
# ---------------------------------------------------------------------------- #


from datetime import UTC, datetime, timedelta  # noqa: E402

from registry.egress_auth.schemas import StoredToken  # noqa: E402
from registry.egress_auth.service import EgressAuthService  # noqa: E402
from registry.secrets.interfaces import SecretStoreBase  # noqa: E402


class _InMemoryStore(SecretStoreBase):
    def __init__(self) -> None:
        self._data: dict[tuple[str, str, str, str], StoredToken] = {}

    async def put_token(self, auth_method, user_id, provider, server_path, token):
        self._data[(auth_method, user_id, provider, server_path)] = token

    async def get_token(self, auth_method, user_id, provider, server_path):
        return self._data.get((auth_method, user_id, provider, server_path))

    async def delete_token(self, auth_method, user_id, provider, server_path):
        self._data.pop((auth_method, user_id, provider, server_path), None)

    async def list_for_user(self, auth_method, user_id):
        return [
            (provider, server_path, token)
            for (am, uid, provider, server_path), token in self._data.items()
            if am == auth_method and uid == user_id
        ]


def _svc():
    return EgressAuthService(secret_store=_InMemoryStore(), callback_base_url="https://gw.example")


def _future_iso(seconds: int) -> str:
    return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat()


@pytest.mark.unit
class TestGetPatService:
    async def test_hit_with_client_id_none_vends(self):
        # A PAT stores client_id=None; get_pat must NOT apply the OAuth
        # client_id-rotation check (that would reject it). It vends.
        svc = _svc()
        await svc._store.put_token(
            "oauth2",
            "alice",
            "github",
            "/github",
            StoredToken(access_token="ghp_x", client_id=None, expires_at=_future_iso(3600)),
        )
        token = await svc.get_pat("oauth2", "alice", "github", "/github")
        assert token == "ghp_x"

    async def test_expired_pat_is_miss(self):
        svc = _svc()
        await svc._store.put_token(
            "oauth2",
            "alice",
            "github",
            "/github",
            StoredToken(access_token="ghp_x", expires_at="2000-01-01T00:00:00+00:00"),
        )
        assert await svc.get_pat("oauth2", "alice", "github", "/github") is None

    async def test_missing_expires_at_is_miss(self):
        # A pat entry without expires_at is treated as a miss (fail closed) --
        # the submit endpoint always stamps one, so a bare entry is anomalous.
        svc = _svc()
        await svc._store.put_token(
            "oauth2", "alice", "github", "/github", StoredToken(access_token="ghp_x")
        )
        assert await svc.get_pat("oauth2", "alice", "github", "/github") is None

    async def test_never_submitted_is_miss(self):
        assert await _svc().get_pat("oauth2", "nobody", "github", "/github") is None

    async def test_non_per_user_is_miss(self):
        svc = _svc()
        await svc._store.put_token(
            "network-trusted",
            "alice",
            "github",
            "/github",
            StoredToken(access_token="ghp_x", expires_at=_future_iso(3600)),
        )
        assert await svc.get_pat("network-trusted", "alice", "github", "/github") is None
