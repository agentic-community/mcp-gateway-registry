"""PUT/GET/DELETE /servers/{path}/egress-pat route tests.

TestClient over the router with dependencies stubbed: nginx_proxied_auth (user
context), CSRF (no-op by default), server_service (async-mocked), and the
secret store / EgressAuthService. Covers the critical authz paths (admin-gated
sub, CSRF), the mandatory bounded TTL, and the write-only invariant.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import registry.api.egress_auth_routes as routes
from registry.egress_auth.schemas import StoredToken
from registry.secrets.interfaces import SecretStoreError

ADMIN = {"username": "admin", "is_admin": True, "auth_method": "oauth2", "egress_user": "admin"}
USER = {"username": "alice", "is_admin": False, "auth_method": "oauth2", "egress_user": "alice"}
STATIC = {"username": "ci-bot", "is_admin": False, "auth_method": "network-trusted"}


def _route_has_csrf_dependency(method: str, path: str) -> bool:
    """True if the router's route for (method, path) depends on the CSRF verifier.

    Inspects the registered FastAPI route's dependant tree rather than driving a
    live request (the CSRF dependency only enforces when a session cookie is
    present, which a stubbed unit request cannot faithfully simulate).
    """
    for route in routes.router.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            calls = [d.call for d in route.dependant.dependencies]
            return routes.verify_csrf_token_flexible in calls
    return False


def _server(**over):
    base = {
        "path": "/github",
        "proxy_pass_url": "https://api.githubcopilot.com/mcp",
        "egress_auth_mode": "pat",
        "egress_oauth": {"provider": "github", "scopes": []},
    }
    base.update(over)
    return base


class _StubStore:
    """Records put/delete/get for assertions."""

    def __init__(self, token=None, put_error=None):
        self._token = token
        self._put_error = put_error
        self.put_calls: list[tuple] = []
        self.delete_calls: list[tuple] = []

    async def put_token(self, auth_method, user_id, provider, server_path, token):
        if self._put_error:
            raise self._put_error
        self.put_calls.append((auth_method, user_id, provider, server_path, token))

    async def get_token(self, auth_method, user_id, provider, server_path):
        return self._token

    async def delete_token(self, auth_method, user_id, provider, server_path):
        self.delete_calls.append((auth_method, user_id, provider, server_path))


@pytest.fixture
def client(monkeypatch):
    def _build(
        user_context,
        *,
        server=None,
        store=None,
        svc=None,
        enabled=True,
        csrf_ok=True,
    ):
        monkeypatch.setattr(routes.settings, "egress_auth_enabled", enabled)
        monkeypatch.setattr(
            routes.server_service, "get_server_info", AsyncMock(return_value=server)
        )
        if store is not None:
            monkeypatch.setattr(routes, "get_secret_store", lambda: store)
        if svc is not None:
            monkeypatch.setattr(routes, "get_egress_auth_service", lambda: svc)

        app = FastAPI()
        app.include_router(routes.router, prefix="/api")
        app.dependency_overrides[routes.nginx_proxied_auth] = lambda: user_context
        if csrf_ok:
            app.dependency_overrides[routes.verify_csrf_token_flexible] = lambda: None
        # When csrf_ok is False we leave the real CSRF dependency in place so the
        # request is rejected the way a missing/invalid token would be.
        c = TestClient(app)
        c._store = store
        c._svc = svc
        return c

    return _build


def _put_body(**over):
    base = {"secret": "ghp_secret", "ttl_value": 7, "ttl_unit": "days"}
    base.update(over)
    return base


@pytest.mark.unit
class TestSetPat:
    def test_submit_stores_pat_secret_never_echoed(self, client):
        store = _StubStore()
        c = client(USER, server=_server(), store=store)
        r = c.put("/api/servers/github/egress-pat", json=_put_body())
        assert r.status_code == 200
        body = r.json()
        assert body["configured"] is True
        assert body["sub"] == "alice"
        assert body["expires_at"]
        # The secret is NEVER returned by any endpoint.
        assert "ghp_secret" not in str(body)
        assert "secret" not in body
        # Stored under the verified identity's canonical key.
        assert len(store.put_calls) == 1
        auth_method, user_id, provider, path, token = store.put_calls[0]
        assert (auth_method, user_id, provider, path) == ("oauth2", "alice", "github", "/github")
        assert token.access_token == "ghp_secret"
        assert token.expires_at

    def test_expires_at_is_now_plus_ttl(self, client):
        store = _StubStore()
        c = client(USER, server=_server(), store=store)
        before = datetime.now(UTC)
        r = c.put("/api/servers/github/egress-pat", json=_put_body(ttl_value=1, ttl_unit="hours"))
        after = datetime.now(UTC)
        exp = datetime.fromisoformat(r.json()["expires_at"])
        assert before + timedelta(hours=1) - timedelta(seconds=5) <= exp
        assert exp <= after + timedelta(hours=1) + timedelta(seconds=5)

    def test_non_admin_supplying_sub_403(self, client):
        # The single most important rule: a non-admin cannot store into another
        # user's bucket, and there is NO silent fall-back to self.
        store = _StubStore()
        c = client(USER, server=_server(), store=store)
        r = c.put("/api/servers/github/egress-pat", json=_put_body(sub="bob"))
        assert r.status_code == 403
        assert store.put_calls == []

    def test_admin_may_submit_on_behalf(self, client):
        # On-behalf REQUIRES the target's auth_method (the partition the target
        # vends from); it is NOT assumed from the admin's own auth_method.
        store = _StubStore()
        c = client(ADMIN, server=_server(), store=store)
        r = c.put(
            "/api/servers/github/egress-pat",
            json=_put_body(sub="bob", auth_method="oauth2"),
        )
        assert r.status_code == 200
        assert r.json()["sub"] == "bob"
        auth_method, user_id, _, _, _ = store.put_calls[0]
        # Stored under the TARGET's (auth_method, sub), not the admin's.
        assert (auth_method, user_id) == ("oauth2", "bob")

    def test_admin_on_behalf_missing_auth_method_400(self, client):
        # Fail closed: an on-behalf write without the target auth_method would
        # land the PAT in the wrong vault partition, so it is rejected.
        store = _StubStore()
        c = client(ADMIN, server=_server(), store=store)
        r = c.put("/api/servers/github/egress-pat", json=_put_body(sub="bob"))
        assert r.status_code == 400
        assert store.put_calls == []

    def test_admin_on_behalf_uses_target_auth_method_not_admins(self, client):
        # The admin logs in via oauth2 but the target vends via jwt: the PAT must
        # be stored under jwt (the target's partition), never the admin's oauth2.
        store = _StubStore()
        c = client(ADMIN, server=_server(), store=store)
        r = c.put(
            "/api/servers/github/egress-pat",
            json=_put_body(sub="bob", auth_method="jwt"),
        )
        assert r.status_code == 200
        auth_method, user_id, _, _, _ = store.put_calls[0]
        assert (auth_method, user_id) == ("jwt", "bob")

    def test_admin_on_behalf_non_per_user_auth_method_403(self, client):
        # A non-per-user target auth_method can never own a per-user PAT.
        store = _StubStore()
        c = client(ADMIN, server=_server(), store=store)
        r = c.put(
            "/api/servers/github/egress-pat",
            json=_put_body(sub="bob", auth_method="network-trusted"),
        )
        assert r.status_code == 403
        assert store.put_calls == []

    def test_empty_secret_400(self, client):
        store = _StubStore()
        c = client(USER, server=_server(), store=store)
        r = c.put("/api/servers/github/egress-pat", json=_put_body(secret=""))
        assert r.status_code == 400
        assert store.put_calls == []

    @pytest.mark.parametrize(
        "ttl_value,ttl_unit",
        [(0, "days"), (-1, "hours"), (1, "weeks"), (31, "days"), (43201, "minutes")],
    )
    def test_bad_ttl_400(self, client, ttl_value, ttl_unit):
        store = _StubStore()
        c = client(USER, server=_server(), store=store)
        r = c.put(
            "/api/servers/github/egress-pat",
            json=_put_body(ttl_value=ttl_value, ttl_unit=ttl_unit),
        )
        assert r.status_code == 400
        assert store.put_calls == []

    @pytest.mark.parametrize("ttl_unit", ["minutes", "hours", "days"])
    def test_good_ttl_units_accepted(self, client, ttl_unit):
        store = _StubStore()
        c = client(USER, server=_server(), store=store)
        r = c.put("/api/servers/github/egress-pat", json=_put_body(ttl_value=1, ttl_unit=ttl_unit))
        assert r.status_code == 200

    def test_non_pat_server_409(self, client):
        store = _StubStore()
        c = client(USER, server=_server(egress_auth_mode="oauth_user"), store=store)
        r = c.put("/api/servers/github/egress-pat", json=_put_body())
        assert r.status_code == 409
        assert store.put_calls == []

    def test_unknown_server_404(self, client):
        store = _StubStore()
        c = client(USER, server=None, store=store)
        r = c.put("/api/servers/github/egress-pat", json=_put_body())
        assert r.status_code == 404

    def test_non_per_user_caller_403(self, client):
        store = _StubStore()
        c = client(STATIC, server=_server(), store=store)
        r = c.put("/api/servers/github/egress-pat", json=_put_body())
        assert r.status_code == 403
        assert store.put_calls == []

    def test_store_unavailable_503_nothing_written(self, client):
        store = _StubStore(put_error=SecretStoreError("vault down"))
        c = client(USER, server=_server(), store=store)
        r = c.put("/api/servers/github/egress-pat", json=_put_body())
        assert r.status_code == 503

    def test_feature_disabled_404(self, client):
        c = client(USER, server=_server(), store=_StubStore(), enabled=False)
        r = c.put("/api/servers/github/egress-pat", json=_put_body())
        assert r.status_code == 404

    def test_put_is_csrf_protected(self):
        # The mutating endpoint must depend on verify_csrf_token_flexible.
        assert _route_has_csrf_dependency("PUT", "/servers/{server_path:path}/egress-pat")


def _stored(**over):
    base = {"access_token": "ghp_secret", "expires_at": None}
    base.update(over)
    return StoredToken(**base)


@pytest.mark.unit
class TestGetPatStatus:
    def _svc(self, token):
        svc = AsyncMock()
        svc.get_pat_status = AsyncMock(return_value=token)
        return svc

    def test_status_configured_never_returns_secret(self, client):
        future = (datetime.now(UTC) + timedelta(days=1)).isoformat()
        svc = self._svc(_stored(expires_at=future))
        c = client(USER, server=_server(), svc=svc)
        r = c.get("/api/servers/github/egress-pat")
        assert r.status_code == 200
        body = r.json()
        assert body["configured"] is True
        assert body["expires_at"] == future
        assert body["expired"] is False
        assert "ghp_secret" not in str(body)
        assert "access_token" not in str(body)

    def test_status_expired_flag(self, client):
        svc = self._svc(_stored(expires_at="2000-01-01T00:00:00+00:00"))
        c = client(USER, server=_server(), svc=svc)
        body = c.get("/api/servers/github/egress-pat").json()
        assert body["configured"] is True
        assert body["expired"] is True

    def test_status_miss(self, client):
        svc = self._svc(None)
        c = client(USER, server=_server(), svc=svc)
        body = c.get("/api/servers/github/egress-pat").json()
        assert body["configured"] is False
        assert body["expires_at"] is None
        assert body["expired"] is False

    def test_non_admin_sub_query_403(self, client):
        svc = self._svc(None)
        c = client(USER, server=_server(), svc=svc)
        r = c.get("/api/servers/github/egress-pat", params={"sub": "bob"})
        assert r.status_code == 403

    def test_admin_sub_query_allowed(self, client):
        svc = self._svc(None)
        c = client(ADMIN, server=_server(), svc=svc)
        r = c.get(
            "/api/servers/github/egress-pat",
            params={"sub": "bob", "auth_method": "oauth2"},
        )
        assert r.status_code == 200
        kwargs = svc.get_pat_status.await_args.kwargs
        # Reads the TARGET's (auth_method, sub) partition.
        assert kwargs["user_id"] == "bob"
        assert kwargs["auth_method"] == "oauth2"

    def test_admin_sub_query_missing_auth_method_400(self, client):
        svc = self._svc(None)
        c = client(ADMIN, server=_server(), svc=svc)
        r = c.get("/api/servers/github/egress-pat", params={"sub": "bob"})
        assert r.status_code == 400
        assert svc.get_pat_status.await_count == 0


@pytest.mark.unit
class TestDeletePat:
    def test_delete_idempotent(self, client):
        svc = AsyncMock()
        svc.delete_pat = AsyncMock(return_value=None)
        c = client(USER, server=_server(), svc=svc)
        r = c.request("DELETE", "/api/servers/github/egress-pat")
        assert r.status_code == 200
        assert r.json() == {"path": "/github", "configured": False}
        assert svc.delete_pat.await_count == 1

    def test_delete_non_admin_sub_403(self, client):
        svc = AsyncMock()
        svc.delete_pat = AsyncMock(return_value=None)
        c = client(USER, server=_server(), svc=svc)
        r = c.request("DELETE", "/api/servers/github/egress-pat", params={"sub": "bob"})
        assert r.status_code == 403
        assert svc.delete_pat.await_count == 0

    def test_delete_admin_on_behalf_uses_target_partition(self, client):
        svc = AsyncMock()
        svc.delete_pat = AsyncMock(return_value=None)
        c = client(ADMIN, server=_server(), svc=svc)
        r = c.request(
            "DELETE",
            "/api/servers/github/egress-pat",
            params={"sub": "bob", "auth_method": "jwt"},
        )
        assert r.status_code == 200
        kwargs = svc.delete_pat.await_args.kwargs
        assert (kwargs["auth_method"], kwargs["user_id"]) == ("jwt", "bob")

    def test_delete_admin_on_behalf_missing_auth_method_400(self, client):
        svc = AsyncMock()
        svc.delete_pat = AsyncMock(return_value=None)
        c = client(ADMIN, server=_server(), svc=svc)
        r = c.request("DELETE", "/api/servers/github/egress-pat", params={"sub": "bob"})
        assert r.status_code == 400
        assert svc.delete_pat.await_count == 0

    def test_delete_is_csrf_protected(self):
        assert _route_has_csrf_dependency("DELETE", "/servers/{server_path:path}/egress-pat")


@pytest.mark.unit
class TestConfigurePatMode:
    """Operator config branch (POST /servers/{path}/egress-auth mode=pat)."""

    def test_configure_pat_ok(self, client, monkeypatch):
        monkeypatch.setattr(routes.server_service, "update_server", AsyncMock(return_value=True))
        c = client(ADMIN, server=_server(egress_auth_mode="none", egress_oauth=None))
        r = c.post(
            "/api/servers/github/egress-auth",
            json={"egress_auth_mode": "pat", "egress_provider": "github"},
        )
        assert r.status_code == 200
        assert r.json()["egress_auth_mode"] == "pat"
        assert r.json()["egress_provider"] == "github"

    def test_configure_pat_non_admin_403(self, client):
        c = client(USER, server=_server())
        r = c.post(
            "/api/servers/github/egress-auth",
            json={"egress_auth_mode": "pat", "egress_provider": "github"},
        )
        assert r.status_code == 403

    @pytest.mark.parametrize("provider", ["", "Bad Provider!", "x" * 65, "UPPER"])
    def test_configure_pat_bad_provider_slug_400(self, client, provider):
        c = client(ADMIN, server=_server())
        r = c.post(
            "/api/servers/github/egress-auth",
            json={"egress_auth_mode": "pat", "egress_provider": provider},
        )
        assert r.status_code == 400

    def test_configure_pat_does_not_store_inject_header(self, client, monkeypatch):
        # The inject header is inherited from Backend Auth at vend time, so pat
        # config stores only the provider (no header fields on the stored config).
        monkeypatch.setattr(routes.server_service, "update_server", AsyncMock(return_value=True))
        c = client(ADMIN, server=_server(egress_auth_mode="none", egress_oauth=None))
        r = c.post(
            "/api/servers/github/egress-auth",
            json={"egress_auth_mode": "pat", "egress_provider": "github"},
        )
        assert r.status_code == 200
        body = r.json()
        assert "pat_header_name" not in body
        assert "pat_value_prefix" not in body
