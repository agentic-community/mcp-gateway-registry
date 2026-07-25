"""CSRF wiring tests for routers fixed after the initial CSRF sweep.

The rate-limit admin router (``registry/api/rate_limit_routes.py``) and the
data-export audit endpoint (``registry/api/export_routes.py``) are
session-cookie (``nginx_proxied_auth``) mutation surfaces that were missing the
canonical ``verify_csrf_token_flexible`` dependency. These tests assert the same
three-way contract as ``test_server_routes_csrf.py`` for each fixed endpoint:

- a browser request (session cookie present) WITHOUT a CSRF token is rejected
  with 403 and the CSRF-specific detail (fail closed),
- the same request WITH a valid ``X-CSRF-Token`` header passes the CSRF gate,
- a non-browser (no-cookie) request is NOT CSRF-rejected.

The valid-token / no-cookie cases only assert the CSRF gate outcome; downstream
validation (e.g. a 400 id-mismatch or a 404) is expected and does not affect the
CSRF assertion.
"""

import pytest
from fastapi.testclient import TestClient

from registry.auth.csrf import generate_csrf_token
from registry.auth.dependencies import enhanced_auth, nginx_proxied_auth
from registry.core.config import settings
from registry.main import app

_SESSION_ID = "csrf-followup-session-id"


def _admin_ctx() -> dict:
    """Admin user context so authorization never short-circuits before CSRF."""
    return {
        "username": "admin",
        "groups": ["mcp-registry-admin"],
        "scopes": ["mcp-registry-admin"],
        "is_admin": True,
        "auth_method": "session",
    }


@pytest.fixture(autouse=True)
def _resolve_session(monkeypatch):
    """Resolve any non-empty session cookie to a fixed session_id."""

    async def _fake_resolve(cookie_value: str):
        return {"session_id": _SESSION_ID, "username": "admin"}

    monkeypatch.setattr("registry.auth.csrf.resolve_session_from_cookie", _fake_resolve)


@pytest.fixture
def _client():
    """TestClient with admin auth overrides and a browser session cookie."""
    app.dependency_overrides[enhanced_auth] = _admin_ctx
    app.dependency_overrides[nginx_proxied_auth] = _admin_ctx
    cookie_name = settings.session_cookie_name
    client = TestClient(
        app,
        cookies={cookie_name: "browser-session-cookie"},
        raise_server_exceptions=False,
    )
    try:
        yield client
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def _no_cookie_client():
    """TestClient with admin auth overrides but NO session cookie."""
    app.dependency_overrides[enhanced_auth] = _admin_ctx
    app.dependency_overrides[nginx_proxied_auth] = _admin_ctx
    client = TestClient(app, raise_server_exceptions=False)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()


def _valid_token_header() -> dict[str, str]:
    return {"X-CSRF-Token": generate_csrf_token(_SESSION_ID)}


def _assert_csrf_rejected(resp) -> None:
    assert resp.status_code == 403, f"expected 403, got {resp.status_code}: {resp.text}"
    detail = resp.json().get("detail", "")
    assert "CSRF" in detail, f"expected CSRF rejection, got: {detail}"


def _assert_csrf_cleared(resp) -> None:
    if resp.status_code == 403:
        detail = resp.json().get("detail", "")
        assert "CSRF" not in detail, f"unexpected CSRF rejection: {detail}"


# Each entry: (label, http_method, url, kind, payload).
_ENDPOINTS = [
    # Rate-limit admin router (nginx_proxied_auth mutations).
    (
        "rate_limits_enabled",
        "post",
        "/api/rate-limits-enabled/test-id?enabled=false",
        None,
        None,
    ),
    ("rate_limits_put", "put", "/api/rate-limits/test-id", "json", {"foo": "bar"}),
    ("rate_limits_delete", "delete", "/api/rate-limits/test-id", None, None),
    (
        "rate_limit_memberships_put",
        "put",
        "/api/rate-limit-memberships/test-id",
        "json",
        {"foo": "bar"},
    ),
    (
        "rate_limit_memberships_delete",
        "delete",
        "/api/rate-limit-memberships/test-id",
        None,
        None,
    ),
    # Data-export audit event (nginx_proxied_auth via _require_admin).
    (
        "export_audit_event",
        "post",
        "/api/export/audit-event",
        "json",
        {"export_type": "single", "collections": ["servers"]},
    ),
]


def _call(client: TestClient, method: str, url: str, kind, payload, headers=None):
    kwargs = {"headers": headers} if headers else {}
    if kind == "json" and payload is not None:
        kwargs["json"] = payload
    elif kind == "form" and payload is not None:
        kwargs["data"] = payload
    return getattr(client, method)(url, **kwargs)


@pytest.mark.unit
@pytest.mark.api
class TestFollowupMutationsRequireCsrf:
    """Every fixed mutating session-auth endpoint fails closed without CSRF."""

    @pytest.mark.parametrize("label,method,url,kind,payload", _ENDPOINTS)
    def test_rejects_without_csrf_token(self, _client, label, method, url, kind, payload):
        """Session-cookie request with no CSRF token is rejected with 403."""
        resp = _call(_client, method, url, kind, payload)
        _assert_csrf_rejected(resp)

    @pytest.mark.parametrize("label,method,url,kind,payload", _ENDPOINTS)
    def test_accepts_with_valid_csrf_header(self, _client, label, method, url, kind, payload):
        """A valid CSRF token in the header clears the CSRF gate."""
        resp = _call(_client, method, url, kind, payload, headers=_valid_token_header())
        _assert_csrf_cleared(resp)

    @pytest.mark.parametrize("label,method,url,kind,payload", _ENDPOINTS)
    def test_no_cookie_request_skips_csrf(
        self, _no_cookie_client, label, method, url, kind, payload
    ):
        """A non-browser (no-cookie) request is not CSRF-rejected."""
        resp = _call(_no_cookie_client, method, url, kind, payload)
        _assert_csrf_cleared(resp)
