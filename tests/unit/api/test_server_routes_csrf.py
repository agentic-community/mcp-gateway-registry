"""CSRF protection wiring tests for the server-management router.

Every state-changing session-authenticated endpoint in
``registry/api/server_routes.py`` must enforce CSRF via the
``verify_csrf_token_flexible`` dependency. These tests drive each mutating
route through the real router (via ``TestClient``) and assert three things:

- a browser-style request (session cookie present) WITHOUT a CSRF token is
  rejected with 403 and the CSRF-specific detail (fail closed),
- the same request WITH a valid CSRF token in the ``X-CSRF-Token`` header
  passes the CSRF gate (the response is no longer a CSRF rejection), and
- a non-browser request (no session cookie, e.g. a Bearer/CLI client) is NOT
  CSRF-rejected -- the flexible dependency correctly no-ops when there is no
  cookie to be abused.

The valid-token / no-cookie cases only assert the CSRF gate outcome, not the
full business result: downstream auth/repo behavior varies per endpoint and is
covered by the dedicated behavior suites. What matters here is that the CSRF
dependency is wired in and fails closed when a cookie is present without a
valid token.

The JSON-body endpoint (``/api/tokens/generate``) is included specifically to
prove that a valid CSRF token supplied via the header (not a form field) clears
the gate for a non-form route.
"""

import pytest
from fastapi.testclient import TestClient

from registry.auth.csrf import generate_csrf_token
from registry.auth.dependencies import enhanced_auth, nginx_proxied_auth
from registry.core.config import settings
from registry.main import app

# Fixed opaque session id the cookie resolves to for the whole module.
_SESSION_ID = "csrf-test-session-id"


def _admin_ctx() -> dict:
    """Admin user context so authorization never short-circuits before CSRF.

    CSRF runs as a sibling dependency to auth, so the authorization outcome
    does not gate the CSRF check; using an admin context keeps the valid-token
    assertion focused on "CSRF gate cleared" rather than a 403 from authz.
    """
    return {
        "username": "admin",
        "groups": ["mcp-registry-admin"],
        "scopes": ["mcp-registry-admin"],
        "is_admin": True,
        "can_modify_servers": True,
        "auth_method": "session",
        "ui_permissions": {
            "register_service": ["all"],
            "health_check_service": ["all"],
            "modify_service": ["all"],
        },
    }


@pytest.fixture(autouse=True)
def _resolve_session(monkeypatch):
    """Resolve any non-empty session cookie to a fixed session_id.

    This makes ``verify_csrf_token_flexible`` treat the request as a browser
    session (cookie present AND resolvable), so CSRF enforcement engages
    instead of the non-browser bypass.
    """

    async def _fake_resolve(cookie_value: str):
        return {"session_id": _SESSION_ID, "username": "admin"}

    monkeypatch.setattr("registry.auth.csrf.resolve_session_from_cookie", _fake_resolve)


@pytest.fixture
def _client():
    """TestClient with admin auth overrides and a browser session cookie.

    raise_server_exceptions=False so a downstream 500 (from the deliberately
    thin mocks / missing form fields) surfaces as a response instead of
    propagating. The CSRF gate runs before any handler body, so a non-CSRF
    error still proves the CSRF dependency was cleared.
    """
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
    """TestClient with admin auth overrides but NO session cookie.

    Represents a non-browser (Bearer/CLI) caller: verify_csrf_token_flexible
    must skip CSRF entirely because there is no cookie to abuse.
    """
    app.dependency_overrides[enhanced_auth] = _admin_ctx
    app.dependency_overrides[nginx_proxied_auth] = _admin_ctx
    client = TestClient(app, raise_server_exceptions=False)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()


def _valid_token_header() -> dict[str, str]:
    """Header carrying a CSRF token valid for the resolved session id."""
    return {"X-CSRF-Token": generate_csrf_token(_SESSION_ID)}


def _assert_csrf_rejected(resp) -> None:
    """Assert the response is a CSRF rejection (403 with CSRF detail)."""
    assert resp.status_code == 403, f"expected 403, got {resp.status_code}: {resp.text}"
    detail = resp.json().get("detail", "")
    assert "CSRF" in detail, f"expected CSRF rejection, got: {detail}"


def _assert_csrf_cleared(resp) -> None:
    """Assert the CSRF gate was passed (response is not a CSRF rejection)."""
    if resp.status_code == 403:
        detail = resp.json().get("detail", "")
        assert "CSRF" not in detail, f"unexpected CSRF rejection: {detail}"


# Each entry: (label, http_method, url, kind, payload).
# kind is "json" or "form" (or None for a bodyless request).
_ENDPOINTS = [
    # The three explicitly reported gaps first.
    (
        "register_form",
        "post",
        "/api/register",
        "form",
        {"name": "x", "description": "d", "path": "/x"},
    ),
    ("refresh", "post", "/api/refresh/some-service", None, None),
    ("tokens_generate_json", "post", "/api/tokens/generate", "json", {"scopes": []}),
    # A representative sample of the sibling /servers/* mutation family.
    ("servers_toggle", "post", "/api/servers/toggle", "json", {"path": "/x", "enabled": True}),
    ("servers_remove", "post", "/api/servers/remove", "json", {"path": "/x"}),
    ("servers_groups_add", "post", "/api/servers/groups/add", "json", {"path": "/x", "groups": []}),
    ("servers_rate", "post", "/api/servers/x/rate", "json", {"rating": 5}),
    ("servers_rescan", "post", "/api/servers/x/rescan", None, None),
]


def _call(client: TestClient, method: str, url: str, kind, payload, headers=None):
    """Invoke the given HTTP method with the appropriate body encoding."""
    kwargs = {"headers": headers} if headers else {}
    if kind == "json" and payload is not None:
        kwargs["json"] = payload
    elif kind == "form" and payload is not None:
        kwargs["data"] = payload
    return getattr(client, method)(url, **kwargs)


@pytest.mark.unit
@pytest.mark.api
class TestServerMutationsRequireCsrf:
    """Every mutating session-auth server endpoint fails closed without CSRF."""

    @pytest.mark.parametrize("label,method,url,kind,payload", _ENDPOINTS)
    def test_rejects_without_csrf_token(self, _client, label, method, url, kind, payload):
        """Session-cookie request with no CSRF token is rejected with 403."""
        resp = _call(_client, method, url, kind, payload)
        _assert_csrf_rejected(resp)

    @pytest.mark.parametrize("label,method,url,kind,payload", _ENDPOINTS)
    def test_accepts_with_valid_csrf_header(self, _client, label, method, url, kind, payload):
        """A valid CSRF token in the header clears the CSRF gate (incl. JSON body)."""
        resp = _call(_client, method, url, kind, payload, headers=_valid_token_header())
        _assert_csrf_cleared(resp)

    @pytest.mark.parametrize("label,method,url,kind,payload", _ENDPOINTS)
    def test_no_cookie_request_skips_csrf(
        self, _no_cookie_client, label, method, url, kind, payload
    ):
        """A non-browser (no-cookie) request is not CSRF-rejected."""
        resp = _call(_no_cookie_client, method, url, kind, payload)
        _assert_csrf_cleared(resp)
