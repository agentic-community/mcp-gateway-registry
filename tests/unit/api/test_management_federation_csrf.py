"""CSRF protection wiring tests for management and federation routers.

Every state-changing endpoint in ``registry/api/management_routes.py`` and
``registry/api/federation_routes.py`` must enforce CSRF via the
``verify_csrf_token_flexible`` dependency. These tests drive each mutating
route through the real router (via ``TestClient``) and assert that:

- a browser-style request (session cookie present) WITHOUT a CSRF token is
  rejected with 403 and the CSRF-specific detail, and
- the same request WITH a valid CSRF token passes the CSRF gate (the response
  is no longer a CSRF rejection).

The valid-token case only asserts the CSRF gate was cleared, not the full
business outcome: downstream auth/repo behavior varies per endpoint and is
covered by the dedicated authz/behavior suites. What matters here is that the
CSRF dependency is actually wired in and fails closed when the token is absent.
"""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from registry.api.federation_routes import _get_federation_repo
from registry.auth.csrf import generate_csrf_token
from registry.auth.dependencies import nginx_proxied_auth
from registry.core.config import settings
from registry.main import app

# Fixed opaque session id the cookie resolves to for the whole module.
_SESSION_ID = "csrf-test-session-id"


def _admin_ctx() -> dict:
    """Admin user context so authorization never short-circuits before CSRF.

    CSRF runs as a sibling dependency to auth, so authorization outcome does
    not gate the CSRF check; using an admin context keeps the valid-token
    assertion focused on "CSRF gate cleared" rather than a 403 from authz.
    """
    return {
        "username": "admin",
        "groups": ["mcp-registry-admin"],
        "scopes": ["mcp-registry-admin"],
        "is_admin": True,
        "auth_method": "session",
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
def _repo():
    """Federation repo whose methods are AsyncMocks (used by federation routes)."""
    repo = AsyncMock()
    repo.get_config = AsyncMock(return_value={"id": "default"})
    repo.save_config = AsyncMock(return_value={"id": "default"})
    repo.update_config = AsyncMock(return_value={"id": "default"})
    repo.delete_config = AsyncMock(return_value=True)
    return repo


@pytest.fixture
def _client(_repo):
    """TestClient with admin auth and federation repo overridden.

    A session cookie under the real cookie name is set so the CSRF dependency
    treats the request as browser-originated.
    """
    app.dependency_overrides[nginx_proxied_auth] = _admin_ctx
    app.dependency_overrides[_get_federation_repo] = lambda: _repo
    cookie_name = settings.session_cookie_name
    # raise_server_exceptions=False so a downstream 500 (from the deliberately
    # thin repo/IAM mocks) surfaces as a response instead of propagating. The
    # CSRF gate runs before any handler body, so a non-CSRF error still proves
    # the CSRF dependency was cleared.
    client = TestClient(
        app,
        cookies={cookie_name: "browser-session-cookie"},
        raise_server_exceptions=False,
    )
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
    """Assert the CSRF gate was passed (not a CSRF rejection)."""
    if resp.status_code == 403:
        detail = resp.json().get("detail", "")
        assert "CSRF" not in detail, f"unexpected CSRF rejection with valid token: {detail}"


# Each entry: (http_method, url, json_body_or_None).
# Covers every mutating endpoint in management_routes.py and federation_routes.py.
_MANAGEMENT_ENDPOINTS = [
    ("post", "/api/management/iam/users/m2m", {"name": "svc", "groups": [], "description": "d"}),
    (
        "post",
        "/api/management/iam/users/human",
        {"username": "u", "email": "u@example.com", "groups": []},
    ),
    ("delete", "/api/management/iam/users/someuser", None),
    ("patch", "/api/management/iam/users/someuser/groups", {"groups": []}),
    ("post", "/api/management/iam/groups", {"name": "g"}),
    ("delete", "/api/management/iam/groups/somegroup", None),
    ("patch", "/api/management/iam/groups/somegroup", {}),
]

_FEDERATION_ENDPOINTS = [
    ("post", "/api/federation/config", {"id": "default"}),
    ("put", "/api/federation/config/default", {"id": "default"}),
    ("delete", "/api/federation/config/default", None),
    ("post", "/api/federation/config/default/anthropic/servers?server_name=s", None),
    ("delete", "/api/federation/config/default/anthropic/servers/s", None),
    ("post", "/api/federation/config/default/asor/agents?agent_id=a", None),
    ("delete", "/api/federation/config/default/asor/agents/a", None),
    (
        "post",
        "/api/federation/config/default/aws_registry/registries",
        {"registry_arn": "arn:aws:x", "region": "us-east-1"},
    ),
    ("delete", "/api/federation/config/default/aws_registry/registries/r", None),
    ("post", "/api/federation/sync", None),
    (
        "post",
        "/api/federation/config/default/ai_catalog/sources",
        {"source_id": "s", "url": "https://example.com/ai-catalog.json"},
    ),
    ("delete", "/api/federation/config/default/ai_catalog/sources/s", None),
    ("post", "/api/federation/ai_catalog/sync", None),
]

_ALL_ENDPOINTS = _MANAGEMENT_ENDPOINTS + _FEDERATION_ENDPOINTS


def _call(client: TestClient, method: str, url: str, body, headers=None):
    """Invoke the given HTTP method on the client."""
    kwargs = {"headers": headers} if headers else {}
    if body is not None:
        kwargs["json"] = body
    return getattr(client, method)(url, **kwargs)


@pytest.mark.unit
@pytest.mark.api
class TestMutatingEndpointsRequireCsrf:
    """Every mutating management/federation endpoint fails closed without CSRF."""

    @pytest.mark.parametrize("method,url,body", _ALL_ENDPOINTS)
    def test_rejects_without_csrf_token(self, _client, method, url, body):
        """Session-cookie request with no CSRF token is rejected with 403."""
        resp = _call(_client, method, url, body)
        _assert_csrf_rejected(resp)

    @pytest.mark.parametrize("method,url,body", _ALL_ENDPOINTS)
    def test_accepts_with_valid_csrf_token(self, _client, method, url, body):
        """Session-cookie request with a valid CSRF token clears the CSRF gate."""
        resp = _call(_client, method, url, body, headers=_valid_token_header())
        _assert_csrf_cleared(resp)


@pytest.mark.unit
@pytest.mark.api
class TestReadEndpointsUnaffectedByCsrf:
    """Read-only GET endpoints must not be broken by the CSRF additions."""

    @pytest.mark.parametrize(
        "url",
        [
            "/api/management/iam/users",
            "/api/management/iam/groups",
            "/api/federation/config",
            "/api/federation/configs",
        ],
    )
    def test_get_not_csrf_rejected(self, _client, url):
        """A GET with a session cookie and no CSRF token is not CSRF-rejected."""
        resp = _client.get(url)
        _assert_csrf_cleared(resp)
