"""End-to-end tests for the generic_proxy endpoint.

Drives the /proxy/{entity_type}/{entity_path} route via TestClient with the
token dependency overridden (verified separately in test_generic_proxy_token)
and httpx mocked, to lock in the runtime behavior:

- feature latch: fail-closed 503 when the process latch is off (flag disabled or
  egress self-check failed) even with a valid token;
- redirect NOT auto-followed: a 302 upstream is returned verbatim with Location
  forwarded and produces no second outbound (follow_redirects=False);
- Set-Cookie from the backend is dropped; gateway security headers are set;
- the pinned upstream is used (inbound X-Upstream-Url ignored).
"""

import os
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest

os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-definitely-long-enough-32b")

from fastapi import Request  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from internal_request_token import verify_generic_proxy_token  # noqa: E402

import auth_server.server as server_module  # noqa: E402
from auth_server.server import app  # noqa: E402

pytestmark = pytest.mark.unit


def _override_claims(upstream="https://backend.example/", server="skills/proxy-demo"):
    """Return a dependency override that stashes generic_proxy_claims + passes."""

    async def _dep(request: Request):
        request.state.generic_proxy_claims = {
            "upstream_url": upstream,
            "server": server,
            "entity_type": "skill",
            "scopes": ["s/read"],
            "sub": "alice",
        }

    return _dep


class _FakeStreamResponse:
    def __init__(self, status_code, headers, body=b""):
        self.status_code = status_code
        self.headers = headers
        self._body = body

    async def aiter_bytes(self, chunk_size=65536):
        yield self._body


def _fake_client(captured, response: _FakeStreamResponse):
    """Build a MagicMock httpx.AsyncClient whose .stream() records the call."""

    class _Client:
        def __init__(self, *a, **kw):
            captured["client_kwargs"] = kw

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def stream(self, method, url, **kw):
            captured["method"] = method
            captured["url"] = url

            @asynccontextmanager
            async def _cm():
                yield response

            return _cm()

    return _Client


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


class TestFeatureLatchFailClosed:
    def test_503_when_feature_latch_off(self):
        app.dependency_overrides[verify_generic_proxy_token] = _override_claims()
        with patch.object(server_module, "_generic_proxy_feature_active", False):
            client = TestClient(app)
            resp = client.get("/proxy/skill/skills/proxy-demo")
        assert resp.status_code == 503

    def test_503_when_latch_none_prestartup(self):
        app.dependency_overrides[verify_generic_proxy_token] = _override_claims()
        with patch.object(server_module, "_generic_proxy_feature_active", None):
            client = TestClient(app)
            resp = client.get("/proxy/skill/skills/proxy-demo")
        assert resp.status_code == 503


class TestHappyPathAndRedirect:
    def test_forwards_to_pinned_upstream_and_sets_security_headers(self):
        app.dependency_overrides[verify_generic_proxy_token] = _override_claims(
            upstream="https://backend.example/"
        )
        captured = {}
        resp_obj = _FakeStreamResponse(
            200,
            {"Content-Type": "application/json", "Set-Cookie": "evil=1"},
            b'{"ok":true}',
        )
        with (
            patch.object(server_module, "_generic_proxy_feature_active", True),
            patch("auth_server.server.httpx.AsyncClient", _fake_client(captured, resp_obj)),
        ):
            client = TestClient(app)
            resp = client.get(
                "/proxy/skill/skills/proxy-demo",
                headers={"X-Upstream-Url": "https://attacker.example/"},
            )
        assert resp.status_code == 200
        # pinned upstream used, forgeable inbound header ignored
        assert captured["url"] == "https://backend.example/"
        # backend Set-Cookie dropped; gateway security headers set
        assert "set-cookie" not in {k.lower() for k in resp.headers}
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "DENY"

    def test_redirect_not_followed_location_forwarded(self):
        app.dependency_overrides[verify_generic_proxy_token] = _override_claims()
        captured = {}
        resp_obj = _FakeStreamResponse(
            302, {"Location": "http://169.254.169.254/latest/meta-data/"}, b""
        )
        with (
            patch.object(server_module, "_generic_proxy_feature_active", True),
            patch("auth_server.server.httpx.AsyncClient", _fake_client(captured, resp_obj)),
        ):
            client = TestClient(app, follow_redirects=False)
            resp = client.get("/proxy/skill/skills/proxy-demo")
        # 302 returned verbatim; the httpx client was created with follow_redirects=False
        assert resp.status_code == 302
        assert resp.headers["Location"] == "http://169.254.169.254/latest/meta-data/"
        assert captured["client_kwargs"].get("follow_redirects") is False

    def test_subpath_appended_to_pinned_base(self):
        app.dependency_overrides[verify_generic_proxy_token] = _override_claims(
            upstream="https://backend.example/api", server="skills/proxy-demo"
        )
        captured = {}
        resp_obj = _FakeStreamResponse(200, {"Content-Type": "application/json"}, b"{}")
        with (
            patch.object(server_module, "_generic_proxy_feature_active", True),
            patch("auth_server.server.httpx.AsyncClient", _fake_client(captured, resp_obj)),
        ):
            client = TestClient(app)
            resp = client.get("/proxy/skill/skills/proxy-demo/reports/2024")
        assert resp.status_code == 200
        assert captured["url"] == "https://backend.example/api/reports/2024"
