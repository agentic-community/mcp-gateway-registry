"""Unit tests for the generic-hop /validate wiring.

Covers the token-attach discriminator and the cookie-auth helper in isolation
(the full validate_request round-trip is exercised by the integration suite):

- _attach_generic_proxy_token mints ONLY when X-Resolved-Generic-Upstream is set
  (the SEPARATE marker), and binds entity_type + full registered path + the
  server-set upstream — never a forgeable inbound header;
- it is disjoint from _attach_mcp_proxy_token: a generic request (no
  X-Resolved-Upstream) mints no MCP token, and vice-versa — exactly one token;
- _is_cookie_auth_method classifies the CSRF-relevant (ambient) auth methods.
"""

import os
from unittest.mock import patch

import jwt as pyjwt
import pytest

os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-definitely-long-enough-32b")

from internal_request_token import GENERIC_PROXY_AUDIENCE  # noqa: E402

from auth_server.server import (  # noqa: E402
    _attach_generic_proxy_token,
    _attach_mcp_proxy_token,
    _generic_write_csrf_refused,
    _is_cookie_auth_method,
    settings,
)

pytestmark = pytest.mark.unit

_SECRET = "test-secret-key-that-is-definitely-long-enough-32b"


class _Req:
    def __init__(self, headers):
        self.headers = headers


class _Resp:
    def __init__(self):
        self.headers = {}


class TestAttachGenericProxyToken:
    def test_mints_when_generic_upstream_and_nginx_marker_present(self):
        req = _Req(
            {
                "X-Resolved-Generic-Upstream": "https://backend.example/",
                "X-Validate-Source-Secret": settings.auth_server_nginx_marker_secret,
            }
        )
        resp = _Resp()
        with patch.dict(os.environ, {"SECRET_KEY": _SECRET}, clear=False):
            _attach_generic_proxy_token(
                req,
                resp,
                subject="alice",
                scopes=["s/read"],
                entity_type="skill",
                registered_path="skills/proxy-demo",
                http_method="GET",
            )
        tok = resp.headers["X-Internal-Token-Generic"]
        claims = pyjwt.decode(tok, _SECRET, algorithms=["HS256"], audience=GENERIC_PROXY_AUDIENCE)
        assert claims["entity_type"] == "skill"
        assert claims["server"] == "skills/proxy-demo"  # FULL registered path
        assert claims["upstream_url"] == "https://backend.example/"  # server-set marker
        assert claims["method"] == "GET"
        assert claims["sub"] == "alice"

    def test_no_mint_when_resolved_upstream_absent(self):
        req = _Req({})  # not a generic request
        resp = _Resp()
        with patch.dict(os.environ, {"SECRET_KEY": _SECRET}, clear=False):
            _attach_generic_proxy_token(
                req,
                resp,
                subject="alice",
                scopes=[],
                entity_type="skill",
                registered_path="x",
                http_method="GET",
            )
        assert "X-Internal-Token-Generic" not in resp.headers

    @pytest.mark.parametrize("marker", [None, "wrong-marker-secret"])
    def test_no_mint_when_nginx_marker_missing_or_mismatched(self, marker):
        headers = {"X-Resolved-Generic-Upstream": "https://backend.example/"}
        if marker is not None:
            headers["X-Validate-Source-Secret"] = marker
        req = _Req(headers)
        resp = _Resp()
        with patch.dict(os.environ, {"SECRET_KEY": _SECRET}, clear=False):
            _attach_generic_proxy_token(
                req,
                resp,
                subject="alice",
                scopes=[],
                entity_type="skill",
                registered_path="skills/x",
                http_method="GET",
            )
        assert "X-Internal-Token-Generic" not in resp.headers

    def test_empty_subject_mints_nothing(self):
        req = _Req(
            {
                "X-Resolved-Generic-Upstream": "https://b/",
                "X-Validate-Source-Secret": settings.auth_server_nginx_marker_secret,
            }
        )
        resp = _Resp()
        with patch.dict(os.environ, {"SECRET_KEY": _SECRET}, clear=False):
            _attach_generic_proxy_token(
                req,
                resp,
                subject="",
                scopes=[],
                entity_type="skill",
                registered_path="x",
                http_method="GET",
            )
        # fail-closed: no token attached, generic hop rejects downstream
        assert "X-Internal-Token-Generic" not in resp.headers


class TestDisjointFromMcpMint:
    def test_generic_request_does_not_mint_mcp_token(self):
        # A generic request sets X-Resolved-Generic-Upstream but NOT
        # X-Resolved-Upstream, so the MCP attach must short-circuit. The shared
        # nginx marker permits only the generic mint.
        req = _Req(
            {
                "X-Resolved-Generic-Upstream": "https://b/",
                "X-Validate-Source-Secret": settings.auth_server_nginx_marker_secret,
            }
        )
        resp = _Resp()
        with patch.dict(os.environ, {"SECRET_KEY": _SECRET}, clear=False):
            _attach_mcp_proxy_token(req, resp, subject="alice", scopes=[], server_name="skill/x")
            _attach_generic_proxy_token(
                req,
                resp,
                subject="alice",
                scopes=[],
                entity_type="skill",
                registered_path="skills/x",
                http_method="GET",
            )
        assert "X-Internal-Token" not in resp.headers  # MCP mint did not fire
        assert "X-Internal-Token-Generic" in resp.headers  # generic mint did

    def test_mcp_request_does_not_mint_generic_token(self):
        req = _Req(
            {
                "X-Resolved-Upstream": "https://mcp-backend/",
                "X-Validate-Source-Secret": settings.auth_server_nginx_marker_secret,
            }
        )
        resp = _Resp()
        with patch.dict(os.environ, {"SECRET_KEY": _SECRET}, clear=False):
            _attach_mcp_proxy_token(req, resp, subject="alice", scopes=[], server_name="foo/mcp")
            _attach_generic_proxy_token(
                req,
                resp,
                subject="alice",
                scopes=[],
                entity_type="skill",
                registered_path="x",
                http_method="GET",
            )
        assert "X-Internal-Token" in resp.headers  # MCP mint fired
        assert "X-Internal-Token-Generic" not in resp.headers  # generic mint did not


class TestCookieAuthClassifier:
    def test_session_cookie_is_cookie_auth(self):
        assert _is_cookie_auth_method("session_cookie") is True
        assert _is_cookie_auth_method("cookie") is True
        assert _is_cookie_auth_method("SESSION") is True

    def test_bearer_methods_are_not_cookie_auth(self):
        for m in ("keycloak", "cognito", "entra", "okta", "auth0", "self_signed", "bearer", ""):
            assert _is_cookie_auth_method(m) is False


class TestGenericWriteCsrfRefused:
    def test_cookie_delete_refused(self):
        # The core CSRF case: ambient cookie + state-changing verb on generic hop.
        assert (
            _generic_write_csrf_refused(
                is_generic_request=True,
                http_verb="DELETE",
                auth_method="session_cookie",
                require_bearer=True,
            )
            is True
        )

    def test_cookie_get_allowed(self):
        # Safe verb under cookie auth is fine (read).
        assert (
            _generic_write_csrf_refused(
                is_generic_request=True,
                http_verb="GET",
                auth_method="session_cookie",
                require_bearer=True,
            )
            is False
        )

    def test_bearer_delete_allowed(self):
        # Programmatic Bearer caller is exempt from the CSRF refusal.
        assert (
            _generic_write_csrf_refused(
                is_generic_request=True,
                http_verb="DELETE",
                auth_method="keycloak",
                require_bearer=True,
            )
            is False
        )

    def test_disabled_knob_never_refuses(self):
        # Operator opt-out (same-site deploy): cookie DELETE passes.
        assert (
            _generic_write_csrf_refused(
                is_generic_request=True,
                http_verb="DELETE",
                auth_method="session_cookie",
                require_bearer=False,
            )
            is False
        )

    def test_non_generic_request_never_refused(self):
        # The gate only applies to the generic hop; MCP/api requests are untouched.
        assert (
            _generic_write_csrf_refused(
                is_generic_request=False,
                http_verb="DELETE",
                auth_method="session_cookie",
                require_bearer=True,
            )
            is False
        )

    def test_options_is_safe(self):
        assert (
            _generic_write_csrf_refused(
                is_generic_request=True,
                http_verb="OPTIONS",
                auth_method="session_cookie",
                require_bearer=True,
            )
            is False
        )
