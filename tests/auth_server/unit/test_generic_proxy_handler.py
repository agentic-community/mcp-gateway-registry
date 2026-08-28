"""Unit tests for the generic-proxy handler helpers.

Covers the security-critical pure helpers in isolation (the full handler
round-trip needs a live upstream; these lock the guards):

- _build_generic_outbound_url: sub-path confinement (reject '..'/scheme/userinfo,
  stay under the bound registered prefix);
- _assert_outbound_host_pinned: post-join scheme/host/port equality with the
  pinned upstream (the real SSRF-escape backstop);
- _select_forwarded_generic_response_headers: the WIDER allowlist forwards
  Location/caching/download headers but DROPS Set-Cookie/HSTS/CSP/framing;
- _generic_tls_verify: true|false|path resolution;
- _run_egress_selfcheck: reachable metadata IP => unsafe (feature must disable).
"""

import os
from unittest.mock import patch

import pytest

os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-definitely-long-enough-32b")

from fastapi import HTTPException  # noqa: E402

from auth_server.server import (  # noqa: E402
    _GATEWAY_SET_SECURITY_HEADERS,
    _GENERIC_HOP_STRIP_HEADERS,
    _assert_outbound_host_pinned,
    _build_generic_outbound_url,
    _forward_headers,
    _generic_tls_verify,
    _run_egress_selfcheck,
    _select_forwarded_generic_response_headers,
)

pytestmark = pytest.mark.unit


class TestGenericHopHeaderStrip:
    """The generic hop fronts arbitrary (third-party) backends, so no gateway
    identity/credential/routing header may leak to the upstream. Replicates the
    handler's filter: _forward_headers(...) minus _GENERIC_HOP_STRIP_HEADERS.
    """

    def _forwarded(self, incoming: dict[str, str]) -> dict[str, str]:
        return {
            key: value
            for key, value in _forward_headers(dict(incoming)).items()
            if key.lower() not in _GENERIC_HOP_STRIP_HEADERS
        }

    def test_strips_internal_identity_token_and_markers(self):
        incoming = {
            "X-Internal-Token-Generic": "signed.jwt.value",
            "X-User": "alice",
            "X-Scopes": "read write",
            "X-Groups": "admins",
            "X-Original-URL": "https://gw/skill/skills/x",
            "X-Generic-Proxy-Kind": "skill",
            "X-Entity-Path": "skills/x",
            "Authorization": "Bearer caller-token",
            "Cookie": "mcp_gateway_session=abc",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        forwarded_lower = {k.lower() for k in self._forwarded(incoming)}
        # A proxied backend must never receive the signed internal token, the
        # caller's identity/scopes, ambient credentials, or the routing markers.
        for leaked in (
            "x-internal-token-generic",
            "x-user",
            "x-scopes",
            "x-groups",
            "x-original-url",
            "x-generic-proxy-kind",
            "x-entity-path",
            "authorization",
            "cookie",
        ):
            assert leaked not in forwarded_lower

    def test_keeps_legitimate_client_headers(self):
        incoming = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Internal-Token-Generic": "signed.jwt.value",
        }
        forwarded = self._forwarded(incoming)
        assert forwarded.get("Content-Type") == "application/json"
        assert forwarded.get("Accept") == "application/json"


class TestBuildOutboundUrl:
    def test_no_subpath_returns_base(self):
        url = _build_generic_outbound_url(
            "https://backend.example/", "skills/proxy-demo", "skills/proxy-demo"
        )
        assert url == "https://backend.example/"

    def test_subpath_appended_to_base(self):
        url = _build_generic_outbound_url(
            "https://backend.example/api", "skills/proxy-demo/reports/2024", "skills/proxy-demo"
        )
        assert url == "https://backend.example/api/reports/2024"

    def test_dotdot_segment_rejected(self):
        with pytest.raises(HTTPException) as e:
            _build_generic_outbound_url(
                "https://b/", "skills/proxy-demo/../../etc", "skills/proxy-demo"
            )
        assert e.value.status_code == 400

    def test_scheme_in_subpath_rejected(self):
        with pytest.raises(HTTPException) as e:
            _build_generic_outbound_url(
                "https://b/", "skills/proxy-demo/https://evil.com", "skills/proxy-demo"
            )
        assert e.value.status_code == 400

    def test_userinfo_in_subpath_rejected(self):
        with pytest.raises(HTTPException) as e:
            _build_generic_outbound_url(
                "https://b/", "skills/proxy-demo/x@evil.com", "skills/proxy-demo"
            )
        assert e.value.status_code == 400

    def test_route_outside_bound_prefix_rejected(self):
        # verify_generic_proxy_token should have caught this, but the handler
        # fails closed rather than appending an unconfined remainder.
        with pytest.raises(HTTPException) as e:
            _build_generic_outbound_url("https://b/", "skills/other", "skills/proxy-demo")
        assert e.value.status_code == 400


class TestAssertHostPinned:
    def test_same_host_passes(self):
        _assert_outbound_host_pinned("https://b.example/api/x", "https://b.example/api")

    def test_different_host_rejected(self):
        with pytest.raises(HTTPException) as e:
            _assert_outbound_host_pinned("https://evil.example/", "https://b.example/")
        assert e.value.status_code == 400

    def test_different_scheme_rejected(self):
        with pytest.raises(HTTPException) as e:
            _assert_outbound_host_pinned("http://b.example/", "https://b.example/")
        assert e.value.status_code == 400

    def test_different_port_rejected(self):
        with pytest.raises(HTTPException) as e:
            _assert_outbound_host_pinned("https://b.example:9000/", "https://b.example:443/")
        assert e.value.status_code == 400


class TestResponseHeaderAllowlist:
    def test_forwards_location_and_caching_and_download(self):
        upstream = {
            "Location": "https://b/next",
            "Content-Type": "application/pdf",
            "Content-Disposition": 'attachment; filename="r.pdf"',
            "Content-Length": "1024",
            "Cache-Control": "max-age=60",
            "ETag": '"abc"',
            "Accept-Ranges": "bytes",
        }
        out = _select_forwarded_generic_response_headers(upstream)
        assert out["Location"] == "https://b/next"
        assert out["Content-Disposition"] == 'attachment; filename="r.pdf"'
        assert out["Accept-Ranges"] == "bytes"
        assert "Cache-Control" in out

    def test_drops_set_cookie_and_security_policy_headers(self):
        upstream = {
            "Set-Cookie": "session=evil",
            "Strict-Transport-Security": "max-age=99999",
            "Content-Security-Policy": "default-src *",
            "X-Frame-Options": "ALLOWALL",
            "Content-Type": "text/html",
        }
        out = _select_forwarded_generic_response_headers(upstream)
        assert "Set-Cookie" not in out
        assert "Strict-Transport-Security" not in out
        # A backend must not dictate CSP/framing; only Content-Type survives.
        assert "Content-Security-Policy" not in out
        assert "X-Frame-Options" not in out
        assert out["Content-Type"] == "text/html"

    def test_gateway_sets_its_own_security_headers(self):
        # The handler updates the response with these; assert the constant is
        # the restrictive set (locked in so a relaxation is a visible diff).
        assert _GATEWAY_SET_SECURITY_HEADERS["X-Content-Type-Options"] == "nosniff"
        assert _GATEWAY_SET_SECURITY_HEADERS["X-Frame-Options"] == "DENY"
        assert "frame-ancestors 'none'" in _GATEWAY_SET_SECURITY_HEADERS["Content-Security-Policy"]


class TestTlsVerifyResolution:
    def test_true(self):
        with patch("auth_server.server.settings") as s:
            s.gateway_generic_tls_verify = "true"
            assert _generic_tls_verify() is True

    def test_false(self):
        with patch("auth_server.server.settings") as s:
            s.gateway_generic_tls_verify = "false"
            assert _generic_tls_verify() is False

    def test_ca_bundle_path(self):
        with patch("auth_server.server.settings") as s:
            s.gateway_generic_tls_verify = "/etc/ssl/private-ca.pem"
            assert _generic_tls_verify() == "/etc/ssl/private-ca.pem"


class TestEgressSelfCheck:
    async def test_metadata_reachable_is_unsafe(self):
        # Both probes "connect" -> reachable -> egress NOT restricted -> unsafe.
        async def _fake_open(host, port):
            class _W:
                def close(self):
                    pass

                async def wait_closed(self):
                    pass

            return (None, _W())

        with patch("auth_server.server.asyncio.open_connection", side_effect=_fake_open):
            assert await _run_egress_selfcheck() is False

    async def test_metadata_unreachable_is_safe(self):
        async def _fake_open(host, port):
            raise OSError("connection refused")

        with patch("auth_server.server.asyncio.open_connection", side_effect=_fake_open):
            assert await _run_egress_selfcheck() is True

    async def test_timeout_is_safe(self):
        async def _fake_open(host, port):
            raise TimeoutError()

        with patch("auth_server.server.asyncio.open_connection", side_effect=_fake_open):
            assert await _run_egress_selfcheck() is True
