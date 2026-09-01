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
    _assert_generic_authorization_not_gateway_cred,
    _assert_outbound_host_pinned,
    _build_generic_outbound_url,
    _effective_ingress_gateway_credential,
    _generic_tls_verify,
    _merge_generic_upstream_headers,
    _run_egress_selfcheck,
    _select_forwarded_generic_request_headers,
    _select_forwarded_generic_response_headers,
    _strip_generic_internal_headers,
)
from registry.common.log_redaction import redact_url  # noqa: E402

pytestmark = pytest.mark.unit


class TestGenericHopHeaderStrip:
    """The generic hop fronts arbitrary (third-party) backends, so no gateway
    identity/credential/routing header may leak to the upstream. Replicates the
    handler's real egress pipeline: the positive protocol allowlist
    (_select_forwarded_generic_request_headers) followed by the defense-in-depth
    backstop (_strip_generic_internal_headers).
    """

    def _forwarded(self, incoming: dict[str, str]) -> dict[str, str]:
        return _strip_generic_internal_headers(
            _select_forwarded_generic_request_headers(dict(incoming))
        )

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

    def test_registered_query_is_preserved(self):
        url = _build_generic_outbound_url(
            "https://backend.example/api?fixed=a%2Fb&blank=",
            "skills/proxy-demo",
            "skills/proxy-demo",
        )
        assert url == "https://backend.example/api?fixed=a%2Fb&blank="

    def test_subpath_appends_before_registered_query(self):
        url = _build_generic_outbound_url(
            "https://backend.example/api?fixed=registered",
            "skills/proxy-demo/reports/2024",
            "skills/proxy-demo",
        )
        assert url == "https://backend.example/api/reports/2024?fixed=registered"

    def test_caller_query_cannot_conflict_with_registered_fixed_key(self):
        url = _build_generic_outbound_url(
            "https://backend.example/api?fixed=registered&token=operator",
            "skills/proxy-demo",
            "skills/proxy-demo",
            [("fixed", "caller"), ("other", "allowed"), ("token", "caller-token")],
        )
        assert url == ("https://backend.example/api?fixed=registered&token=operator&other=allowed")

    def test_legacy_semicolon_fixed_key_cannot_be_shadowed(self):
        url = _build_generic_outbound_url(
            "https://backend.example/api?fixed=ok;token=operator",
            "skills/proxy-demo",
            "skills/proxy-demo",
            [("token", "attacker"), ("other", "allowed")],
        )
        assert url == "https://backend.example/api?fixed=ok;token=operator&other=allowed"

    def test_encoded_and_case_variant_fixed_keys_cannot_be_shadowed(self):
        url = _build_generic_outbound_url(
            "https://backend.example/api?To%6Ben=operator",
            "skills/proxy-demo",
            "skills/proxy-demo",
            [("TOKEN", "attacker"), ("other", "allowed")],
        )
        assert url == "https://backend.example/api?To%6Ben=operator&other=allowed"

    def test_duplicate_caller_query_values_survive(self):
        url = _build_generic_outbound_url(
            "https://backend.example/api?fixed=registered",
            "skills/proxy-demo",
            "skills/proxy-demo",
            [("tag", "one"), ("tag", "two"), ("empty", "")],
        )
        assert url == ("https://backend.example/api?fixed=registered&tag=one&tag=two&empty=")

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
        # Buffered bodies are decoded by httpx, so stale framing is removed.
        assert "Content-Length" not in out

    def test_buffered_decoded_body_drops_stale_framing(self):
        upstream = {
            "Content-Length": "12",
            "Content-Encoding": "gzip",
            "Content-Range": "bytes 0-11/12",
            "Content-Type": "application/json",
        }
        out = _select_forwarded_generic_response_headers(upstream)
        assert out == {"Content-Type": "application/json"}

    def test_raw_stream_may_preserve_matching_framing(self):
        upstream = {
            "Content-Length": "12",
            "Content-Encoding": "gzip",
            "Content-Range": "bytes 0-11/12",
        }
        out = _select_forwarded_generic_response_headers(upstream, preserve_body_framing=True)
        assert out == upstream

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
        csp = _GATEWAY_SET_SECURITY_HEADERS["Content-Security-Policy"]
        assert "default-src 'none'" in csp
        assert "sandbox" in csp
        assert "frame-ancestors 'none'" in csp
        assert "default-src 'self'" not in csp


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


class TestStripGenericInternalHeaders:
    """The generic hop must strip the gateway-internal identity + signed-token set
    unconditionally (the #1391 leak class), independent of the registration
    denylist -- so neither the caller's identity nor the gateway's internal tokens
    reach a registrant-controlled backend."""

    def test_internal_tokens_and_identity_stripped(self):
        incoming = {
            "X-Internal-Token": "signed-mcp",
            "X-Internal-Token-Generic": "signed-generic",
            "X-Internal-Token-Registry": "signed-registry",
            "X-User": "alice",
            "X-Username": "alice@example.com",
            "X-Scopes": "admin",
            "X-Groups": "everyone",
            "X-Auth-Method": "jwt",
            "X-Client-Id": "cid",
            "X-Original-URL": "/proxy/x/y",
            "X-Generic-Has-Upstream-Auth": "1",
            "X-Upstream-Url": "https://internal/",
            # A benign end-to-end header must survive.
            "Content-Type": "application/json",
            "X-Api-Key": "caller-key",
        }
        out = _strip_generic_internal_headers(incoming)
        assert out == {"Content-Type": "application/json", "X-Api-Key": "caller-key"}

    def test_strip_is_case_insensitive(self):
        out = _strip_generic_internal_headers({"x-internal-token-generic": "t", "X-USER": "a"})
        assert out == {}

    def test_merge_cannot_readmit_stripped_internal_header(self):
        # Even if an internal header name were (wrongly) vended as overridable, the
        # base strip runs first and removed it; the caller's copy is gone, so the
        # merge cannot re-admit it as an upstream value.
        incoming = {"X-Internal-Token-Generic": "signed-generic"}
        fwd = _strip_generic_internal_headers(
            _select_forwarded_generic_request_headers(dict(incoming))
        )
        assert "X-Internal-Token-Generic" not in fwd
        _merge_generic_upstream_headers(
            fwd, incoming, {}, overridable_names=["X-Internal-Token-Generic"]
        )
        assert all(k.lower() != "x-internal-token-generic" for k in fwd)


class TestMergeGenericUpstreamHeaders:
    """The per-header overridable merge policy on the generic egress hop.

    forward_headers is what the protocol allowlist
    (_select_forwarded_generic_request_headers) already produced (gateway creds and
    caller identity excluded, benign caller headers kept); the merge applies operator
    defaults + caller passthrough on top.
    """

    def _base(self, incoming: dict) -> dict:
        # Mirror the handler base: run the raw request headers through the protocol
        # allowlist (_select_forwarded_generic_request_headers). Authorization /
        # X-Authorization are not in the allowlist, so they are absent before the
        # merge and re-admitted only via an explicitly overridable slot.
        return _select_forwarded_generic_request_headers(incoming)

    def test_fixed_header_overwrites_caller(self):
        incoming = {"X-Api-Key": "caller-key", "Content-Type": "application/json"}
        fwd = self._base(incoming)
        _merge_generic_upstream_headers(
            fwd, incoming, {"X-Api-Key": "operator-key"}, overridable_names=[]
        )
        assert fwd["X-Api-Key"] == "operator-key"
        # Benign end-to-end header is untouched.
        assert fwd["Content-Type"] == "application/json"

    def test_fixed_header_overwrites_caller_case_insensitively(self):
        # Caller sends a differently-cased copy; the operator value must be the
        # ONLY one on the wire (no dict-casing duplicate leak).
        incoming = {"x-api-key": "caller-key"}
        fwd = self._base(incoming)
        _merge_generic_upstream_headers(
            fwd, incoming, {"X-Api-Key": "operator-key"}, overridable_names=[]
        )
        keys = [k for k in fwd if k.lower() == "x-api-key"]
        assert keys == ["X-Api-Key"]
        assert fwd["X-Api-Key"] == "operator-key"

    def test_overridable_caller_wins_over_default(self):
        incoming = {"X-Tenant": "caller-tenant"}
        fwd = self._base(incoming)
        _merge_generic_upstream_headers(
            fwd, incoming, {"X-Tenant": "default-tenant"}, overridable_names=["X-Tenant"]
        )
        keys = [k for k in fwd if k.lower() == "x-tenant"]
        assert len(keys) == 1
        assert fwd[keys[0]] == "caller-tenant"

    def test_overridable_default_used_when_caller_absent(self):
        incoming = {"Content-Type": "application/json"}
        fwd = self._base(incoming)
        _merge_generic_upstream_headers(
            fwd, incoming, {"X-Tenant": "default-tenant"}, overridable_names=["X-Tenant"]
        )
        assert fwd["X-Tenant"] == "default-tenant"

    def test_caller_only_slot_forwarded(self):
        # Overridable, NO default value: the caller's header (survived
        # _forward_headers) is forwarded as-is.
        incoming = {"X-Tenant": "caller-tenant"}
        fwd = self._base(incoming)
        _merge_generic_upstream_headers(fwd, incoming, {}, overridable_names=["X-Tenant"])
        assert fwd["X-Tenant"] == "caller-tenant"

    def test_caller_only_slot_absent_yields_nothing(self):
        incoming = {"Content-Type": "application/json"}
        fwd = self._base(incoming)
        _merge_generic_upstream_headers(fwd, incoming, {}, overridable_names=["X-Tenant"])
        assert "X-Tenant" not in fwd

    def test_non_registered_caller_header_is_dropped(self):
        incoming = {"X-Random": "whatever", "Content-Type": "application/json"}
        fwd = self._base(incoming)
        _merge_generic_upstream_headers(fwd, incoming, {"X-Api-Key": "op"}, overridable_names=[])
        assert "X-Random" not in fwd
        assert fwd["Content-Type"] == "application/json"
        assert fwd["X-Api-Key"] == "op"

    def test_every_safe_registered_name_is_readmitted(self):
        incoming = {"x-tenant": "caller-tenant", "X-Correlation-Id": "cid"}
        fwd = self._base(incoming)
        assert fwd == {}
        _merge_generic_upstream_headers(
            fwd,
            incoming,
            {},
            overridable_names=["X-Tenant", "X-Correlation-Id"],
        )
        assert fwd == {"X-Tenant": "caller-tenant", "X-Correlation-Id": "cid"}

    def test_reserved_registered_name_cannot_be_readmitted(self):
        incoming = {"X-Internal-Token-Generic": "signed"}
        fwd = self._base(incoming)
        _merge_generic_upstream_headers(
            fwd,
            incoming,
            {},
            overridable_names=["X-Internal-Token-Generic"],
        )
        assert fwd == {}

    def test_authorization_readmitted_only_when_overridable(self):
        # Authorization is not in the protocol allowlist, so it is absent from the
        # base; it is re-admitted from incoming ONLY when overridable and supplied.
        incoming = {"Authorization": "Bearer caller-token"}
        fwd = self._base(incoming)
        assert "Authorization" not in fwd  # not selected by the protocol allowlist

        _merge_generic_upstream_headers(fwd, incoming, {}, overridable_names=["Authorization"])
        assert fwd["Authorization"] == "Bearer caller-token"

    def test_authorization_not_readmitted_when_not_overridable(self):
        incoming = {"Authorization": "Bearer caller-token"}
        fwd = self._base(incoming)
        _merge_generic_upstream_headers(fwd, incoming, {}, overridable_names=[])
        assert "Authorization" not in fwd

    def test_authorization_default_used_when_caller_absent(self):
        incoming = {"Content-Type": "application/json"}
        fwd = self._base(incoming)
        _merge_generic_upstream_headers(
            fwd,
            incoming,
            {"Authorization": "Bearer operator-default"},
            overridable_names=["Authorization"],
        )
        assert fwd["Authorization"] == "Bearer operator-default"

    def test_authorization_caller_overrides_default(self):
        incoming = {"Authorization": "Bearer caller-token"}
        fwd = self._base(incoming)
        _merge_generic_upstream_headers(
            fwd,
            incoming,
            {"Authorization": "Bearer operator-default"},
            overridable_names=["Authorization"],
        )
        assert fwd["Authorization"] == "Bearer caller-token"


class TestAssertGenericAuthorizationNotGatewayCred:
    """The A2A equal-token guard on the generic hop: the caller's gateway
    credential (X-Authorization) must never reach the backend via Authorization."""

    def test_rejects_when_outbound_auth_equals_gateway_cred(self):
        fwd = {"Authorization": "Bearer gwtoken"}
        with pytest.raises(HTTPException) as e:
            _assert_generic_authorization_not_gateway_cred(fwd, "Bearer gwtoken")
        assert e.value.status_code == 401

    def test_rejects_mixed_case_authorization_header(self):
        fwd = {"aUtHoRiZaTiOn": "Bearer gwtoken"}
        with pytest.raises(HTTPException) as excinfo:
            _assert_generic_authorization_not_gateway_cred(fwd, "Bearer gwtoken")
        assert excinfo.value.status_code == 401

    def test_rejects_ignoring_scheme_and_whitespace(self):
        # X-Authorization sent without the Bearer prefix, Authorization with it.
        fwd = {"Authorization": "Bearer gwtoken"}
        with pytest.raises(HTTPException):
            _assert_generic_authorization_not_gateway_cred(fwd, "  gwtoken  ")

    def test_allows_distinct_upstream_token(self):
        fwd = {"Authorization": "Bearer upstream-token"}
        _assert_generic_authorization_not_gateway_cred(fwd, "Bearer gwtoken")

    def test_noop_when_no_gateway_cred(self):
        fwd = {"Authorization": "Bearer upstream-token"}
        _assert_generic_authorization_not_gateway_cred(fwd, None)

    def test_noop_when_no_outbound_auth(self):
        _assert_generic_authorization_not_gateway_cred({}, "Bearer gwtoken")

    def test_standard_authorization_is_effective_gateway_credential(self):
        incoming = {"Authorization": "Bearer gwtoken"}
        fwd = _select_forwarded_generic_request_headers(incoming)
        _merge_generic_upstream_headers(
            fwd,
            incoming,
            {},
            overridable_names=["Authorization"],
        )
        with pytest.raises(HTTPException) as excinfo:
            _assert_generic_authorization_not_gateway_cred(
                fwd,
                _effective_ingress_gateway_credential(incoming),
            )
        assert excinfo.value.status_code == 401

    def test_x_authorization_keeps_validate_precedence_for_mixed_casing(self):
        incoming = {
            "x-AuThOrIzAtIoN": "Bearer gateway-token",
            "aUtHoRiZaTiOn": "Bearer upstream-token",
        }
        assert _effective_ingress_gateway_credential(incoming) == "Bearer gateway-token"


class TestOutboundUrlLogSanitization:
    def test_drops_userinfo_query_and_fragment(self):
        sanitized = redact_url(
            "https://alice:secret@backend.example:8443/api/run?token=secret#fragment"
        )
        assert sanitized == "https://backend.example:8443/api/run"
        assert "alice" not in sanitized
        assert "secret" not in sanitized
        assert "?" not in sanitized

    def test_invalid_port_fails_closed(self):
        assert redact_url("https://backend.example:bad/x?q=1") == "[REDACTED]"


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
