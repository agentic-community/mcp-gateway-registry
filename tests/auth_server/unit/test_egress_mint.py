"""Auth-server egress mint-path tests: canonical auth_method + nginx marker.

These exercise the two auth_server-side guards:
- _canonical_auth_method: cookie path 'session_cookie' -> session record 'oauth2'.
- _attach_mcp_proxy_token: only mints the egress-capable token when the nginx
  marker matches (when configured); the auth_method claim is stamped.
"""

import jwt as pyjwt
import pytest

from auth_server import server


class _FakeHeaders(dict):
    """Case-insensitive-ish header stub: tests pass exact-case keys."""

    def get(self, key, default=""):
        return super().get(key, default)


class _FakeRequest:
    def __init__(self, headers: dict):
        self.headers = _FakeHeaders(headers)


class _FakeResponse:
    def __init__(self):
        self.headers: dict = {}


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-for-testing-only-do-not-use")


@pytest.mark.unit
class TestCanonicalAuthMethod:
    def test_cookie_maps_to_session_record_value(self):
        vr = {"method": "session_cookie", "data": {"auth_method": "oauth2"}}
        assert server._canonical_auth_method(vr) == "oauth2"

    def test_cookie_defaults_oauth2(self):
        assert server._canonical_auth_method({"method": "session_cookie", "data": {}}) == "oauth2"

    def test_idp_provider_methods_canonicalize_to_oauth2(self):
        # A bearer issued directly by the per-user IdP (what a DCR client like
        # Claude Code / Codex presents) reports the provider name as `method`. It
        # MUST fold into the same `oauth2` bucket the cookie-consent path wrote --
        # otherwise the DCR vend misses the vault and the user loops on consent.
        for method in ("keycloak", "entra", "cognito", "okta", "auth0", "pingfederate"):
            assert server._canonical_auth_method({"method": method}) == "oauth2", method

    def test_network_trusted_passthrough(self):
        # Non-per-user methods are NOT folded -- they pass through so the vend's
        # is_per_user_auth_method() check still rejects them.
        assert server._canonical_auth_method({"method": "network-trusted"}) == "network-trusted"
        assert server._canonical_auth_method({"method": "federation-static"}) == "federation-static"
        assert server._canonical_auth_method({"method": "future-unknown"}) == "future-unknown"

    def test_self_signed_maps_to_inner_auth_method_claim(self):
        # A self-signed JWT (UI 'generate token', or the egress OAuth-facade
        # /token mint) reports method='self_signed' (the FORMAT) but carries the
        # principal's auth_method as an inner claim. The vault keys on the
        # principal method, so this MUST canonicalize to the claim -- else a user
        # who consents via a cookie session (bucket 'oauth2') and vends with a
        # minted token (would-be bucket 'self_signed') loops on consent forever.
        vr = {"method": "self_signed", "data": {"auth_method": "oauth2"}}
        assert server._canonical_auth_method(vr) == "oauth2"

    def test_self_signed_defaults_oauth2_when_claim_absent(self):
        assert server._canonical_auth_method({"method": "self_signed", "data": {}}) == "oauth2"


@pytest.mark.unit
class TestCanonicalEgressUser:
    """The per-user egress vault id must resolve identically on the consent-write
    (cookie) and vend (bearer) paths, or the vaulted token is written under one
    id and looked up under another (permanent vend miss). It keys on the OIDC
    ``sub``, which is present in both id_tokens and access tokens across providers.
    """

    def test_bearer_uses_data_sub(self):
        # Vend path: verified bearer claims land in ``data``; the sub wins.
        vr = {"username": "alice@example.com", "data": {"sub": "00000000-sub-alice"}}
        assert server._canonical_egress_user(vr) == "00000000-sub-alice"

    def test_cookie_uses_persisted_subject(self):
        # Consent-write path: the session carries the sub persisted at login as
        # ``subject`` (create_session), NOT ``sub``. It must resolve the same value.
        vr = {
            "method": "session_cookie",
            "username": "alice@example.com",
            "data": {"subject": "00000000-sub-alice", "auth_method": "oauth2"},
        }
        assert server._canonical_egress_user(vr) == "00000000-sub-alice"

    def test_consent_and_vend_agree_for_entra_shaped_result(self):
        # Entra: the browser id_token has preferred_username (email) but the DCR
        # client's access token does not -- yet both carry the same sub. Keying on
        # sub makes the two paths agree even though ``username`` differs.
        cookie_vr = {
            "method": "session_cookie",
            "username": "alice@contoso.com",  # from preferred_username / email
            "data": {"subject": "entra-oid-sub-123", "auth_method": "oauth2"},
        }
        bearer_vr = {
            "username": "entra-oid-sub-123",  # access token lacks preferred_username
            "data": {"sub": "entra-oid-sub-123"},
        }
        assert server._canonical_egress_user(cookie_vr) == server._canonical_egress_user(bearer_vr)

    def test_falls_back_to_username_when_no_sub(self):
        # Non-OIDC callers (no sub anywhere) keep their pre-existing bucket.
        vr = {"username": "svc-account", "data": {}}
        assert server._canonical_egress_user(vr) == "svc-account"

    def test_empty_when_nothing_present(self):
        assert server._canonical_egress_user({}) == ""


def _decode(token: str) -> dict:
    return pyjwt.decode(
        token,
        "test-secret-key-for-testing-only-do-not-use",
        algorithms=["HS256"],
        audience="mcp-proxy",
        issuer="mcp-auth-server",
    )


@pytest.mark.unit
class TestAttachMcpProxyTokenMarker:
    def test_no_upstream_does_not_mint(self):
        resp = _FakeResponse()
        server._attach_mcp_proxy_token(
            _FakeRequest({}), resp, subject="alice", scopes=[], server_name="github-mcp"
        )
        assert "X-Internal-Token" not in resp.headers

    def test_empty_marker_mints_unconditionally(self, monkeypatch):
        # Function-level fallback only: an empty marker is rejected at startup
        # (Settings.__init__), so this state is unreachable in a running server.
        # Kept to pin the helper's branch behavior.
        monkeypatch.setattr(server.settings, "auth_server_nginx_marker_secret", "")
        resp = _FakeResponse()
        server._attach_mcp_proxy_token(
            _FakeRequest({"X-Resolved-Upstream": "https://u/mcp"}),
            resp,
            subject="alice",
            scopes=["repo"],
            server_name="github-mcp",
            auth_method="oauth2",
        )
        claims = _decode(resp.headers["X-Internal-Token"])
        assert claims["sub"] == "alice"
        assert claims["auth_method"] == "oauth2"
        assert claims["upstream_url"] == "https://u/mcp"

    def test_marker_enabled_and_matching_mints(self, monkeypatch):
        monkeypatch.setattr(server.settings, "auth_server_nginx_marker_secret", "s3cret")
        resp = _FakeResponse()
        server._attach_mcp_proxy_token(
            _FakeRequest(
                {"X-Resolved-Upstream": "https://u/mcp", "X-Validate-Source-Secret": "s3cret"}
            ),
            resp,
            subject="alice",
            scopes=[],
            server_name="github-mcp",
            auth_method="oauth2",
        )
        assert "X-Internal-Token" in resp.headers

    def test_marker_enabled_and_missing_does_not_mint(self, monkeypatch):
        # Direct :8888 caller (no nginx marker) gets no egress-capable token
        # even with a forged X-Resolved-Upstream.
        monkeypatch.setattr(server.settings, "auth_server_nginx_marker_secret", "s3cret")
        resp = _FakeResponse()
        server._attach_mcp_proxy_token(
            _FakeRequest({"X-Resolved-Upstream": "https://attacker.example/mcp"}),
            resp,
            subject="alice",
            scopes=[],
            server_name="github-mcp",
            auth_method="oauth2",
        )
        assert "X-Internal-Token" not in resp.headers

    def test_marker_enabled_and_mismatch_does_not_mint(self, monkeypatch):
        monkeypatch.setattr(server.settings, "auth_server_nginx_marker_secret", "s3cret")
        resp = _FakeResponse()
        server._attach_mcp_proxy_token(
            _FakeRequest(
                {"X-Resolved-Upstream": "https://u/mcp", "X-Validate-Source-Secret": "wrong"}
            ),
            resp,
            subject="alice",
            scopes=[],
            server_name="github-mcp",
            auth_method="oauth2",
        )
        assert "X-Internal-Token" not in resp.headers

    def test_egress_user_claim_is_stamped(self, monkeypatch):
        # The vend path reads egress_user off this token to key the vault, so the
        # canonical per-user id must ride along even when it differs from subject.
        monkeypatch.setattr(server.settings, "auth_server_nginx_marker_secret", "")
        resp = _FakeResponse()
        server._attach_mcp_proxy_token(
            _FakeRequest({"X-Resolved-Upstream": "https://u/mcp"}),
            resp,
            subject="alice@example.com",
            scopes=[],
            server_name="github-mcp",
            auth_method="oauth2",
            egress_user="00000000-sub-alice",
        )
        claims = _decode(resp.headers["X-Internal-Token"])
        assert claims["egress_user"] == "00000000-sub-alice"
