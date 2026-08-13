"""OAuth engine tests: PKCE S256, authorize URL, exchange/refresh, quirk hooks.

Network is stubbed by monkeypatching the single chokepoint ``_post_token`` so
no real provider is contacted.
"""

import base64
import hashlib
import json
from datetime import UTC, datetime
from urllib.parse import parse_qs, urlparse

import pytest

from registry.egress_auth import oauth_engine
from registry.egress_auth.providers import PROVIDER_REGISTRY, resolve_provider
from registry.egress_auth.schemas import OAuthProviderConfig


@pytest.mark.unit
class TestPKCE:
    def test_verifier_charset_and_length(self):
        v = oauth_engine.generate_pkce_verifier()
        assert 43 <= len(v) <= 128
        assert "=" not in v and "+" not in v and "/" not in v

    def test_s256_challenge_matches_spec(self):
        v = "test-verifier"
        expected = (
            base64.urlsafe_b64encode(hashlib.sha256(v.encode()).digest()).rstrip(b"=").decode()
        )
        assert oauth_engine.pkce_challenge_s256(v) == expected


@pytest.mark.unit
class TestAuthorizeUrl:
    def test_contains_required_params(self):
        cfg = PROVIDER_REGISTRY["github"]
        url = oauth_engine.build_authorize_url(
            cfg=cfg,
            client_id="Iv1.abc",
            redirect_uri="https://gw/oauth2/egress/callback",
            scopes=["repo", "read:user"],
            state="STATEBLOB",
            pkce_challenge="CHAL",
        )
        q = parse_qs(urlparse(url).query)
        assert q["response_type"] == ["code"]
        assert q["client_id"] == ["Iv1.abc"]
        assert q["redirect_uri"] == ["https://gw/oauth2/egress/callback"]
        assert q["state"] == ["STATEBLOB"]
        assert q["scope"] == ["repo read:user"]
        assert q["code_challenge"] == ["CHAL"]
        assert q["code_challenge_method"] == ["S256"]

    def test_extra_authorize_params_included(self):
        cfg = PROVIDER_REGISTRY["google"]
        url = oauth_engine.build_authorize_url(cfg, "cid", "https://gw/cb", ["openid"], "S", "CHAL")
        q = parse_qs(urlparse(url).query)
        assert q["access_type"] == ["offline"]
        assert q["prompt"] == ["consent"]

    def test_custom_scope_separator(self):
        cfg = resolve_provider(
            {
                "provider": "custom",
                "custom_authorize_url": "https://idp/auth",
                "custom_token_url": "https://idp/token",
                "custom_scope_separator": ",",
            }
        )
        url = oauth_engine.build_authorize_url(cfg, "cid", "https://gw/cb", ["a", "b"], "S", "C")
        q = parse_qs(urlparse(url).query)
        assert q["scope"] == ["a,b"]


@pytest.mark.unit
class TestExchangeAndRefresh:
    async def test_exchange_standard(self, monkeypatch):
        async def fake_post(cfg, data, headers):
            assert data["grant_type"] == "authorization_code"
            assert data["code"] == "the-code"
            assert data["code_verifier"] == "verif"
            return {
                "access_token": "at_123",
                "refresh_token": "rt_123",
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": "repo read:user",
            }

        monkeypatch.setattr(oauth_engine, "_post_token", fake_post)
        tok = await oauth_engine.exchange_code(
            PROVIDER_REGISTRY["github"], "cid", "secret", "the-code", "https://gw/cb", "verif"
        )
        assert tok.access_token == "at_123"
        assert tok.refresh_token == "rt_123"
        assert tok.scopes == ["repo", "read:user"]
        assert tok.expires_at is not None
        assert tok.client_id == "cid"

    async def test_refresh_keeps_old_refresh_when_not_returned(self, monkeypatch):
        async def fake_post(cfg, data, headers):
            assert data["grant_type"] == "refresh_token"
            return {"access_token": "at_new", "token_type": "Bearer", "expires_in": 3600}

        monkeypatch.setattr(oauth_engine, "_post_token", fake_post)
        tok = await oauth_engine.refresh_token(
            PROVIDER_REGISTRY["google"], "cid", "secret", "rt_old"
        )
        assert tok.access_token == "at_new"
        assert tok.refresh_token == "rt_old"  # fallback retained

    async def test_refresh_rotation_takes_new_refresh(self, monkeypatch):
        async def fake_post(cfg, data, headers):
            return {"access_token": "at2", "refresh_token": "rt2", "expires_in": 3600}

        monkeypatch.setattr(oauth_engine, "_post_token", fake_post)
        tok = await oauth_engine.refresh_token(PROVIDER_REGISTRY["slack"], "cid", "secret", "rt1")
        assert tok.refresh_token == "rt2"

    async def test_missing_access_token_raises(self, monkeypatch):
        async def fake_post(cfg, data, headers):
            return {"token_type": "Bearer"}

        monkeypatch.setattr(oauth_engine, "_post_token", fake_post)
        with pytest.raises(oauth_engine.OAuthEngineError, match="missing access_token"):
            await oauth_engine.exchange_code(
                PROVIDER_REGISTRY["github"], "cid", "secret", "c", "https://gw/cb", "v"
            )


def _make_jwt(exp: int | None) -> str:
    """Minimal unsigned JWT (header.payload.sig) carrying an optional ``exp`` claim."""

    def seg(obj: dict) -> str:
        raw = json.dumps(obj).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    claims: dict = {"iss": "https://example.my.salesforce.com"}
    if exp is not None:
        claims["exp"] = exp
    return f"{seg({'alg': 'RS256', 'typ': 'JWT'})}.{seg(claims)}.sig"


@pytest.mark.unit
class TestExpiresAtFallback:
    """Providers that omit ``expires_in`` (e.g. Salesforce) still bound the access
    token via the JWT ``exp`` claim. Cover both token-endpoint call sites, since
    exchange and refresh both funnel through ``_to_stored_token``."""

    def test_expires_in_takes_precedence_over_jwt_exp(self):
        # Stale JWT exp must not override an explicit, fresher expires_in.
        past = int(datetime.now(UTC).timestamp()) - 3600
        at = oauth_engine._expires_at(3600, _make_jwt(past))
        assert at is not None
        assert datetime.fromisoformat(at) > datetime.now(UTC)

    def test_jwt_exp_used_when_expires_in_missing(self):
        exp = int(datetime.now(UTC).timestamp()) + 7200
        at = oauth_engine._expires_at(None, _make_jwt(exp))
        assert at is not None
        assert datetime.fromisoformat(at) == datetime.fromtimestamp(exp, tz=UTC)

    def test_none_for_opaque_token_without_expires_in(self):
        assert oauth_engine._expires_at(None, "opaque-not-a-jwt") is None

    def test_none_for_jwt_without_exp_claim(self):
        assert oauth_engine._expires_at(None, _make_jwt(None)) is None

    def test_none_for_malformed_jwt(self):
        assert oauth_engine._expires_at(None, "a.!!!notb64!!!.c") is None

    async def test_exchange_sets_expires_at_from_jwt(self, monkeypatch):
        exp = int(datetime.now(UTC).timestamp()) + 14400  # Salesforce ~4h JWT

        async def fake_post(cfg, data, headers):
            # No expires_in, JWT access token -- the Salesforce shape.
            return {"access_token": _make_jwt(exp), "refresh_token": "rt", "scope": "mcp_api"}

        monkeypatch.setattr(oauth_engine, "_post_token", fake_post)
        tok = await oauth_engine.exchange_code(
            PROVIDER_REGISTRY["github"], "cid", "secret", "c", "https://gw/cb", "v"
        )
        assert tok.expires_at is not None
        assert datetime.fromisoformat(tok.expires_at) == datetime.fromtimestamp(exp, tz=UTC)

    async def test_refresh_sets_expires_at_from_jwt(self, monkeypatch):
        exp = int(datetime.now(UTC).timestamp()) + 14400

        async def fake_post(cfg, data, headers):
            return {"access_token": _make_jwt(exp)}

        monkeypatch.setattr(oauth_engine, "_post_token", fake_post)
        tok = await oauth_engine.refresh_token(PROVIDER_REGISTRY["google"], "cid", "secret", "rt")
        assert tok.expires_at is not None
        assert datetime.fromisoformat(tok.expires_at) == datetime.fromtimestamp(exp, tz=UTC)


@pytest.mark.unit
class TestResourceIndicator:
    """RFC 8707 resource indicator threads through authorize + exchange + refresh.

    A resource server that mints per-resource tokens (e.g. Atlassian's Rovo MCP)
    requires the ``resource`` param on the authorize request AND both token
    grants, or it rejects the flow ('Invalid context provided'). A provider
    without a resource (every built-in) must never emit the param.
    """

    _RES = "https://mcp.atlassian.com/v1/mcp/authv2"

    def _custom_cfg_with_resource(self):
        return resolve_provider(
            {
                "provider": "custom",
                "custom_authorize_url": "https://auth.atlassian.com/authorize",
                "custom_token_url": "https://auth.atlassian.com/oauth/token",
                "custom_resource": self._RES,
            }
        )

    def test_authorize_url_includes_resource(self):
        cfg = self._custom_cfg_with_resource()
        url = oauth_engine.build_authorize_url(
            cfg, "cid", "https://gw/cb", ["read:jira-work"], "S", "CHAL"
        )
        assert parse_qs(urlparse(url).query)["resource"] == [self._RES]

    def test_authorize_url_omits_resource_when_unset(self):
        url = oauth_engine.build_authorize_url(
            PROVIDER_REGISTRY["github"], "cid", "https://gw/cb", ["repo"], "S", "CHAL"
        )
        assert "resource" not in parse_qs(urlparse(url).query)

    async def test_exchange_sends_resource(self, monkeypatch):
        captured: dict = {}

        async def fake_post(cfg, data, headers):
            captured.update(data)
            return {"access_token": "at", "expires_in": 3600}

        monkeypatch.setattr(oauth_engine, "_post_token", fake_post)
        await oauth_engine.exchange_code(
            self._custom_cfg_with_resource(), "cid", "sec", "code", "https://gw/cb", "verif"
        )
        assert captured["resource"] == self._RES

    async def test_refresh_sends_resource(self, monkeypatch):
        captured: dict = {}

        async def fake_post(cfg, data, headers):
            captured.update(data)
            return {"access_token": "at2", "expires_in": 3600}

        monkeypatch.setattr(oauth_engine, "_post_token", fake_post)
        await oauth_engine.refresh_token(self._custom_cfg_with_resource(), "cid", "sec", "rt")
        assert captured["resource"] == self._RES

    async def test_exchange_omits_resource_when_unset(self, monkeypatch):
        captured: dict = {}

        async def fake_post(cfg, data, headers):
            captured.update(data)
            return {"access_token": "at", "expires_in": 3600}

        monkeypatch.setattr(oauth_engine, "_post_token", fake_post)
        await oauth_engine.exchange_code(
            PROVIDER_REGISTRY["github"], "cid", "sec", "code", "https://gw/cb", "verif"
        )
        assert "resource" not in captured


@pytest.mark.unit
class TestQuirkParsers:
    def test_slack_nested_lifts_user_token(self):
        cfg = PROVIDER_REGISTRY["slack"]
        payload = {
            "ok": True,
            "authed_user": {
                "access_token": "xoxp-user",
                "token_type": "Bearer",
                "scope": "search:read",
            },
        }
        out = oauth_engine._parse_token_response(cfg, payload)
        assert out["access_token"] == "xoxp-user"
        assert out["scope"] == "search:read"

    def test_slack_user_endpoint_top_level_token(self):
        # The v2_user token endpoint (oauth.v2.user.access) returns the user
        # token at the TOP level rather than nested under authed_user. The parser
        # must fall through to it instead of dropping the token.
        cfg = PROVIDER_REGISTRY["slack"]
        payload = {
            "ok": True,
            "access_token": "xoxp-user-top",
            "token_type": "Bearer",
            "scope": "search:read,chat:write",
        }
        out = oauth_engine._parse_token_response(cfg, payload)
        assert out["access_token"] == "xoxp-user-top"
        assert out["scope"] == "search:read,chat:write"

    def test_slack_error_raises(self):
        cfg = PROVIDER_REGISTRY["slack"]
        with pytest.raises(oauth_engine.OAuthEngineError, match="Slack token error"):
            oauth_engine._parse_token_response(cfg, {"ok": False, "error": "invalid_code"})

    def test_basic_header_auth_style(self):
        cfg = OAuthProviderConfig(
            name="c",
            display_name="C",
            authorize_url="https://i/a",
            token_url="https://i/t",
            token_endpoint_auth_style="basic_header",
        )
        data, headers = oauth_engine._build_token_request(cfg, "cid", "sec", {"grant_type": "x"})
        assert headers["Authorization"].startswith("Basic ")
        assert "client_secret" not in data  # secret is in the header, not the body
        assert data["client_id"] == "cid"

    def test_post_body_auth_style_default(self):
        cfg = PROVIDER_REGISTRY["github"]
        data, headers = oauth_engine._build_token_request(cfg, "cid", "sec", {"grant_type": "x"})
        assert data["client_id"] == "cid"
        assert data["client_secret"] == "sec"
        assert "Authorization" not in headers
        assert headers["Accept"] == "application/json"


@pytest.mark.unit
class TestPostTokenSsrfGuard:
    """The token endpoint receives the operator client_secret (and, on refresh,
    the user's refresh_token). For a 'custom' provider the token_url is
    registrant-supplied, so ``_post_token`` must route through the SSRF/rebinding
    -safe client and refuse a target that resolves to a private/metadata address
    -- otherwise the credential is exfiltrated via SSRF. These exercise the real
    ``_post_token`` (no monkeypatch of the chokepoint) so the transport guard is
    on the path.
    """

    def _custom_cfg(self, token_url: str) -> OAuthProviderConfig:
        return OAuthProviderConfig(
            name="custom",
            display_name="Custom OIDC",
            is_builtin=False,
            authorize_url="https://evil.example.com/authorize",
            token_url=token_url,
            use_pkce=True,
        )

    async def test_token_url_to_metadata_ip_fails_closed(self):
        # A literal cloud-metadata target must be blocked before the secret is
        # ever sent, surfaced as an OAuthEngineError (unreachable), not a token.
        cfg = self._custom_cfg("http://169.254.169.254/latest/meta-data/")
        data, headers = oauth_engine._build_token_request(
            cfg, "cid", "supersecret", {"grant_type": "x"}
        )
        with pytest.raises(oauth_engine.OAuthEngineError, match="blocked by security policy"):
            await oauth_engine._post_token(cfg, data, headers)

    async def test_token_url_to_loopback_fails_closed(self):
        cfg = self._custom_cfg("http://127.0.0.1:8200/v1/secret")
        data, headers = oauth_engine._build_token_request(
            cfg, "cid", "supersecret", {"grant_type": "x"}
        )
        with pytest.raises(oauth_engine.OAuthEngineError, match="blocked by security policy"):
            await oauth_engine._post_token(cfg, data, headers)

    async def test_token_url_to_rfc1918_fails_closed(self):
        cfg = self._custom_cfg("http://10.0.0.5/token")
        data, headers = oauth_engine._build_token_request(
            cfg, "cid", "supersecret", {"grant_type": "x"}
        )
        with pytest.raises(oauth_engine.OAuthEngineError, match="blocked by security policy"):
            await oauth_engine._post_token(cfg, data, headers)


@pytest.mark.unit
class TestCredentialedOAuthTransportProfile:
    async def test_post_uses_https_only_empty_allowlist_profile(self, monkeypatch):
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock, MagicMock

        captured = {}

        @asynccontextmanager
        async def fake_client(*, profile, timeout):
            captured["profile"] = profile
            response = MagicMock(status_code=200)
            response.json.return_value = {"access_token": "ok"}
            client = MagicMock()
            client.post = AsyncMock(return_value=response)
            yield client

        monkeypatch.setattr(oauth_engine, "guarded_async_client", fake_client)
        cfg = OAuthProviderConfig(
            name="custom",
            display_name="Custom",
            authorize_url="https://tokens.example/authorize",
            token_url="https://tokens.example/oauth/token",
        )
        await oauth_engine._post_token(cfg, {"client_secret": "secret"}, {})
        assert captured["profile"] is oauth_engine.CREDENTIALED_OAUTH_PROFILE
        assert captured["profile"].allowlist_factory().hosts == frozenset()

    async def test_guard_error_detail_is_not_propagated(self, monkeypatch):
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def blocked_client(*, profile, timeout):
            del profile, timeout
            raise oauth_engine.UrlValidationError(
                "https://tokens.example/token?api_key=query-secret",
                "raw-exception-secret",
            )
            yield  # pragma: no cover

        monkeypatch.setattr(oauth_engine, "guarded_async_client", blocked_client)
        cfg = OAuthProviderConfig(
            name="custom",
            display_name="Custom",
            authorize_url="https://tokens.example/authorize",
            token_url="https://tokens.example/oauth/token",
        )

        with pytest.raises(oauth_engine.OAuthEngineError) as exc_info:
            await oauth_engine._post_token(cfg, {"client_secret": "secret"}, {})

        assert str(exc_info.value) == "token endpoint blocked by security policy"
        assert "query-secret" not in str(exc_info.value)

    async def test_transport_error_detail_is_not_propagated(self, monkeypatch):
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock, MagicMock

        @asynccontextmanager
        async def failing_client(*, profile, timeout):
            del profile, timeout
            client = MagicMock()
            client.post = AsyncMock(
                side_effect=oauth_engine.httpx.ConnectError("raw-exception-secret")
            )
            yield client

        monkeypatch.setattr(oauth_engine, "guarded_async_client", failing_client)
        cfg = OAuthProviderConfig(
            name="custom",
            display_name="Custom",
            authorize_url="https://tokens.example/authorize",
            token_url="https://tokens.example/oauth/token",
        )

        with pytest.raises(oauth_engine.OAuthEngineError) as exc_info:
            await oauth_engine._post_token(cfg, {"client_secret": "secret"}, {})

        assert str(exc_info.value) == "token endpoint unreachable"
        assert "raw-exception-secret" not in str(exc_info.value)
