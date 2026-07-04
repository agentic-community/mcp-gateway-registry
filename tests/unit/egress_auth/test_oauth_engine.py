"""OAuth engine tests: PKCE S256, authorize URL, exchange/refresh, quirk hooks.

Network is stubbed by monkeypatching the single chokepoint ``_post_token`` so
no real provider is contacted.
"""

import base64
import hashlib
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
