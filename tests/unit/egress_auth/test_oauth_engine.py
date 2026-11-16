"""OAuth engine tests: PKCE S256, authorize URL, exchange/refresh, quirk hooks.

Network is stubbed by monkeypatching the single chokepoint ``_post_token`` so
no real provider is contacted.
"""

import base64
import hashlib
import json
from datetime import UTC, datetime
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from registry.egress_auth import oauth_engine
from registry.egress_auth.providers import PROVIDER_REGISTRY, resolve_provider
from registry.egress_auth.schemas import OAuthProviderConfig
from tests.unit.egress_auth.dcr_fixtures import (
    DCR_TEST_ISSUER,
    DCR_TEST_PRM_URL,
    DCR_TEST_RESOURCE,
    make_dcr_test_config,
)


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
class TestClientCredentials:
    """RFC 6749 §4.4 client_credentials grant for REGISTRY-SELF backend auth."""

    def _cfg(self):
        return OAuthProviderConfig(
            name="backend-client-credentials",
            display_name="Backend OAuth",
            authorize_url="https://idp.example/token",
            token_url="https://idp.example/token",
            use_pkce=False,
        )

    async def test_grant_type_and_scopes(self, monkeypatch):
        captured = {}

        async def fake_post(cfg, data, headers):
            captured.update(data)
            return {"access_token": "cc_tok", "token_type": "Bearer", "expires_in": 3600}

        monkeypatch.setattr(oauth_engine, "_post_token", fake_post)
        tok = await oauth_engine.client_credentials_token(
            self._cfg(), "cid", "secret", ["api:read", "api:write"]
        )
        assert captured["grant_type"] == "client_credentials"
        assert captured["scope"] == "api:read api:write"
        assert captured["client_id"] == "cid"
        assert captured["client_secret"] == "secret"
        assert tok.access_token == "cc_tok"
        assert tok.expires_at is not None
        # client_credentials has no refresh token.
        assert tok.refresh_token is None

    async def test_no_scopes_omits_scope_param(self, monkeypatch):
        captured = {}

        async def fake_post(cfg, data, headers):
            captured.update(data)
            return {"access_token": "t", "expires_in": 60}

        monkeypatch.setattr(oauth_engine, "_post_token", fake_post)
        await oauth_engine.client_credentials_token(self._cfg(), "cid", "secret", [])
        assert "scope" not in captured

    async def test_missing_secret_confidential_raises(self, monkeypatch):
        # post_body (default) is a confidential style; a missing secret must fail
        # closed before any network call.
        with pytest.raises(oauth_engine.OAuthEngineError, match="no client_secret"):
            await oauth_engine.client_credentials_token(self._cfg(), "cid", None, ["s"])


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

    A resource server that binds tokens to one resource expects the same
    ``resource`` param on the authorize request AND both token grants. A
    provider without a resource (every built-in) must never emit the param.
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
class TestPublicClientNoneStyle:
    """RFC 7591 ``token_endpoint_auth_method=none``: a PUBLIC client (e.g. one
    minted by an MCP resource server's DCR endpoint, such as Datadog's MCP) has
    no client_secret at all. The engine must send ``client_id`` + PKCE verifier
    only -- never a ``client_secret`` key, never a Basic header -- on both the
    code exchange and the refresh grant. Confidential styles fail closed on a
    missing secret instead of sending an empty one.
    """

    _RES = "https://mcp.datadoghq.com/api/unstable/mcp-server/mcp"

    def _public_cfg(self):
        return resolve_provider(
            {
                "provider": "custom",
                "custom_authorize_url": "https://app.datadoghq.com/oauth2/v1/authorize",
                "custom_token_url": "https://app.datadoghq.com/api/v2/oauth2/token",
                "custom_token_auth_style": "none",
                "custom_resource": self._RES,
            }
        )

    def test_build_token_request_sends_client_id_only(self):
        cfg = self._public_cfg()
        data, headers = oauth_engine._build_token_request(cfg, "cid", None, {"grant_type": "x"})
        assert data["client_id"] == "cid"
        assert "client_secret" not in data
        assert "Authorization" not in headers

    @pytest.mark.parametrize("style", ["post_body", "basic_header"])
    def test_confidential_style_fails_closed_without_secret(self, style):
        cfg = OAuthProviderConfig(
            name="c",
            display_name="C",
            authorize_url="https://i/a",
            token_url="https://i/t",
            token_endpoint_auth_style=style,
        )
        with pytest.raises(oauth_engine.OAuthEngineError, match="no client_secret"):
            oauth_engine._build_token_request(cfg, "cid", None, {"grant_type": "x"})

    async def test_exchange_sends_verifier_and_resource_without_secret(self, monkeypatch):
        captured: dict = {}
        captured_headers: dict = {}

        async def fake_post(cfg, data, headers):
            captured.update(data)
            captured_headers.update(headers)
            return {"access_token": "at", "refresh_token": "rt", "expires_in": 3600}

        monkeypatch.setattr(oauth_engine, "_post_token", fake_post)
        tok = await oauth_engine.exchange_code(
            self._public_cfg(), "cid", None, "code", "https://gw/cb", "verif"
        )
        assert captured["client_id"] == "cid"
        assert captured["code_verifier"] == "verif"
        assert captured["resource"] == self._RES
        assert "client_secret" not in captured
        assert "Authorization" not in captured_headers
        assert tok.access_token == "at"

    async def test_refresh_sends_client_id_and_resource_without_secret(self, monkeypatch):
        captured: dict = {}

        async def fake_post(cfg, data, headers):
            captured.update(data)
            return {"access_token": "at2", "expires_in": 3600}

        monkeypatch.setattr(oauth_engine, "_post_token", fake_post)
        tok = await oauth_engine.refresh_token(self._public_cfg(), "cid", None, "rt_old")
        assert captured["grant_type"] == "refresh_token"
        assert captured["client_id"] == "cid"
        assert captured["resource"] == self._RES
        assert "client_secret" not in captured
        # provider did not rotate the refresh token -> keep the old one
        assert tok.refresh_token == "rt_old"

    def test_public_client_keeps_pkce_mandatory(self):
        # use_pkce stays True for custom providers regardless of auth style;
        # PKCE is the public client's only proof of possession.
        assert self._public_cfg().use_pkce is True


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
        expected = base64.b64encode(b"cid:sec").decode()
        assert headers["Authorization"] == f"Basic {expected}"
        assert "client_id" not in data
        assert "client_secret" not in data

    def test_post_body_auth_style_default(self):
        cfg = PROVIDER_REGISTRY["github"]
        data, headers = oauth_engine._build_token_request(cfg, "cid", "sec", {"grant_type": "x"})
        assert data["client_id"] == "cid"
        assert data["client_secret"] == "sec"
        assert "Authorization" not in headers
        assert headers["Accept"] == "application/json"


# --------------------------------------------------------------------------- #
# Fakes for the guarded transport (covers the real _get_json / _post_dcr bodies)
# --------------------------------------------------------------------------- #


class _FakeResp:
    def __init__(self, payload, status_code=200, non_json=False):
        self._payload = payload
        self.status_code = status_code
        self._non_json = non_json

    def json(self):
        if self._non_json:
            raise ValueError("not json")
        return self._payload


def _fake_guarded_sequence(responses, rec=None):
    """A shared_guarded_async_client stand-in that hands back queued responses in order."""
    it = iter(responses)

    class _Client:
        async def get(self, url, headers=None, timeout=None):
            if rec is not None:
                rec.setdefault("get_urls", []).append(url)
            return next(it)

        async def post(self, url, json=None, headers=None, timeout=None):
            if rec is not None:
                rec["post_url"] = url
                rec["post_json"] = json
            return next(it)

    def _factory(*, profile, verify=True):
        if rec is not None:
            rec["profile"] = profile
        return _Client()

    return _factory


def _fake_guarded_raising(exc):
    """A shared_guarded_async_client stand-in whose get/post raise ``exc`` (transport error)."""

    class _Client:
        async def get(self, url, headers=None, timeout=None):
            raise exc

        async def post(self, url, json=None, headers=None, timeout=None):
            raise exc

    def _factory(*, profile, verify=True):
        return _Client()

    return _factory


# Token endpoints the issuer check anchors on: the test provider's own, and Atlassian's.
_DCR_TEST_TOKEN_URL = "https://auth.example.com/oauth/token"
_ATLASSIAN_TOKEN_URL = "https://auth.atlassian.com/oauth/token"


def _prm(**over):
    """Protected-resource metadata that passes the RFC 9728 resource check."""
    doc = {
        "resource": DCR_TEST_RESOURCE,
        "authorization_servers": [DCR_TEST_ISSUER],
    }
    doc.update(over)
    return doc


def _as_meta(**over):
    """Authorization-server metadata that passes the RFC 8414 issuer and host checks."""
    doc = {
        "issuer": DCR_TEST_ISSUER,
        "registration_endpoint": f"{DCR_TEST_ISSUER}/dcr/register",
    }
    doc.update(over)
    return doc


def _fake_discovery(monkeypatch, prm=None, as_meta=None, seen=None):
    """Serve ``prm`` for the PRM URL and ``as_meta`` for anything else via _get_json."""

    async def fake_get_json(url):
        if seen is not None:
            seen.append(url)
        if "oauth-protected-resource" in url:
            return prm if prm is not None else _prm()
        return as_meta if as_meta is not None else _as_meta()

    monkeypatch.setattr(oauth_engine, "_get_json", fake_get_json)


@pytest.mark.unit
class TestDynamicClientRegistration:
    """RFC 7591 DCR: discovery (RFC 9728 -> RFC 8414) + public/confidential register."""

    async def test_discovery_walks_prm_then_as_metadata(self, monkeypatch):
        seen: list[str] = []
        _fake_discovery(monkeypatch, seen=seen)
        url = await oauth_engine._discover_registration_url(make_dcr_test_config())
        assert url == f"{DCR_TEST_ISSUER}/dcr/register"
        assert seen == [
            DCR_TEST_PRM_URL,
            f"{DCR_TEST_ISSUER}/.well-known/oauth-authorization-server",
        ]

    async def test_discovery_prefers_pinned_registration_url(self, monkeypatch):
        async def boom(url):  # must not be called when registration_url is pinned
            raise AssertionError("discovery should be skipped")

        monkeypatch.setattr(oauth_engine, "_get_json", boom)
        cfg = OAuthProviderConfig(
            name="p",
            display_name="P",
            authorize_url="https://i/a",
            token_url="https://i/t",
            requires_dcr=True,
            registration_url="https://i/register",
        )
        assert await oauth_engine._discover_registration_url(cfg) == "https://i/register"

    async def test_discovery_no_source_raises(self):
        cfg = make_dcr_test_config(protected_resource_metadata_url=None)
        with pytest.raises(oauth_engine.OAuthEngineError, match="registration_url"):
            await oauth_engine._discover_registration_url(cfg)

    async def test_register_public_client_no_secret(self, monkeypatch):
        captured: dict = {}

        async def fake_disc(cfg, prm=None):
            return f"{DCR_TEST_ISSUER}/dcr/register"

        async def fake_post(reg_url, body):
            captured["reg_url"] = reg_url
            captured["body"] = body
            return {"client_id": "DCRID", "client_secret_expires_at": 0}

        monkeypatch.setattr(oauth_engine, "_discover_registration_url", fake_disc)
        monkeypatch.setattr(oauth_engine, "_post_dcr", fake_post)
        cid, secret = await oauth_engine.register_dcr_client(
            make_dcr_test_config(),
            redirect_uri="https://gw/oauth2/egress/callback",
            scopes=["read:work", "offline_access"],
        )
        assert cid == "DCRID"
        assert secret is None  # public client
        assert captured["body"]["token_endpoint_auth_method"] == "none"
        assert captured["body"]["redirect_uris"] == ["https://gw/oauth2/egress/callback"]
        assert captured["body"]["scope"] == "read:work offline_access"
        assert "authorization_code" in captured["body"]["grant_types"]

    async def test_register_confidential_client_returns_secret(self, monkeypatch):
        async def fake_disc(cfg, prm=None):
            return "https://idp/register"

        async def fake_post(reg_url, body):
            assert body["token_endpoint_auth_method"] == "client_secret_post"
            return {"client_id": "CID", "client_secret": "SEC"}

        cfg = OAuthProviderConfig(
            name="p",
            display_name="P",
            authorize_url="https://i/a",
            token_url="https://i/t",
            requires_dcr=True,
            registration_url="https://idp/register",
        )
        monkeypatch.setattr(oauth_engine, "_discover_registration_url", fake_disc)
        monkeypatch.setattr(oauth_engine, "_post_dcr", fake_post)
        cid, secret = await oauth_engine.register_dcr_client(cfg, "https://gw/cb", [])
        assert (cid, secret) == ("CID", "SEC")

    async def test_register_missing_client_id_raises(self, monkeypatch):
        async def fake_disc(cfg, prm=None):
            return "https://idp/register"

        async def fake_post(reg_url, body):
            return {"client_secret": "SEC"}  # no client_id

        monkeypatch.setattr(oauth_engine, "_discover_registration_url", fake_disc)
        monkeypatch.setattr(oauth_engine, "_post_dcr", fake_post)
        with pytest.raises(oauth_engine.OAuthEngineError, match="missing client_id"):
            await oauth_engine.register_dcr_client(make_dcr_test_config(), "https://gw/cb", [])

    async def test_discovery_no_authorization_servers_raises(self, monkeypatch):
        _fake_discovery(monkeypatch, prm=_prm(authorization_servers=[]))
        with pytest.raises(oauth_engine.OAuthEngineError, match="no authorization_servers"):
            await oauth_engine._discover_registration_url(make_dcr_test_config())

    async def test_discovery_no_registration_endpoint_raises(self, monkeypatch):
        _fake_discovery(monkeypatch, as_meta={"issuer": DCR_TEST_ISSUER})
        with pytest.raises(oauth_engine.OAuthEngineError, match="no registration_endpoint"):
            await oauth_engine._discover_registration_url(make_dcr_test_config())

    async def test_register_end_to_end_through_guarded_client(self, monkeypatch):
        # Exercise the real _discover_registration_url + _post_dcr bodies (only the
        # guarded transport is faked), so discovery + registration are covered
        # end-to-end rather than monkeypatched away.
        rec: dict = {}
        monkeypatch.setattr(
            oauth_engine,
            "shared_guarded_async_client",
            _fake_guarded_sequence(
                [_FakeResp(_prm()), _FakeResp(_as_meta()), _FakeResp({"client_id": "DCRID"})],
                rec,
            ),
        )
        cid, secret = await oauth_engine.register_dcr_client(
            make_dcr_test_config(), "https://gw/cb", ["read:work"]
        )
        assert (cid, secret) == ("DCRID", None)
        assert rec["post_url"] == f"{DCR_TEST_ISSUER}/dcr/register"
        assert rec["post_json"]["token_endpoint_auth_method"] == "none"
        assert rec["profile"] is oauth_engine.CREDENTIALED_OAUTH_PROFILE


@pytest.mark.unit
class TestDiscoveryMetadataChecks:
    """RFC 9728 section 3.3 and RFC 8414 section 3.3 checks on the fetched metadata."""

    def test_expected_resource_strips_the_well_known_segment(self):
        assert oauth_engine._expected_resource(DCR_TEST_PRM_URL) == DCR_TEST_RESOURCE

    def test_expected_resource_for_a_host_level_resource(self):
        url = "https://mcp.example.com/.well-known/oauth-protected-resource"
        assert oauth_engine._expected_resource(url) == "https://mcp.example.com"

    def test_expected_resource_rejects_a_url_without_the_segment(self):
        with pytest.raises(oauth_engine.OAuthEngineError, match="not an RFC 9728 URL"):
            oauth_engine._expected_resource("https://mcp.example.com/metadata")

    def test_expected_resource_rejects_a_longer_segment(self):
        url = "https://mcp.example.com/.well-known/oauth-protected-resourceX/v2"
        with pytest.raises(oauth_engine.OAuthEngineError, match="not an RFC 9728 URL"):
            oauth_engine._expected_resource(url)

    async def test_wrong_resource_is_rejected(self, monkeypatch):
        _fake_discovery(monkeypatch, prm=_prm(resource="https://evil.example.com/mcp"))
        with pytest.raises(oauth_engine.OAuthEngineError, match="different resource"):
            await oauth_engine._discover_registration_url(make_dcr_test_config())

    async def test_missing_resource_is_rejected(self, monkeypatch):
        prm = _prm()
        del prm["resource"]
        _fake_discovery(monkeypatch, prm=prm)
        with pytest.raises(oauth_engine.OAuthEngineError, match="different resource"):
            await oauth_engine._discover_registration_url(make_dcr_test_config())

    async def test_prefetched_prm_with_wrong_resource_is_rejected(self, monkeypatch):
        _fake_discovery(monkeypatch)
        with pytest.raises(oauth_engine.OAuthEngineError, match="different resource"):
            await oauth_engine._discover_registration_url(
                make_dcr_test_config(), prm=_prm(resource="https://evil.example.com/mcp")
            )

    async def test_wrong_issuer_is_rejected(self, monkeypatch):
        _fake_discovery(monkeypatch, as_meta=_as_meta(issuer="https://auth.example.com/OTHER"))
        with pytest.raises(oauth_engine.OAuthEngineError, match="different issuer"):
            await oauth_engine._discover_registration_url(make_dcr_test_config())

    @pytest.mark.parametrize(
        ("token_url", "evil_issuer"),
        [
            pytest.param(_DCR_TEST_TOKEN_URL, "https://evil.example", id="other-host"),
            pytest.param(
                _DCR_TEST_TOKEN_URL,
                "https://auth.example.com.evil.example/TENANT",
                id="suffix-host",
            ),
            pytest.param(
                _DCR_TEST_TOKEN_URL, "https://auth.example.com:8443/TENANT", id="other-port"
            ),
            pytest.param(_DCR_TEST_TOKEN_URL, "http://auth.example.com/TENANT", id="other-scheme"),
            # Edge cases against the Atlassian token origin. urlparse reads the host
            # of each one as something other than auth.atlassian.com.
            pytest.param(
                _ATLASSIAN_TOKEN_URL,
                "https://auth.atlassian.com@evil.example",
                id="userinfo-names-the-origin",
            ),
            pytest.param(
                _ATLASSIAN_TOKEN_URL,
                "https://auth.atlassian.com./x",
                id="trailing-dot",
            ),
            pytest.param(
                _ATLASSIAN_TOKEN_URL,
                "https://\u0430uth.atlassian.com/x",
                id="idn-cyrillic-a",
            ),
        ],
    )
    async def test_issuer_on_another_origin_than_token_url_is_rejected(
        self, monkeypatch, token_url, evil_issuer
    ):
        # Spoofed protected-resource metadata names an authorization server of its
        # own that is internally consistent (issuer and registration endpoint match
        # the URL it is served from), so only the token_url anchor can refuse it.
        seen: list[str] = []
        _fake_discovery(
            monkeypatch,
            prm=_prm(authorization_servers=[evil_issuer]),
            as_meta=_as_meta(
                issuer=evil_issuer, registration_endpoint=f"{evil_issuer}/dcr/register"
            ),
            seen=seen,
        )
        with pytest.raises(oauth_engine.OAuthEngineError, match="not on the provider's host"):
            await oauth_engine._discover_registration_url(make_dcr_test_config(token_url=token_url))
        # The authorization server's metadata is never fetched.
        assert seen == [DCR_TEST_PRM_URL]

    async def test_issuer_on_another_origin_registers_nothing(self, monkeypatch):
        _fake_discovery(
            monkeypatch,
            prm=_prm(authorization_servers=["https://evil.example"]),
            as_meta=_as_meta(
                issuer="https://evil.example",
                registration_endpoint="https://evil.example/dcr/register",
            ),
        )
        posted: list[str] = []

        async def fake_post_dcr(url, body):
            posted.append(url)
            return {"client_id": "X"}

        monkeypatch.setattr(oauth_engine, "_post_dcr", fake_post_dcr)
        with pytest.raises(oauth_engine.OAuthEngineError, match="not on the provider's host"):
            await oauth_engine.register_dcr_client(
                make_dcr_test_config(), "https://gw/oauth2/egress/callback", ["read:account"]
            )
        assert posted == []

    async def test_prefetched_prm_naming_another_origin_is_rejected(self, monkeypatch):
        _fake_discovery(monkeypatch)
        with pytest.raises(oauth_engine.OAuthEngineError, match="not on the provider's host"):
            await oauth_engine._discover_registration_url(
                make_dcr_test_config(), prm=_prm(authorization_servers=["https://evil.example"])
            )

    async def test_issuer_on_the_token_url_origin_with_default_port_passes(self, monkeypatch):
        _fake_discovery(monkeypatch)
        cfg = make_dcr_test_config(token_url="https://AUTH.example.com:443/oauth/token")
        assert (
            await oauth_engine._discover_registration_url(cfg) == f"{DCR_TEST_ISSUER}/dcr/register"
        )

    async def test_upper_case_issuer_host_is_the_same_origin(self, monkeypatch):
        # urlparse lowercases the scheme and host, so an issuer written in upper
        # case is the token_url origin and passes, like the default-port case above.
        issuer = "HTTPS://AUTH.ATLASSIAN.COM/x"
        _fake_discovery(
            monkeypatch,
            prm=_prm(authorization_servers=[issuer]),
            as_meta=_as_meta(issuer=issuer, registration_endpoint=f"{issuer}/dcr/register"),
        )
        cfg = make_dcr_test_config(token_url=_ATLASSIAN_TOKEN_URL)
        assert await oauth_engine._discover_registration_url(cfg) == f"{issuer}/dcr/register"

    async def test_provider_without_token_url_keeps_the_other_checks_only(self, monkeypatch):
        _fake_discovery(
            monkeypatch,
            prm=_prm(authorization_servers=["https://other.example/TENANT"]),
            as_meta=_as_meta(
                issuer="https://other.example/TENANT",
                registration_endpoint="https://other.example/TENANT/dcr/register",
            ),
        )
        cfg = make_dcr_test_config(token_url="")
        assert (
            await oauth_engine._discover_registration_url(cfg)
            == "https://other.example/TENANT/dcr/register"
        )

    async def test_registration_endpoint_on_another_host_is_rejected(self, monkeypatch):
        _fake_discovery(
            monkeypatch,
            as_meta=_as_meta(registration_endpoint="https://evil.example.net/dcr/register"),
        )
        with pytest.raises(oauth_engine.OAuthEngineError, match="not on the issuer's host"):
            await oauth_engine._discover_registration_url(make_dcr_test_config())

    async def test_registration_endpoint_on_another_port_is_rejected(self, monkeypatch):
        _fake_discovery(
            monkeypatch,
            as_meta=_as_meta(registration_endpoint="https://auth.example.com:8443/dcr/register"),
        )
        with pytest.raises(oauth_engine.OAuthEngineError, match="not on the issuer's host"):
            await oauth_engine._discover_registration_url(make_dcr_test_config())

    async def test_non_string_authorization_server_is_rejected(self, monkeypatch):
        _fake_discovery(monkeypatch, prm=_prm(authorization_servers=[{"url": "x"}]))
        with pytest.raises(oauth_engine.OAuthEngineError, match="invalid authorization server"):
            await oauth_engine._discover_registration_url(make_dcr_test_config())

    async def test_atlassian_shaped_metadata_passes_every_check(self, monkeypatch):
        # Values served live by Atlassian's Rovo MCP v2 endpoint and its
        # authorization server.
        issuer = "https://auth.atlassian.com/VCeDsk8ZHncYF1g234fKtc4lNipbBhu3"
        cfg = make_dcr_test_config(
            protected_resource_metadata_url=(
                "https://mcp.atlassian.com/.well-known/oauth-protected-resource/v2/mcp"
            ),
            token_url="https://auth.atlassian.com/oauth/token",
        )
        rec: dict = {}
        monkeypatch.setattr(
            oauth_engine,
            "shared_guarded_async_client",
            _fake_guarded_sequence(
                [
                    _FakeResp(
                        {
                            "resource": "https://mcp.atlassian.com/v2/mcp",
                            "authorization_servers": [issuer],
                        }
                    ),
                    _FakeResp(
                        {"issuer": issuer, "registration_endpoint": f"{issuer}/dcr/register"}
                    ),
                ],
                rec,
            ),
        )
        url = await oauth_engine._discover_registration_url(cfg)
        assert url == f"{issuer}/dcr/register"
        assert rec["get_urls"][1] == f"{issuer}/.well-known/oauth-authorization-server"


@pytest.mark.unit
class TestDcrTransport:
    """The real _get_json / _post_dcr bodies: status, JSON shape, transport, SSRF guard."""

    async def test_get_json_happy_path(self, monkeypatch):
        monkeypatch.setattr(
            oauth_engine,
            "shared_guarded_async_client",
            _fake_guarded_sequence([_FakeResp({"a": 1})]),
        )
        assert await oauth_engine._get_json("https://as/meta") == {"a": 1}

    async def test_get_json_non_2xx_raises(self, monkeypatch):
        monkeypatch.setattr(
            oauth_engine,
            "shared_guarded_async_client",
            _fake_guarded_sequence([_FakeResp({"resource": "x"}, status_code=404)]),
        )
        with pytest.raises(oauth_engine.OAuthEngineError, match="returned http 404"):
            await oauth_engine._get_json("https://as/meta")

    async def test_get_json_non_json_raises(self, monkeypatch):
        monkeypatch.setattr(
            oauth_engine,
            "shared_guarded_async_client",
            _fake_guarded_sequence([_FakeResp(None, non_json=True)]),
        )
        with pytest.raises(oauth_engine.OAuthEngineError, match="non-JSON"):
            await oauth_engine._get_json("https://as/meta")

    async def test_get_json_non_object_raises(self, monkeypatch):
        # A JSON list used to reach dict.get and surface as an AttributeError/500.
        monkeypatch.setattr(
            oauth_engine,
            "shared_guarded_async_client",
            _fake_guarded_sequence([_FakeResp(["not", "an", "object"])]),
        )
        with pytest.raises(oauth_engine.OAuthEngineError, match="not an object"):
            await oauth_engine._get_json("https://as/meta")

    async def test_get_json_ssrf_guard_fails_closed(self):
        # A metadata-IP discovery URL must be blocked by the real guarded client,
        # and the guard's own text must not reach the raised message.
        with pytest.raises(oauth_engine.OAuthEngineError) as exc_info:
            await oauth_engine._get_json("http://169.254.169.254/latest/meta-data/")
        assert str(exc_info.value) == "discovery endpoint blocked by security policy"

    async def test_get_json_http_error_message_is_fixed(self, monkeypatch):
        monkeypatch.setattr(
            oauth_engine,
            "shared_guarded_async_client",
            _fake_guarded_raising(httpx.ConnectError("boom 10.0.0.7:443")),
        )
        with pytest.raises(oauth_engine.OAuthEngineError) as exc_info:
            await oauth_engine._get_json("https://as/meta")
        assert str(exc_info.value) == "discovery endpoint unreachable"

    async def test_post_dcr_happy_path(self, monkeypatch):
        rec: dict = {}
        monkeypatch.setattr(
            oauth_engine,
            "shared_guarded_async_client",
            _fake_guarded_sequence([_FakeResp({"client_id": "X"}, status_code=201)], rec),
        )
        out = await oauth_engine._post_dcr("https://as/register", {"client_name": "n"})
        assert out == {"client_id": "X"}
        assert rec["post_json"] == {"client_name": "n"}

    async def test_post_dcr_rfc7591_error_code_is_passed_on(self, monkeypatch):
        monkeypatch.setattr(
            oauth_engine,
            "shared_guarded_async_client",
            _fake_guarded_sequence([_FakeResp({"error": "invalid_redirect_uri"}, status_code=400)]),
        )
        with pytest.raises(oauth_engine.OAuthEngineError, match="invalid_redirect_uri"):
            await oauth_engine._post_dcr("https://as/register", {})

    async def test_post_dcr_unknown_error_text_is_not_passed_on(self, monkeypatch):
        reply = {"error": "<script>steer me</script>", "error_description": "free text"}
        monkeypatch.setattr(
            oauth_engine,
            "shared_guarded_async_client",
            _fake_guarded_sequence([_FakeResp(reply, status_code=400)]),
        )
        with pytest.raises(oauth_engine.OAuthEngineError) as exc_info:
            await oauth_engine._post_dcr("https://as/register", {})
        assert str(exc_info.value) == "registration rejected (http 400)"

    async def test_post_dcr_non_json_raises(self, monkeypatch):
        monkeypatch.setattr(
            oauth_engine,
            "shared_guarded_async_client",
            _fake_guarded_sequence([_FakeResp(None, non_json=True)]),
        )
        with pytest.raises(oauth_engine.OAuthEngineError, match="non-JSON"):
            await oauth_engine._post_dcr("https://as/register", {})

    async def test_post_dcr_non_object_raises(self, monkeypatch):
        monkeypatch.setattr(
            oauth_engine,
            "shared_guarded_async_client",
            _fake_guarded_sequence([_FakeResp(["client_id", "X"], status_code=201)]),
        )
        with pytest.raises(oauth_engine.OAuthEngineError, match="not an object"):
            await oauth_engine._post_dcr("https://as/register", {})

    async def test_post_dcr_ssrf_guard_fails_closed(self):
        with pytest.raises(oauth_engine.OAuthEngineError) as exc_info:
            await oauth_engine._post_dcr("http://169.254.169.254/register", {})
        assert str(exc_info.value) == "registration endpoint blocked by security policy"

    async def test_post_dcr_http_error_message_is_fixed(self, monkeypatch):
        monkeypatch.setattr(
            oauth_engine,
            "shared_guarded_async_client",
            _fake_guarded_raising(httpx.ConnectError("boom 10.0.0.7:443")),
        )
        with pytest.raises(oauth_engine.OAuthEngineError) as exc_info:
            await oauth_engine._post_dcr("https://as/register", {})
        assert str(exc_info.value) == "registration endpoint unreachable"


@pytest.mark.unit
class TestScopeValidation:
    """validate_scopes_against_prm and fetch_protected_resource_metadata."""

    _SUPPORTED = ["read:me", "read:account", "offline_access", "read:work"]

    def test_all_supported_returns_empty(self):
        prm = {"scopes_supported": self._SUPPORTED}
        assert oauth_engine.validate_scopes_against_prm(["read:me", "read:account"], prm) == []

    def test_unsupported_scope_returned(self):
        prm = {"scopes_supported": self._SUPPORTED}
        bad = oauth_engine.validate_scopes_against_prm(
            ["read:me", "read:legacy", "write:legacy"], prm
        )
        assert set(bad) == {"read:legacy", "write:legacy"}

    def test_empty_scopes_always_valid(self):
        prm = {"scopes_supported": self._SUPPORTED}
        assert oauth_engine.validate_scopes_against_prm([], prm) == []

    def test_no_scopes_supported_skips_validation(self):
        # A PRM without scopes_supported must not reject anything.
        assert oauth_engine.validate_scopes_against_prm(["anything"], {}) == []
        assert (
            oauth_engine.validate_scopes_against_prm(["anything"], {"scopes_supported": []}) == []
        )

    async def test_fetch_prm_returns_document(self, monkeypatch):
        async def fake_get_json(url):
            assert url == DCR_TEST_PRM_URL
            return _prm(scopes_supported=self._SUPPORTED)

        monkeypatch.setattr(oauth_engine, "_get_json", fake_get_json)
        prm = await oauth_engine.fetch_protected_resource_metadata(make_dcr_test_config())
        assert prm["scopes_supported"] == self._SUPPORTED

    async def test_fetch_prm_checks_resource(self, monkeypatch):
        async def fake_get_json(url):
            return _prm(resource="https://evil.example.com/mcp")

        monkeypatch.setattr(oauth_engine, "_get_json", fake_get_json)
        with pytest.raises(oauth_engine.OAuthEngineError, match="different resource"):
            await oauth_engine.fetch_protected_resource_metadata(make_dcr_test_config())

    async def test_fetch_prm_no_url_raises(self):
        cfg = OAuthProviderConfig(
            name="x",
            display_name="X",
            authorize_url="https://x/a",
            token_url="https://x/t",
        )
        with pytest.raises(oauth_engine.OAuthEngineError, match="protected_resource_metadata_url"):
            await oauth_engine.fetch_protected_resource_metadata(cfg)

    async def test_discovery_uses_provided_prm_without_refetch(self, monkeypatch):
        # When a pre-fetched PRM is passed, _get_json must NOT be called for the
        # PRM URL, only for the AS metadata URL.
        fetched: list[str] = []

        async def fake_get_json(url):
            fetched.append(url)
            if "oauth-authorization-server" in url:
                return _as_meta()
            raise AssertionError(f"unexpected URL fetched: {url}")

        monkeypatch.setattr(oauth_engine, "_get_json", fake_get_json)
        prm = _prm(scopes_supported=["read:me"])
        reg_url = await oauth_engine._discover_registration_url(make_dcr_test_config(), prm=prm)
        assert reg_url == f"{DCR_TEST_ISSUER}/dcr/register"
        # Only the AS metadata URL was fetched; the PRM URL was not re-fetched.
        assert fetched == [f"{DCR_TEST_ISSUER}/.well-known/oauth-authorization-server"]


_ROVO_ISSUER = "https://auth.atlassian.com/VCeDsk8ZHncYF1g234fKtc4lNipbBhu3"


@pytest.mark.unit
class TestAtlassianRovoDcr:
    """The atlassian-rovo provider through the real discovery and registration code."""

    def _metadata_replies(self):
        return [
            _FakeResp(
                {
                    "resource": "https://mcp.atlassian.com/v2/mcp",
                    "authorization_servers": [_ROVO_ISSUER],
                    "scopes_supported": PROVIDER_REGISTRY["atlassian-rovo"].default_scopes,
                }
            ),
            _FakeResp(
                {"issuer": _ROVO_ISSUER, "registration_endpoint": f"{_ROVO_ISSUER}/dcr/register"}
            ),
        ]

    async def test_registers_a_public_client_at_the_discovered_endpoint(self, monkeypatch):
        rec: dict = {}
        replies = [*self._metadata_replies(), _FakeResp({"client_id": "DCRID"}, status_code=201)]
        monkeypatch.setattr(
            oauth_engine, "shared_guarded_async_client", _fake_guarded_sequence(replies, rec)
        )
        rovo = PROVIDER_REGISTRY["atlassian-rovo"]
        cid, secret = await oauth_engine.register_dcr_client(
            rovo, "https://gw/oauth2/egress/callback", rovo.default_scopes
        )
        assert (cid, secret) == ("DCRID", None)
        assert rec["get_urls"][0] == (
            "https://mcp.atlassian.com/.well-known/oauth-protected-resource/v2/mcp"
        )
        assert rec["post_url"] == f"{_ROVO_ISSUER}/dcr/register"
        assert rec["post_json"]["token_endpoint_auth_method"] == "none"
        assert rec["post_json"]["scope"] == " ".join(rovo.default_scopes)

    async def test_prm_for_v1_authv2_is_rejected_for_v2(self, monkeypatch):
        # Metadata naming another resource (here the earlier v1 authv2 endpoint)
        # must not steer the v2 registration.
        replies = self._metadata_replies()
        replies[0] = _FakeResp(
            {
                "resource": "https://mcp.atlassian.com/v1/mcp/authv2",
                "authorization_servers": [_ROVO_ISSUER],
            }
        )
        monkeypatch.setattr(
            oauth_engine, "shared_guarded_async_client", _fake_guarded_sequence(replies)
        )
        with pytest.raises(oauth_engine.OAuthEngineError, match="different resource"):
            await oauth_engine._discover_registration_url(PROVIDER_REGISTRY["atlassian-rovo"])

    def test_authorize_url_has_no_audience_or_resource(self):
        url = oauth_engine.build_authorize_url(
            PROVIDER_REGISTRY["atlassian-rovo"],
            client_id="DCRID",
            redirect_uri="https://gw/oauth2/egress/callback",
            scopes=["read:jira:agent-interface", "read:account"],
            state="S",
            pkce_challenge="CHAL",
        )
        q = parse_qs(urlparse(url).query)
        assert "audience" not in q
        assert "resource" not in q
        assert q["code_challenge_method"] == ["S256"]
        assert q["prompt"] == ["consent"]
        assert q["scope"] == ["read:jira:agent-interface read:account"]

    def test_classic_atlassian_authorize_url_keeps_the_audience(self):
        url = oauth_engine.build_authorize_url(
            PROVIDER_REGISTRY["atlassian"],
            client_id="CLASSIC",
            redirect_uri="https://gw/oauth2/egress/callback",
            scopes=["read:jira-work", "offline_access"],
            state="S",
        )
        q = parse_qs(urlparse(url).query)
        assert q["audience"] == ["api.atlassian.com"]
        assert q["prompt"] == ["consent"]


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
        from unittest.mock import AsyncMock, MagicMock

        captured = {}

        def fake_shared(*, profile, verify=True):
            captured["profile"] = profile
            captured["verify"] = verify
            response = MagicMock(status_code=200)
            response.json.return_value = {"access_token": "ok"}
            client = MagicMock()
            client.post = AsyncMock(return_value=response)
            return client

        monkeypatch.setattr(oauth_engine, "shared_guarded_async_client", fake_shared)
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
        from unittest.mock import AsyncMock, MagicMock

        # The guarded transport raises UrlValidationError at request time (on the
        # pinned .post), before any credential is sent.
        def blocked_shared(*, profile, verify=True):
            del profile, verify
            client = MagicMock()
            client.post = AsyncMock(
                side_effect=oauth_engine.UrlValidationError(
                    "https://tokens.example/token?api_key=query-secret",
                    "raw-exception-secret",
                )
            )
            return client

        monkeypatch.setattr(oauth_engine, "shared_guarded_async_client", blocked_shared)
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
        from unittest.mock import AsyncMock, MagicMock

        def failing_shared(*, profile, verify=True):
            del profile, verify
            client = MagicMock()
            client.post = AsyncMock(
                side_effect=oauth_engine.httpx.ConnectError("raw-exception-secret")
            )
            return client

        monkeypatch.setattr(oauth_engine, "shared_guarded_async_client", failing_shared)
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
