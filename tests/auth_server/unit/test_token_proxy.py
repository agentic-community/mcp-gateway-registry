"""auth_server core: token-proxy + IdP-signed-token unit tests.

Covers the new /oauth/token proxy route, the /validate self-signed retirement
chokepoint, and the pure helpers behind the conditional per-server audience and
exp-ceiling enforcement. Flag-OFF behavior (the default) is exercised by the
rest of the auth_server suite; these tests target the flag-ON contracts.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import auth_server.server as srv
from auth_server.server import (
    _enforce_exp_ceiling,
    _is_mcp_data_plane,
    _per_server_resources,
    _token_audiences,
)

# Unverified decodes only care that the JWT is well-formed; any secret works.
_SECRET = "x" * 40


class TestTokenAudiences:
    def test_string_aud(self):
        tok = jwt.encode({"aud": "https://gw/s/mcp"}, _SECRET, algorithm="HS256")
        assert _token_audiences(tok) == {"https://gw/s/mcp"}

    def test_list_aud(self):
        tok = jwt.encode({"aud": ["a", "b"]}, _SECRET, algorithm="HS256")
        assert _token_audiences(tok) == {"a", "b"}

    def test_missing_aud(self):
        tok = jwt.encode({"sub": "x"}, _SECRET, algorithm="HS256")
        assert _token_audiences(tok) == set()

    def test_garbage_token(self):
        assert _token_audiences("not-a-jwt") == set()


class TestExpCeiling:
    def _tok(self, iat, exp):
        return jwt.encode({"iat": iat, "exp": exp}, _SECRET, algorithm="HS256")

    def test_within_ceiling_ok(self):
        now = int(time.time())
        _enforce_exp_ceiling(self._tok(now, now + 3600), 2)  # 1h <= 2h

    def test_exceeds_ceiling_rejected(self):
        now = int(time.time())
        with pytest.raises(HTTPException) as ei:
            _enforce_exp_ceiling(self._tok(now, now + 4 * 3600), 2)  # 4h > 2h
        assert ei.value.status_code == 401

    def test_missing_iat_skips(self):
        now = int(time.time())
        tok = jwt.encode({"exp": now + 10 * 3600}, _SECRET, algorithm="HS256")
        _enforce_exp_ceiling(tok, 1)  # no iat -> skip, no raise

    def test_no_ceiling_skips(self):
        now = int(time.time())
        _enforce_exp_ceiling(self._tok(now, now + 10 * 3600), None)


class TestIsMcpDataPlane:
    def test_mcp_server_path_is_data_plane(self):
        assert _is_mcp_data_plane("https://gw/test-server/mcp", "test-server") is True

    def test_no_server_context_is_not_data_plane(self):
        assert _is_mcp_data_plane("https://gw/", None) is False

    def test_registry_api_is_not_data_plane(self):
        # /api requests never parse a server_name; guard the None path explicitly.
        assert _is_mcp_data_plane("https://gw/api/servers", None) is False


class TestPerServerResources:
    def test_derives_path_bound_resources(self):
        with patch.dict("os.environ", {"REGISTRY_URL": "https://gw.example.com"}, clear=False):
            auds = _per_server_resources("myserver/mcp")
        assert auds, "expected at least one per-server resource"
        assert any("myserver" in a for a in auds)
        assert any("gw.example.com" in a for a in auds)

    def test_no_server_returns_empty(self):
        assert _per_server_resources(None) == []


class TestBootInterlock:
    def test_idp_signed_requires_proxy_enabled(self):
        from types import SimpleNamespace

        from registry.core.config import Settings

        bad = SimpleNamespace(mcp_idp_signed_tokens=True, mcp_token_proxy_enabled=False)
        with pytest.raises(ValueError, match="MCP_TOKEN_PROXY_ENABLED"):
            Settings._validate_token_proxy_config(bad)

    def test_both_off_ok(self):
        from types import SimpleNamespace

        from registry.core.config import Settings

        ok = SimpleNamespace(mcp_idp_signed_tokens=False, mcp_token_proxy_enabled=False)
        Settings._validate_token_proxy_config(ok)  # no raise


class TestOauthTokenProxy:
    def test_disabled_returns_404(self, auth_env_vars):
        with patch.object(srv.settings, "mcp_token_proxy_enabled", False):
            client = TestClient(srv.app)
            r = client.post("/oauth/token", data={"grant_type": "authorization_code"})
        assert r.status_code == 404

    def test_unsupported_grant_rejected(self, auth_env_vars):
        with patch.object(srv.settings, "mcp_token_proxy_enabled", True):
            client = TestClient(srv.app)
            r = client.post("/oauth/token", data={"grant_type": "password"})
        assert r.status_code == 400
        assert r.json()["error"] == "unsupported_grant_type"

    def test_resource_required_when_idp_signed(self, auth_env_vars):
        with (
            patch.object(srv.settings, "mcp_token_proxy_enabled", True),
            patch.object(srv.settings, "mcp_idp_signed_tokens", True),
        ):
            client = TestClient(srv.app)
            r = client.post("/oauth/token", data={"grant_type": "authorization_code"})
        assert r.status_code == 400
        assert r.json()["error"] == "invalid_target"

    def test_happy_path_proxies_and_strips_non_token_fields(self, auth_env_vars):
        provider = MagicMock()
        provider.token_url = "https://idp.example.com/token"
        idp_resp = MagicMock()
        idp_resp.status_code = 200
        idp_resp.json.return_value = {
            "access_token": "AT",
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "RT",
            "id_token": "IDT-should-be-stripped",
            "unexpected": "drop-me",
        }
        client_cm = AsyncMock()
        client_cm.__aenter__.return_value.post = AsyncMock(return_value=idp_resp)
        with (
            patch.object(srv.settings, "mcp_token_proxy_enabled", True),
            patch.object(srv.settings, "mcp_idp_signed_tokens", False),  # resource lenient
            patch("auth_server.server.get_auth_provider", return_value=provider),
            patch("registry.utils.url_guard.validate_url"),
            patch("registry.utils.url_guard.guarded_async_client", return_value=client_cm),
        ):
            client = TestClient(srv.app)
            r = client.post(
                "/oauth/token",
                data={"grant_type": "refresh_token", "refresh_token": "R"},
            )
        assert r.status_code == 200
        body = r.json()
        assert body["access_token"] == "AT"
        assert body["refresh_token"] == "RT"
        # Response hygiene: id_token never surfaced as a token field; unknowns dropped.
        assert "id_token" not in body
        assert "unexpected" not in body


class TestSelfSignedRetirement:
    @patch("auth_server.server.get_auth_provider")
    def test_self_signed_rejected_on_data_plane_when_idp_signed(
        self,
        mock_get_provider,
        mock_cognito_provider,
        auth_env_vars,
        mock_scope_repository_with_data,
    ):
        """A self-signed (iss=mcp-auth-server) token is rejected on an MCP
        data-plane path once MCP_IDP_SIGNED_TOKENS is on -- at the single
        chokepoint, before any provider validation."""
        mock_get_provider.return_value = mock_cognito_provider
        tok = jwt.encode({"iss": srv.JWT_ISSUER, "sub": "u"}, _SECRET, algorithm="HS256")
        with (
            patch.object(srv.settings, "mcp_idp_signed_tokens", True),
            patch(
                "auth_server.server.get_scope_repository",
                return_value=mock_scope_repository_with_data,
            ),
        ):
            client = TestClient(srv.app)
            r = client.get(
                "/validate",
                headers={
                    "Authorization": f"Bearer {tok}",
                    "X-Original-URL": "https://example.com/test-server/mcp",
                },
            )
        assert r.status_code == 401
