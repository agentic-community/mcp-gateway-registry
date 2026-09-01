"""Unit tests for the registry-side registry-UI internal-token verifier.

Covers the verify half of the /validate-minted ``mcp-registry-ui`` token: the
auth-server mints it (see tests/unit/auth/test_internal_request_token.py); the
registry verifies it here and reads identity from the verified claims, ignoring
the forgeable inbound headers.
"""

import os
import time
from unittest.mock import patch

import jwt as pyjwt
import pytest
from fastapi import HTTPException

from registry.auth.proxied_token import (
    _api_auth_request_enabled,
    verify_generic_proxy_token,
    verify_registry_ui_token,
)

_SECRET = "test-secret-key-for-testing-only"
_ISSUER = "mcp-auth-server"
_AUDIENCE = "mcp-registry-ui"


def _make_token(
    *,
    secret: str = _SECRET,
    issuer: str = _ISSUER,
    audience: str = _AUDIENCE,
    sub: str = "alice",
    token_use: str = "mcp-registry-ui",
    session_id: str = "sess-1",
    groups: list[str] | None = None,
    auth_method: str = "keycloak",
    client_id: str = "ui",
    iat_offset: int = 0,
    exp_offset: int = 30,
) -> str:
    now = int(time.time())
    claims = {
        "iss": issuer,
        "aud": audience,
        "sub": sub,
        "scopes": [],
        "session_id": session_id,
        "groups": groups or [],
        "auth_method": auth_method,
        "client_id": client_id,
        "token_use": token_use,
        "iat": now + iat_offset,
        "exp": now + exp_offset,
    }
    return pyjwt.encode(claims, secret, algorithm="HS256")


@pytest.fixture(autouse=True)
def _secret_env():
    with patch.dict(os.environ, {"SECRET_KEY": _SECRET}, clear=False):
        yield


class TestVerifyRegistryUiToken:
    def test_valid_token_returns_claims(self) -> None:
        token = _make_token(sub="alice", session_id="sess-1", groups=["g1"])
        claims = verify_registry_ui_token(token)
        assert claims["sub"] == "alice"
        assert claims["session_id"] == "sess-1"
        assert claims["groups"] == ["g1"]
        assert claims["auth_method"] == "keycloak"
        assert claims["client_id"] == "ui"

    def test_garbage_token_rejected(self) -> None:
        with pytest.raises(HTTPException) as exc:
            verify_registry_ui_token("not-a-jwt")
        assert exc.value.status_code == 401

    def test_expired_token_rejected(self) -> None:
        # exp well in the past, beyond the 5s leeway.
        token = _make_token(iat_offset=-120, exp_offset=-60)
        with pytest.raises(HTTPException) as exc:
            verify_registry_ui_token(token)
        assert exc.value.status_code == 401

    def test_future_iat_rejected(self) -> None:
        # iat far in the future, beyond leeway.
        token = _make_token(iat_offset=120, exp_offset=180)
        with pytest.raises(HTTPException) as exc:
            verify_registry_ui_token(token)
        assert exc.value.status_code == 401

    def test_wrong_audience_rejected(self) -> None:
        # An mcp-proxy token must not verify as registry-ui.
        token = _make_token(audience="mcp-proxy")
        with pytest.raises(HTTPException) as exc:
            verify_registry_ui_token(token)
        assert exc.value.status_code == 401

    def test_service_audience_rejected(self) -> None:
        # The mcp-registry service-to-service audience must not verify here either.
        token = _make_token(audience="mcp-registry")
        with pytest.raises(HTTPException) as exc:
            verify_registry_ui_token(token)
        assert exc.value.status_code == 401

    def test_wrong_issuer_rejected(self) -> None:
        token = _make_token(issuer="someone-else")
        with pytest.raises(HTTPException) as exc:
            verify_registry_ui_token(token)
        assert exc.value.status_code == 401

    def test_wrong_token_use_rejected(self) -> None:
        token = _make_token(token_use="access")
        with pytest.raises(HTTPException) as exc:
            verify_registry_ui_token(token)
        assert exc.value.status_code == 401

    def test_tampered_signature_rejected(self) -> None:
        # Signed with a different key.
        token = _make_token(secret="a-different-secret-key-entirely")
        with pytest.raises(HTTPException) as exc:
            verify_registry_ui_token(token)
        assert exc.value.status_code == 401

    def test_missing_secret_raises_500(self) -> None:
        token = _make_token()
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(HTTPException) as exc:
                verify_registry_ui_token(token)
            assert exc.value.status_code == 500


class TestApiAuthRequestEnabled:
    def test_default_enabled(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert _api_auth_request_enabled() is True

    @pytest.mark.parametrize("val", ["true", "1", "yes", "on", "TRUE", "On"])
    def test_disabled_values(self, val: str) -> None:
        with patch.dict(os.environ, {"NGINX_DISABLE_API_AUTH_REQUEST": val}, clear=False):
            assert _api_auth_request_enabled() is False

    @pytest.mark.parametrize("val", ["false", "0", "no", "off", ""])
    def test_enabled_values(self, val: str) -> None:
        with patch.dict(os.environ, {"NGINX_DISABLE_API_AUTH_REQUEST": val}, clear=False):
            assert _api_auth_request_enabled() is True


_GENERIC_PROXY_AUDIENCE = "generic-proxy"
_GENERIC_PROXY_TOKEN_USE = "generic-proxy"


def _make_generic_token(
    *,
    secret: str = _SECRET,
    issuer: str = _ISSUER,
    audience: str = _GENERIC_PROXY_AUDIENCE,
    token_use: str = _GENERIC_PROXY_TOKEN_USE,
    sub: str = "svc-generic",
    entity_type: str = "skill",
    server: str = "/skills/weather",
    upstream_url: str = "https://upstream.example.com/api",
    streaming: bool = False,
    has_upstream_auth: bool = True,
    iat_offset: int = 0,
    exp_offset: int = 30,
    drop: str | None = None,
) -> str:
    now = int(time.time())
    claims = {
        "iss": issuer,
        "aud": audience,
        "sub": sub,
        "scopes": [],
        "entity_type": entity_type,
        "server": server,
        "upstream_url": upstream_url,
        "streaming": streaming,
        "has_upstream_auth": has_upstream_auth,
        "token_use": token_use,
        "iat": now + iat_offset,
        "exp": now + exp_offset,
    }
    if drop is not None:
        claims.pop(drop, None)
    return pyjwt.encode(claims, secret, algorithm="HS256")


class TestVerifyGenericProxyToken:
    def test_valid_token_returns_claims(self) -> None:
        token = _make_generic_token(
            entity_type="skill",
            server="/skills/weather",
            upstream_url="https://upstream.example.com/api",
        )
        claims = verify_generic_proxy_token(token)
        assert claims["entity_type"] == "skill"
        assert claims["server"] == "/skills/weather"
        assert claims["upstream_url"] == "https://upstream.example.com/api"
        assert claims["sub"] == "svc-generic"

    def test_missing_secret_raises_500(self) -> None:
        token = _make_generic_token()
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(HTTPException) as exc:
                verify_generic_proxy_token(token)
            assert exc.value.status_code == 500

    def test_garbage_token_rejected(self) -> None:
        with pytest.raises(HTTPException) as exc:
            verify_generic_proxy_token("not-a-jwt")
        assert exc.value.status_code == 401

    def test_tampered_signature_rejected(self) -> None:
        token = _make_generic_token(secret="a-different-secret-key-entirely")
        with pytest.raises(HTTPException) as exc:
            verify_generic_proxy_token(token)
        assert exc.value.status_code == 401

    def test_expired_token_rejected(self) -> None:
        token = _make_generic_token(iat_offset=-120, exp_offset=-60)
        with pytest.raises(HTTPException) as exc:
            verify_generic_proxy_token(token)
        assert exc.value.status_code == 401

    def test_wrong_audience_rejected(self) -> None:
        token = _make_generic_token(audience="mcp-proxy")
        with pytest.raises(HTTPException) as exc:
            verify_generic_proxy_token(token)
        assert exc.value.status_code == 401

    def test_wrong_issuer_rejected(self) -> None:
        token = _make_generic_token(issuer="someone-else")
        with pytest.raises(HTTPException) as exc:
            verify_generic_proxy_token(token)
        assert exc.value.status_code == 401

    def test_wrong_token_use_rejected(self) -> None:
        # Correct generic-proxy audience but access token_use must fail closed.
        token = _make_generic_token(token_use="access")
        with pytest.raises(HTTPException) as exc:
            verify_generic_proxy_token(token)
        assert exc.value.status_code == 401

    @pytest.mark.parametrize("field", ["entity_type", "server", "upstream_url"])
    def test_missing_binding_rejected(self, field: str) -> None:
        token = _make_generic_token(drop=field)
        with pytest.raises(HTTPException) as exc:
            verify_generic_proxy_token(token)
        assert exc.value.status_code == 401
