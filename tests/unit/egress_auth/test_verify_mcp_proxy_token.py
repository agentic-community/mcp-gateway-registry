"""Tests for registry-side verify_mcp_proxy_token.

Mints mcp-proxy tokens with the shared SECRET_KEY (HS256, the same contract
auth_server uses) and asserts the registry verifier accepts valid ones and
rejects every failure mode: wrong audience, wrong token_use, tampered,
missing upstream binding.
"""

import time

import jwt as pyjwt
import pytest
from fastapi import HTTPException

from registry.auth.proxied_token import verify_mcp_proxy_token

SECRET = "test-secret-key-for-testing-only-do-not-use-in-production"


def _mint(claims: dict, key: str = SECRET) -> str:
    now = int(time.time())
    base = {
        "iss": "mcp-auth-server",
        "aud": "mcp-proxy",
        "sub": "alice",
        "scopes": [],
        "iat": now,
        "exp": now + 30,
        "server": "github-mcp",
        "upstream_url": "https://api.githubcopilot.com/mcp",
        "auth_method": "oauth2",
        "token_use": "mcp-proxy",
    }
    base.update(claims)
    return pyjwt.encode(base, key, algorithm="HS256")


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", SECRET)


@pytest.mark.unit
class TestVerifyMcpProxyToken:
    def test_valid_token_returns_claims(self):
        claims = verify_mcp_proxy_token(_mint({}))
        assert claims["sub"] == "alice"
        assert claims["auth_method"] == "oauth2"
        assert claims["upstream_url"] == "https://api.githubcopilot.com/mcp"

    def test_wrong_audience_rejected(self):
        tok = _mint({"aud": "mcp-registry-ui"})
        with pytest.raises(HTTPException) as exc:
            verify_mcp_proxy_token(tok)
        assert exc.value.status_code == 401

    def test_wrong_token_use_rejected(self):
        tok = _mint({"token_use": "mcp-registry-ui"})
        with pytest.raises(HTTPException) as exc:
            verify_mcp_proxy_token(tok)
        assert exc.value.status_code == 401

    def test_missing_upstream_rejected(self):
        tok = _mint({"upstream_url": ""})
        with pytest.raises(HTTPException) as exc:
            verify_mcp_proxy_token(tok)
        assert exc.value.status_code == 401

    def test_tampered_signature_rejected(self):
        tok = _mint({}, key="a-different-secret-key-that-is-also-32-bytes!!")
        with pytest.raises(HTTPException) as exc:
            verify_mcp_proxy_token(tok)
        assert exc.value.status_code == 401

    def test_expired_rejected(self):
        now = int(time.time())
        tok = _mint({"iat": now - 120, "exp": now - 60})
        with pytest.raises(HTTPException) as exc:
            verify_mcp_proxy_token(tok)
        assert exc.value.status_code == 401

    def test_garbage_rejected(self):
        with pytest.raises(HTTPException):
            verify_mcp_proxy_token("not.a.jwt")
