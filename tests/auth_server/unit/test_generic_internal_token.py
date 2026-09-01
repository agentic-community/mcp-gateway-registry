"""Unit tests for the generic-proxy internal token mint + verify guards.

Covers two branches otherwise unexercised on the gateway-generic-proxy path:

- ``mint_generic_proxy_token`` rejects a blank ``http_method`` (the verb is
  bound into the token so a safe method cannot be replayed as a state-changing
  one -- an empty verb must never mint).
- ``verify_generic_proxy_token`` (the /proxy route dependency) rejects a token
  whose bound ``method`` claim does not match the actual request method (replay
  guard), returning 401.
"""

import os

import pytest
from fastapi import HTTPException
from starlette.requests import Request

os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-definitely-long-enough-32b")

from auth_server.internal_request_token import (  # noqa: E402
    mint_generic_proxy_token,
    verify_generic_proxy_token,
)

pytestmark = pytest.mark.unit


def _request(method: str, token: str, entity_type: str = "skill") -> Request:
    """Build a minimal ASGI Request the verifier reads (headers/method/path_params)."""
    return Request(
        {
            "type": "http",
            "method": method,
            "path": f"/proxy/{entity_type}/skills/proxy-demo",
            "headers": [(b"x-internal-token-generic", token.encode())],
            "path_params": {"entity_type": entity_type, "entity_path": "skills/proxy-demo"},
            "query_string": b"",
            "state": {},
        }
    )


class TestMintGenericProxyToken:
    @pytest.mark.parametrize("bad_method", ["", "   ", "\t"])
    def test_blank_http_method_rejected(self, bad_method: str) -> None:
        with pytest.raises(ValueError, match="http_method is required"):
            mint_generic_proxy_token(
                subject="alice",
                scopes=["skill/x"],
                entity_type="skill",
                registered_path="skills/proxy-demo",
                upstream_url="https://backend.example/",
                http_method=bad_method,
            )

    def test_valid_method_mints(self) -> None:
        tok = mint_generic_proxy_token(
            subject="alice",
            scopes=["skill/x"],
            entity_type="skill",
            registered_path="skills/proxy-demo",
            upstream_url="https://backend.example/",
            http_method="get",
        )
        assert isinstance(tok, str) and tok.count(".") == 2  # a signed JWT


class TestVerifyGenericProxyTokenMethodBinding:
    async def test_method_claim_request_mismatch_rejected(self) -> None:
        # Token bound to GET, request arrives as POST -> replay guard fires (401).
        tok = mint_generic_proxy_token(
            subject="alice",
            scopes=["skill/x"],
            entity_type="skill",
            registered_path="skills/proxy-demo",
            upstream_url="https://backend.example/",
            http_method="GET",
        )
        with pytest.raises(HTTPException) as exc:
            await verify_generic_proxy_token(_request("POST", tok))
        assert exc.value.status_code == 401
        assert "Method claim/request mismatch" in exc.value.detail

    async def test_matching_method_passes_method_guard(self) -> None:
        # Same verb bound and requested: the method guard does NOT fire; the token
        # verifies end-to-end and the claims are stashed on request.state.
        tok = mint_generic_proxy_token(
            subject="alice",
            scopes=["skill/x"],
            entity_type="skill",
            registered_path="skills/proxy-demo",
            upstream_url="https://backend.example/",
            http_method="GET",
        )
        request = _request("GET", tok)
        await verify_generic_proxy_token(request)
        claims = request.state.generic_proxy_claims
        assert claims["method"] == "GET"
        assert claims["entity_type"] == "skill"
        assert claims["upstream_url"] == "https://backend.example/"
