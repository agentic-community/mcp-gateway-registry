"""Unit tests for auth_server/internal_request_token.py.

Covers the /validate-minted internal JWT: mint round-trips, the verify
dependency's accept/reject paths (missing / garbage / expired / tampered /
wrong-audience / wrong-token_use / missing upstream / server-claim-path
mismatch), the sub-path-append parity that the bound upstream_url relies on,
and the audience-parameterization seam the registry fast-follow reuses.
"""

import os
import time
from types import SimpleNamespace
from unittest.mock import patch

import jwt as pyjwt
import pytest
from fastapi import HTTPException

from auth_server.internal_request_token import (
    MCP_PROXY_AUDIENCE,
    MCP_REGISTRY_UI_AUDIENCE,
    _decode_internal_token,
    _mint_internal_token,
    mint_mcp_proxy_token,
    mint_registry_ui_token,
    verify_mcp_proxy_token,
)

_SECRET = "test-secret-key-for-testing-only"


def _request(headers: dict[str, str], server_name: str = "airegistry-tools/mcp"):
    """Build a minimal stand-in for a FastAPI Request with the bits the
    dependency reads: .headers, .path_params, and a mutable .state."""
    lower = {k.lower(): v for k, v in headers.items()}
    lower.setdefault("x-registered-server-path", "/" + server_name.strip("/").removesuffix("/mcp"))
    lower.setdefault("x-registered-route-mode", "direct")
    return SimpleNamespace(
        headers=SimpleNamespace(get=lambda k, default=None: lower.get(k.lower(), default)),
        path_params={"server_name": server_name},
        state=SimpleNamespace(),
    )


@pytest.fixture(autouse=True)
def _secret_env():
    with patch.dict(os.environ, {"SECRET_KEY": _SECRET}, clear=False):
        yield


# --------------------------------------------------------------------------- #
# Mint
# --------------------------------------------------------------------------- #


class TestMint:
    def test_mint_roundtrips_expected_claims(self) -> None:
        token = mint_mcp_proxy_token(
            subject="alice",
            scopes=["mcp-servers-unrestricted/read"],
            server_name="airegistry-tools",
            upstream_url="https://upstream.example/mcp",
        )
        claims = _decode_internal_token(token, audience=MCP_PROXY_AUDIENCE)
        assert claims["sub"] == "alice"
        assert claims["scopes"] == ["mcp-servers-unrestricted/read"]
        assert claims["server"] == "airegistry-tools"
        assert claims["upstream_url"] == "https://upstream.example/mcp"
        assert claims["token_use"] == "mcp-proxy"
        assert claims["aud"] == "mcp-proxy"
        assert claims["iss"] == "mcp-auth-server"

    def test_mint_empty_subject_raises(self) -> None:
        # An empty subject would mint an anonymous-but-valid token; refuse so the
        # proxy fails closed instead of trusting "".
        with pytest.raises(ValueError, match="empty subject"):
            mint_mcp_proxy_token("", [], "srv", "https://u.example")

    def test_mint_without_secret_raises(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="SECRET_KEY"):
                mint_mcp_proxy_token("alice", [], "srv", "https://u.example")


# --------------------------------------------------------------------------- #
# Verify dependency
# --------------------------------------------------------------------------- #


class TestVerify:
    @pytest.mark.asyncio
    async def test_valid_token_passes_and_stashes_claims(self) -> None:
        token = mint_mcp_proxy_token(
            "alice", ["s/read"], "airegistry-tools", "https://u.example/mcp"
        )
        req = _request({"X-Internal-Token": token}, server_name="airegistry-tools/mcp")
        await verify_mcp_proxy_token(req)
        assert req.state.mcp_proxy_claims["sub"] == "alice"
        assert req.state.mcp_proxy_claims["upstream_url"] == "https://u.example/mcp"

    @pytest.mark.asyncio
    async def test_nested_registered_transport_name_preserved(self) -> None:
        token = mint_mcp_proxy_token(
            "alice", ["scope"], "peer/mcp", "https://u.example/peer/mcp", virtual_backend=True
        )
        request = _request(
            {
                "X-Internal-Token": token,
                "X-Registered-Server-Path": "/peer/mcp",
                "X-Registered-Route-Mode": "virtual",
            },
            server_name="peer/mcp",
        )
        await verify_mcp_proxy_token(request)
        assert request.state.mcp_proxy_claims["server"] == "peer/mcp"

        # A token for /peer cannot be replayed at the nested registration.
        parent = mint_mcp_proxy_token("alice", ["scope"], "peer", "https://u.example/peer")
        with pytest.raises(HTTPException) as exc:
            await verify_mcp_proxy_token(
                _request(
                    {
                        "X-Internal-Token": parent,
                        "X-Registered-Server-Path": "/peer/mcp",
                        "X-Registered-Route-Mode": "virtual",
                    },
                    "peer/mcp",
                )
            )
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_version_marker_is_rejected(self) -> None:
        token = mint_mcp_proxy_token(
            "alice", ["scope"], "peer", "https://u.example/peer", version_id="__invalid_version__"
        )
        with pytest.raises(HTTPException) as exc:
            await verify_mcp_proxy_token(_request({"X-Internal-Token": token}, "peer"))
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_token_raises_401(self) -> None:
        # Always fail-closed: no token -> 401.
        req = _request({})
        with pytest.raises(HTTPException) as exc:
            await verify_mcp_proxy_token(req)
        assert exc.value.status_code == 401
        assert exc.value.detail == "Missing internal proxy token"

    @pytest.mark.asyncio
    async def test_garbage_token_raises_401(self) -> None:
        # A non-JWT / tampered token is rejected.
        req = _request({"X-Internal-Token": "not.a.jwt"})
        with pytest.raises(HTTPException) as exc:
            await verify_mcp_proxy_token(req)
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_expired_token_rejected(self) -> None:
        # Mint with a tiny TTL, wait past it + leeway.
        with patch.dict(
            os.environ,
            {"INTERNAL_TOKEN_TTL_SECONDS": "5", "INTERNAL_TOKEN_LEEWAY_SECONDS": "0"},
            clear=False,
        ):
            now = int(time.time())
            token = pyjwt.encode(
                {
                    "iss": "mcp-auth-server",
                    "aud": "mcp-proxy",
                    "sub": "alice",
                    "scopes": [],
                    "server": "airegistry-tools",
                    "upstream_url": "https://u.example",
                    "token_use": "mcp-proxy",
                    "iat": now - 100,
                    "exp": now - 50,
                },
                _SECRET,
                algorithm="HS256",
            )
            req = _request({"X-Internal-Token": token})
            with pytest.raises(HTTPException) as exc:
                await verify_mcp_proxy_token(req)
            assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_wrong_signing_key_rejected(self) -> None:
        now = int(time.time())
        token = pyjwt.encode(
            {
                "iss": "mcp-auth-server",
                "aud": "mcp-proxy",
                "sub": "alice",
                "scopes": [],
                "server": "airegistry-tools",
                "upstream_url": "https://u.example",
                "token_use": "mcp-proxy",
                "iat": now,
                "exp": now + 30,
            },
            "WRONG-KEY-but-long-enough-to-satisfy-hmac-length-check",
            algorithm="HS256",
        )
        req = _request({"X-Internal-Token": token})
        with pytest.raises(HTTPException) as exc:
            await verify_mcp_proxy_token(req)
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_wrong_audience_rejected(self) -> None:
        # A service-to-service internal token (aud=mcp-registry) cannot be
        # replayed at mcp_proxy.
        token = _mint_internal_token(
            audience="mcp-registry",
            subject="registry-service",
            scopes=[],
            extra_claims={
                "token_use": "mcp-proxy",
                "upstream_url": "https://u.example",
                "server": "airegistry-tools",
            },
        )
        req = _request({"X-Internal-Token": token})
        with pytest.raises(HTTPException) as exc:
            await verify_mcp_proxy_token(req)
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_wrong_token_use_rejected(self) -> None:
        token = _mint_internal_token(
            audience=MCP_PROXY_AUDIENCE,
            subject="alice",
            scopes=[],
            extra_claims={
                "token_use": "something-else",
                "upstream_url": "https://u.example",
                "server": "airegistry-tools",
            },
        )
        req = _request({"X-Internal-Token": token})
        with pytest.raises(HTTPException) as exc:
            await verify_mcp_proxy_token(req)
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_upstream_claim_rejected(self) -> None:
        token = _mint_internal_token(
            audience=MCP_PROXY_AUDIENCE,
            subject="alice",
            scopes=[],
            extra_claims={
                "token_use": "mcp-proxy",
                "upstream_url": "",
                "server": "airegistry-tools",
            },
        )
        req = _request({"X-Internal-Token": token})
        with pytest.raises(HTTPException) as exc:
            await verify_mcp_proxy_token(req)
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_server_claim_path_mismatch_rejected(self) -> None:
        # Token minted for server "airegistry-tools" but request path is for "github-mcp".
        token = mint_mcp_proxy_token("alice", [], "airegistry-tools", "https://u.example/mcp")
        req = _request({"X-Internal-Token": token}, server_name="github-mcp/mcp")
        with pytest.raises(HTTPException) as exc:
            await verify_mcp_proxy_token(req)
        assert exc.value.status_code == 401
        assert "mismatch" in exc.value.detail.lower()


# --------------------------------------------------------------------------- #
# Audience-parameterization seam (registry fast-follow reuses this)
# --------------------------------------------------------------------------- #


class TestAudienceParameterization:
    def test_token_minted_for_one_audience_fails_another(self) -> None:
        token = _mint_internal_token(audience="mcp-registry-ui", subject="alice", scopes=["x"])
        # Decoding against a different audience must fail.
        with pytest.raises(pyjwt.InvalidTokenError):
            _decode_internal_token(token, audience="mcp-proxy")
        # But succeeds against the minted audience.
        claims = _decode_internal_token(token, audience="mcp-registry-ui")
        assert claims["sub"] == "alice"


# --------------------------------------------------------------------------- #
# Mint registry-UI token (the registry /api/ hop)
# --------------------------------------------------------------------------- #


class TestMintRegistryUi:
    def test_session_backed_roundtrips_expected_claims(self) -> None:
        # Browser/session-backed caller: session_id carried, groups left empty
        # (the registry resolves live groups from the session store).
        token = mint_registry_ui_token(
            subject="alice",
            session_id="sess-abc123",
            groups=[],
            auth_method="keycloak",
            client_id="ui-client",
        )
        claims = _decode_internal_token(token, audience=MCP_REGISTRY_UI_AUDIENCE)
        assert claims["sub"] == "alice"
        assert claims["session_id"] == "sess-abc123"
        assert claims["groups"] == []
        assert claims["auth_method"] == "keycloak"
        assert claims["client_id"] == "ui-client"
        assert claims["token_use"] == "mcp-registry-ui"
        assert claims["aud"] == "mcp-registry-ui"
        assert claims["iss"] == "mcp-auth-server"
        # Thin identity assertion: no scopes encoded.
        assert claims["scopes"] == []

    def test_bearer_static_binds_groups_not_session(self) -> None:
        # Bearer/static-token caller: no session row, so groups ride the claim.
        token = mint_registry_ui_token(
            subject="svc-key-1",
            session_id="",
            groups=["registry-admins", "team-a"],
            auth_method="network-trusted",
            client_id="key-1",
        )
        claims = _decode_internal_token(token, audience=MCP_REGISTRY_UI_AUDIENCE)
        assert claims["sub"] == "svc-key-1"
        assert claims["session_id"] == ""
        assert claims["groups"] == ["registry-admins", "team-a"]
        assert claims["auth_method"] == "network-trusted"

    def test_empty_subject_raises(self) -> None:
        # Fail-closed: an empty subject must not mint an anonymous-but-valid token.
        with pytest.raises(ValueError, match="empty subject"):
            mint_registry_ui_token("", "", [], "keycloak", "")

    def test_without_secret_raises(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="SECRET_KEY"):
                mint_registry_ui_token("alice", "sess", [], "keycloak", "")

    def test_registry_ui_token_not_replayable_as_mcp_proxy(self) -> None:
        # Cross-audience separation: a registry-UI token must not verify as mcp-proxy.
        token = mint_registry_ui_token("alice", "sess", [], "keycloak", "")
        with pytest.raises(pyjwt.InvalidTokenError):
            _decode_internal_token(token, audience=MCP_PROXY_AUDIENCE)
