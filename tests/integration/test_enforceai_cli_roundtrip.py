"""
Stage 6.3 integration test: CLI roundtrip against an in-process ASGI app.

This test uses httpx.ASGITransport (no network).
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

import auth_server.server as auth_server_module
from auth_server.enforceai.auth import dependency as enforceai_dependency
from auth_server.enforceai.crypto.keyring import (
    load_gateway_keyring_cached,
)
from auth_server.enforceai.db.data_layer import (
    EnforceAIDataLayer,
)
from auth_server.enforceai.fgac.catalog import (
    clear_scope_catalog_cache,
)
from auth_server.enforceai.oidc.jwks import JWKSCache
from cli.enforceai_cli import (
    run_async,
)


def _write_scope_catalog(
    *,
    path: Path,
) -> Path:
    content = "\n".join(
        [
            "UI-Scopes: {}",
            "group_mappings: {}",
            "scope-mgmt:",
            "  - server: mcpgw",
            "    methods: [tools/list, tools/call]",
            "    tools: [good_tool]",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")
    return path


def _reset_enforcement_caches() -> None:
    enforceai_dependency.clear_enforceai_dependency_caches()
    clear_scope_catalog_cache()
    load_gateway_keyring_cached.cache_clear()
    auth_server_module._load_enforceai_runtime.cache_clear()


def _b64url_uint(
    value: int,
) -> str:
    data = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _jwk_from_public_key(
    *,
    public_key: rsa.RSAPublicKey,
    kid: str,
) -> dict[str, Any]:
    numbers = public_key.public_numbers()
    return {
        "kty": "RSA",
        "kid": kid,
        "use": "sig",
        "alg": "RS256",
        "n": _b64url_uint(numbers.n),
        "e": _b64url_uint(numbers.e),
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_enforceai_cli_roundtrip_oidc_create_agent_mint_token_revoke_all(
    tmp_path: Path,
    enforceai_env,
    enforceai_sqlite_db_path: Path,
    enforceai_gateway_key_files,
    enforceai_rsa_keypair_pem: tuple[bytes, bytes],
    enforceai_oidc_issuers_env_json: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    issuer = "https://issuer.example"
    audience = "mcp-registry"

    catalog_path = _write_scope_catalog(path=tmp_path / "scopes.yml")
    pepper_path = tmp_path / "pepper"
    pepper_path.write_bytes(b"pepper-1")

    data_layer = EnforceAIDataLayer(db_path=enforceai_sqlite_db_path)
    data_layer.initialize()
    stores = data_layer.build_stores()

    user_id = f"{issuer}|user-1"
    bootstrap_agent_id = str(uuid.uuid4())
    stores.agent_store.create_agent(
        user_id=user_id,
        agent_id=bootstrap_agent_id,
        scopes=["scope-mgmt"],
    )

    kid = "kid-oidc-1"
    private_pem, _public_pem = enforceai_rsa_keypair_pem
    private_key = serialization.load_pem_private_key(private_pem, password=None)
    assert isinstance(private_key, rsa.RSAPrivateKey)
    public_key = private_key.public_key()
    assert isinstance(public_key, rsa.RSAPublicKey)

    jwks_uri = "https://issuer.example/.well-known/jwks.json"
    jwks = {"keys": [_jwk_from_public_key(public_key=public_key, kid=kid)]}

    async def fetcher(uri: str) -> dict[str, Any]:
        if uri == jwks_uri:
            return jwks
        raise AssertionError(f"Unexpected JWKS URI: {uri}")

    monkeypatch.setattr(
        enforceai_dependency,
        "JWKSCache",
        lambda: JWKSCache(fetcher=fetcher),
    )

    enforceai_env(
        {
            "ENFORCEAI_DB_PATH": str(enforceai_sqlite_db_path),
            "ENFORCEAI_AUTH_PROVIDER": "oidc",
            "OIDC_ISSUERS": enforceai_oidc_issuers_env_json,
            "ENFORCEAI_SCOPES_CATALOG_PATH": str(catalog_path),
            "ENFORCEAI_API_KEY_PEPPER_PATH": str(pepper_path),
            "ENFORCEAI_GATEWAY_PRIVATE_KEY_PATH": str(enforceai_gateway_key_files.private_key_path),
            "ENFORCEAI_GATEWAY_PUBLIC_KEYS_DIR": str(enforceai_gateway_key_files.public_keys_dir),
            "ENFORCEAI_GATEWAY_ACTIVE_KID": enforceai_gateway_key_files.active_kid,
            "ENFORCEAI_GATEWAY_ISSUER": "enforceai-gateway",
        }
    )
    _reset_enforcement_caches()

    now = int(time.time())
    oidc_token = jwt.encode(
        {
            "iss": issuer,
            "sub": "user-1",
            "aud": audience,
            "iat": now - 1,
            "exp": now + 3600,
            "scp": ["scope-mgmt"],
        },
        key=private_pem,
        algorithm="RS256",
        headers={"kid": kid},
    )

    transport = httpx.ASGITransport(app=auth_server_module.app)

    code, out, err = await run_async(
        [
            "--base-url",
            "http://testserver",
            "--authorization",
            oidc_token,
            "--x-agent-id",
            bootstrap_agent_id,
            "agents",
            "create",
            "--scope",
            "scope-mgmt",
            "--alias",
            "agent-2",
        ],
        transport=transport,
    )
    assert code == 0, err
    created = json.loads(out)
    created_agent_id = created["agent_id"]
    assert created_agent_id

    code, out, err = await run_async(
        [
            "--base-url",
            "http://testserver",
            "--authorization",
            oidc_token,
            "--x-agent-id",
            bootstrap_agent_id,
            "tokens",
            "mint",
            created_agent_id,
            "--scope",
            "scope-mgmt",
            "--ttl-seconds",
            "60",
        ],
        transport=transport,
    )
    assert code == 0, err
    minted = json.loads(out)
    assert minted["token"]

    code, out, err = await run_async(
        [
            "--base-url",
            "http://testserver",
            "--authorization",
            oidc_token,
            "--x-agent-id",
            bootstrap_agent_id,
            "tokens",
            "revoke-all",
            created_agent_id,
        ],
        transport=transport,
    )
    assert code == 0, err
    revoked_all = json.loads(out)
    assert revoked_all["tokens_valid_after"] is not None

