"""
Stage 5.6 integration tests: ensure Stage 5 FGAC behavior is consistent across
OIDC, gateway token, API key, and mixed-mode bearer routing.

These tests use FastAPI TestClient against `auth_server.server.app` and avoid
network access by injecting a JWKS fetcher.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

import auth_server.server as auth_server_module
from auth_server.enforceai.auth import dependency as enforceai_dependency
from auth_server.enforceai.crypto.keyring import (
    GatewayKeyring,
    load_gateway_keyring_cached,
)
from auth_server.enforceai.db.data_layer import (
    EnforceAIDataLayer,
)
from auth_server.enforceai.fgac.catalog import (
    clear_scope_catalog_cache,
)
from auth_server.enforceai.oidc.jwks import JWKSCache
from auth_server.enforceai.tokens.mint import (
    mint_gateway_token,
)


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


def _write_scope_catalog(
    *,
    path: Path,
) -> Path:
    content = "\n".join(
        [
            "UI-Scopes: {}",
            "group_mappings: {}",
            "scope-good:",
            "  - server: mcpgw",
            "    methods: [tools/list, tools/call]",
            "    tools: [good_tool]",
            "scope-bad:",
            "  - server: mcpgw",
            "    methods: [tools/list, tools/call]",
            "    tools: [bad_tool]",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")
    return path


def _headers_for(
    *,
    method: str,
    server_name: str = "mcpgw",
    tool_name: str | None = None,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    payload: dict[str, object] = {
        "jsonrpc": "2.0",
        "id": "req-1",
        "method": method,
        "params": {},
    }
    if tool_name is not None:
        payload["params"] = {"name": tool_name, "arguments": {}}

    headers = {
        "X-Original-URL": f"http://localhost/{server_name}/",
        "X-Body": json.dumps(payload),
    }
    if extra:
        headers.update(extra)
    return headers


def _parse_allowed_tools_header(
    value: str,
) -> set[str]:
    stripped = value.strip()
    if not stripped:
        return set()
    if stripped in {"*", "all"}:
        raise AssertionError("Unexpected wildcard tools policy in integration test")
    parsed = json.loads(stripped)
    if not isinstance(parsed, list):
        raise AssertionError("X-Allowed-Tools must be a JSON list")
    return {str(item) for item in parsed}


def _compute_api_key_hash(
    *,
    pepper: bytes,
    secret: str,
) -> str:
    return hmac.new(
        pepper,
        secret.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _reset_enforcement_caches() -> None:
    enforceai_dependency.clear_enforceai_dependency_caches()
    clear_scope_catalog_cache()
    auth_server_module._load_enforceai_runtime.cache_clear()


@pytest.mark.integration
class TestEnforceAIStage5Roundtrip:
    def test_oidc_tools_list_and_call_enforced(
        self,
        tmp_path: Path,
        enforceai_env,
        enforceai_sqlite_db_path: Path,
        enforceai_rsa_keypair_pem: tuple[bytes, bytes],
        enforceai_oidc_issuers_env_json: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        issuer = "https://issuer.example"
        audience = "mcp-registry"

        catalog_path = _write_scope_catalog(path=tmp_path / "scopes.yml")
        data_layer = EnforceAIDataLayer(db_path=enforceai_sqlite_db_path)
        data_layer.initialize()
        stores = data_layer.build_stores()

        user_id = f"{issuer}|user-1"
        agent_id = str(uuid.uuid4())
        stores.agent_store.create_agent(
            user_id=user_id,
            agent_id=agent_id,
            scopes=["scope-good"],
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
            }
        )
        _reset_enforcement_caches()

        now = int(time.time())
        token = jwt.encode(
            {
                "iss": issuer,
                "sub": "user-1",
                "aud": audience,
                "iat": now - 1,
                "exp": now + 3600,
                "scp": ["scope-good"],
            },
            key=private_pem,
            algorithm="RS256",
            headers={"kid": kid},
        )

        client = TestClient(auth_server_module.app)

        list_response = client.get(
            "/validate",
            headers=_headers_for(
                method="tools/list",
                extra={
                    "Authorization": f"Bearer {token}",
                    "X-Agent-Id": agent_id,
                },
            ),
        )
        assert list_response.status_code == 200
        allowed_tools = _parse_allowed_tools_header(
            list_response.headers.get("X-Allowed-Tools", ""),
        )
        assert allowed_tools == {"good_tool"}

        call_ok = client.get(
            "/validate",
            headers=_headers_for(
                method="tools/call",
                tool_name="good_tool",
                extra={
                    "Authorization": f"Bearer {token}",
                    "X-Agent-Id": agent_id,
                },
            ),
        )
        assert call_ok.status_code == 200

        call_denied = client.get(
            "/validate",
            headers=_headers_for(
                method="tools/call",
                tool_name="bad_tool",
                extra={
                    "Authorization": f"Bearer {token}",
                    "X-Agent-Id": agent_id,
                },
            ),
        )
        assert call_denied.status_code == 403

    def test_gateway_token_revocation_denies_tool_call(
        self,
        tmp_path: Path,
        enforceai_env,
        enforceai_sqlite_db_path: Path,
        enforceai_gateway_key_files,
    ) -> None:
        catalog_path = _write_scope_catalog(path=tmp_path / "scopes.yml")
        data_layer = EnforceAIDataLayer(db_path=enforceai_sqlite_db_path)
        data_layer.initialize()
        stores = data_layer.build_stores()

        user_id = "https://issuer.example|user-1"
        agent_id = str(uuid.uuid4())
        stores.agent_store.create_agent(
            user_id=user_id,
            agent_id=agent_id,
            scopes=["scope-good"],
        )

        key_files = enforceai_gateway_key_files
        keyring = GatewayKeyring.load(
            private_key_path=key_files.private_key_path,
            public_keys_dir=key_files.public_keys_dir,
            active_kid=key_files.active_kid,
        )
        token = mint_gateway_token(
            keyring=keyring,
            issuer="enforceai-gateway",
            user_id=user_id,
            agent_id=agent_id,
            scopes=["scope-good"],
            ttl_seconds=3600,
            jti="jti-1",
        )

        enforceai_env(
            {
                "ENFORCEAI_DB_PATH": str(enforceai_sqlite_db_path),
                "ENFORCEAI_AUTH_PROVIDER": "gateway-token",
                "ENFORCEAI_SCOPES_CATALOG_PATH": str(catalog_path),
                "ENFORCEAI_GATEWAY_PRIVATE_KEY_PATH": str(key_files.private_key_path),
                "ENFORCEAI_GATEWAY_PUBLIC_KEYS_DIR": str(key_files.public_keys_dir),
                "ENFORCEAI_GATEWAY_ACTIVE_KID": key_files.active_kid,
                "ENFORCEAI_GATEWAY_ISSUER": "enforceai-gateway",
            }
        )
        _reset_enforcement_caches()
        load_gateway_keyring_cached.cache_clear()

        client = TestClient(auth_server_module.app)

        list_response = client.get(
            "/validate",
            headers=_headers_for(
                method="tools/list",
                extra={"X-Gateway-Token": token},
            ),
        )
        assert list_response.status_code == 200
        assert _parse_allowed_tools_header(
            list_response.headers.get("X-Allowed-Tools", ""),
        ) == {"good_tool"}

        call_ok = client.get(
            "/validate",
            headers=_headers_for(
                method="tools/call",
                tool_name="good_tool",
                extra={"X-Gateway-Token": token},
            ),
        )
        assert call_ok.status_code == 200

        stores.revocation_store.revoke_jti(
            jti="jti-1",
            user_id=user_id,
            agent_id=agent_id,
        )

        call_revoked = client.get(
            "/validate",
            headers=_headers_for(
                method="tools/call",
                tool_name="good_tool",
                extra={"X-Gateway-Token": token},
            ),
        )
        assert call_revoked.status_code == 403

    def test_api_key_scope_restriction_affects_tool_call(
        self,
        tmp_path: Path,
        enforceai_env,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        catalog_path = _write_scope_catalog(path=tmp_path / "scopes.yml")
        data_layer = EnforceAIDataLayer(db_path=enforceai_sqlite_db_path)
        data_layer.initialize()
        stores = data_layer.build_stores()

        user_id = "https://issuer.example|user-1"
        agent_id = str(uuid.uuid4())
        stores.agent_store.create_agent(
            user_id=user_id,
            agent_id=agent_id,
            scopes=["scope-good", "scope-bad"],
        )

        pepper = b"pepper-1"
        pepper_path = tmp_path / "pepper"
        pepper_path.write_bytes(pepper)

        secret = "supersecret"
        key_id = "key-1"
        stores.api_key_store.create_key(
            key_id=key_id,
            secret_hash=_compute_api_key_hash(
                pepper=pepper,
                secret=secret,
            ),
            user_id=user_id,
            agent_id=agent_id,
            scopes=["scope-good"],
        )

        enforceai_env(
            {
                "ENFORCEAI_DB_PATH": str(enforceai_sqlite_db_path),
                "ENFORCEAI_AUTH_PROVIDER": "api-key",
                "ENFORCEAI_SCOPES_CATALOG_PATH": str(catalog_path),
                "ENFORCEAI_API_KEY_PEPPER_PATH": str(pepper_path),
            }
        )
        _reset_enforcement_caches()

        client = TestClient(auth_server_module.app)
        api_key_value = f"eak_{key_id}.{secret}"

        list_response = client.get(
            "/validate",
            headers=_headers_for(
                method="tools/list",
                extra={"X-API-Key": api_key_value},
            ),
        )
        assert list_response.status_code == 200
        assert _parse_allowed_tools_header(
            list_response.headers.get("X-Allowed-Tools", ""),
        ) == {"good_tool"}

        call_ok = client.get(
            "/validate",
            headers=_headers_for(
                method="tools/call",
                tool_name="good_tool",
                extra={"X-API-Key": api_key_value},
            ),
        )
        assert call_ok.status_code == 200

        call_denied = client.get(
            "/validate",
            headers=_headers_for(
                method="tools/call",
                tool_name="bad_tool",
                extra={"X-API-Key": api_key_value},
            ),
        )
        assert call_denied.status_code == 403

    def test_mixed_mode_bearer_routes_by_issuer(
        self,
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
        jwks_uri = "https://issuer.example/.well-known/jwks.json"

        catalog_path = _write_scope_catalog(path=tmp_path / "scopes.yml")
        data_layer = EnforceAIDataLayer(db_path=enforceai_sqlite_db_path)
        data_layer.initialize()
        stores = data_layer.build_stores()

        user_id = f"{issuer}|user-1"
        agent_id = str(uuid.uuid4())
        stores.agent_store.create_agent(
            user_id=user_id,
            agent_id=agent_id,
            scopes=["scope-good"],
        )

        pepper_path = tmp_path / "pepper"
        pepper_path.write_bytes(b"pepper-1")

        key_files = enforceai_gateway_key_files
        keyring = GatewayKeyring.load(
            private_key_path=key_files.private_key_path,
            public_keys_dir=key_files.public_keys_dir,
            active_kid=key_files.active_kid,
        )
        gateway_token = mint_gateway_token(
            keyring=keyring,
            issuer="enforceai-gateway",
            user_id=user_id,
            agent_id=agent_id,
            scopes=["scope-good"],
            ttl_seconds=3600,
            jti="jti-2",
        )

        kid = "kid-oidc-2"
        private_pem, _public_pem = enforceai_rsa_keypair_pem
        private_key = serialization.load_pem_private_key(private_pem, password=None)
        assert isinstance(private_key, rsa.RSAPrivateKey)
        public_key = private_key.public_key()
        assert isinstance(public_key, rsa.RSAPublicKey)
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
                "ENFORCEAI_AUTH_PROVIDER": "mixed",
                "OIDC_ISSUERS": enforceai_oidc_issuers_env_json,
                "ENFORCEAI_SCOPES_CATALOG_PATH": str(catalog_path),
                "ENFORCEAI_API_KEY_PEPPER_PATH": str(pepper_path),
                "ENFORCEAI_GATEWAY_PRIVATE_KEY_PATH": str(key_files.private_key_path),
                "ENFORCEAI_GATEWAY_PUBLIC_KEYS_DIR": str(key_files.public_keys_dir),
                "ENFORCEAI_GATEWAY_ACTIVE_KID": key_files.active_kid,
                "ENFORCEAI_GATEWAY_ISSUER": "enforceai-gateway",
            }
        )
        _reset_enforcement_caches()
        load_gateway_keyring_cached.cache_clear()

        now = int(time.time())
        oidc_token = jwt.encode(
            {
                "iss": issuer,
                "sub": "user-1",
                "aud": audience,
                "iat": now - 1,
                "exp": now + 3600,
                "scp": ["scope-good"],
            },
            key=private_pem,
            algorithm="RS256",
            headers={"kid": kid},
        )

        client = TestClient(auth_server_module.app)

        gateway_response = client.get(
            "/validate",
            headers=_headers_for(
                method="tools/call",
                tool_name="good_tool",
                extra={
                    "Authorization": f"Bearer {gateway_token}",
                },
            ),
        )
        assert gateway_response.status_code == 200
        assert gateway_response.headers.get("X-Auth-Method") == "gateway-token"

        oidc_response = client.get(
            "/validate",
            headers=_headers_for(
                method="tools/call",
                tool_name="good_tool",
                extra={
                    "Authorization": f"Bearer {oidc_token}",
                    "X-Agent-Id": agent_id,
                },
            ),
        )
        assert oidc_response.status_code == 200
        assert oidc_response.headers.get("X-Auth-Method") == "oidc"

