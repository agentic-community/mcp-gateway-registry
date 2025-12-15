"""
Stage 7.4 integration tests: audit failure policy + request-path caching.

These tests use FastAPI TestClient against `auth_server.server.app` and avoid
network access by injecting a JWKS fetcher.
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from pathlib import Path
from typing import Any

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
    EnforceAIStores,
)
from auth_server.enforceai.fgac import catalog as fgac_catalog
from auth_server.enforceai.oidc.jwks import JWKSCache
from auth_server.enforceai.tokens.mint import (
    mint_gateway_token,
)


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
    extra: dict[str, str] | None = None,
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


def _reset_enforcement_caches() -> None:
    enforceai_dependency.clear_enforceai_dependency_caches()
    fgac_catalog.clear_scope_catalog_cache()
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


class _FailingAuditStore:
    def append_event(
        self,
        **_: object,
    ) -> None:
        raise RuntimeError("simulated audit persist failure")


@pytest.mark.integration
class TestEnforceAIStage7Hardening:
    def test_validate_not_denied_when_audit_persist_fails(
        self,
        tmp_path: Path,
        enforceai_env,
        enforceai_sqlite_db_path: Path,
        enforceai_gateway_key_files,
        monkeypatch: pytest.MonkeyPatch,
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

        failing_stores = EnforceAIStores(
            agent_store=stores.agent_store,
            api_key_store=stores.api_key_store,
            revocation_store=stores.revocation_store,
            audit_store=_FailingAuditStore(),  # type: ignore[arg-type]
            user_store=stores.user_store,
            session_store=stores.session_store,
        )
        monkeypatch.setattr(
            auth_server_module,
            "get_enforceai_stores",
            lambda: failing_stores,
        )

        client = TestClient(auth_server_module.app)
        response = client.get(
            "/validate",
            headers=_headers_for(
                method="tools/list",
                extra={"X-Gateway-Token": token},
            ),
        )
        assert response.status_code == 200

    def test_management_not_denied_when_audit_persist_fails(
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
        bootstrap_agent_id = str(uuid.uuid4())
        stores.agent_store.create_agent(
            user_id=user_id,
            agent_id=bootstrap_agent_id,
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

        failing_stores = EnforceAIStores(
            agent_store=stores.agent_store,
            api_key_store=stores.api_key_store,
            revocation_store=stores.revocation_store,
            audit_store=_FailingAuditStore(),  # type: ignore[arg-type]
            user_store=stores.user_store,
            session_store=stores.session_store,
        )
        auth_server_module.app.dependency_overrides[
            enforceai_dependency.get_enforceai_stores
        ] = lambda: failing_stores

        try:
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
            response = client.get(
                "/enforceai/agents",
                headers={
                    "Authorization": f"Bearer {oidc_token}",
                    "X-Agent-Id": bootstrap_agent_id,
                },
            )
            assert response.status_code == 200
        finally:
            auth_server_module.app.dependency_overrides.pop(
                enforceai_dependency.get_enforceai_stores,
                None,
            )

    def test_request_path_uses_cached_catalog_and_jwks(
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

        jwks_fetches: list[str] = []
        kid = "kid-oidc-1"
        private_pem, _public_pem = enforceai_rsa_keypair_pem
        private_key = serialization.load_pem_private_key(private_pem, password=None)
        assert isinstance(private_key, rsa.RSAPrivateKey)
        public_key = private_key.public_key()
        assert isinstance(public_key, rsa.RSAPublicKey)

        jwks_uri = "https://issuer.example/.well-known/jwks.json"
        jwks = {"keys": [_jwk_from_public_key(public_key=public_key, kid=kid)]}

        async def fetcher(uri: str) -> dict[str, Any]:
            jwks_fetches.append(uri)
            if uri == jwks_uri:
                return jwks
            raise AssertionError(f"Unexpected JWKS URI: {uri}")

        monkeypatch.setattr(
            enforceai_dependency,
            "JWKSCache",
            lambda: JWKSCache(fetcher=fetcher),
        )

        catalog_reads: list[Path] = []
        original_read_text = fgac_catalog._read_text

        def _counting_read_text(path: Path) -> str:
            catalog_reads.append(path)
            return original_read_text(path)

        monkeypatch.setattr(
            fgac_catalog,
            "_read_text",
            _counting_read_text,
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
        first = client.get(
            "/validate",
            headers=_headers_for(
                method="tools/list",
                extra={
                    "Authorization": f"Bearer {token}",
                    "X-Agent-Id": agent_id,
                },
            ),
        )
        assert first.status_code == 200

        second = client.get(
            "/validate",
            headers=_headers_for(
                method="tools/list",
                extra={
                    "Authorization": f"Bearer {token}",
                    "X-Agent-Id": agent_id,
                },
            ),
        )
        assert second.status_code == 200

        assert len(jwks_fetches) == 1
        assert len(catalog_reads) == 1

    def test_request_path_uses_cached_gateway_keyring(
        self,
        tmp_path: Path,
        enforceai_env,
        enforceai_sqlite_db_path: Path,
        enforceai_gateway_key_files,
        monkeypatch: pytest.MonkeyPatch,
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

        keyring_loads: list[object] = []
        original_load = GatewayKeyring.load

        def _counting_load(
            *,
            private_key_path: Path,
            public_keys_dir: Path,
            active_kid: str,
        ) -> GatewayKeyring:
            keyring_loads.append(object())
            return original_load(
                private_key_path=private_key_path,
                public_keys_dir=public_keys_dir,
                active_kid=active_kid,
            )

        monkeypatch.setattr(
            GatewayKeyring,
            "load",
            staticmethod(_counting_load),
        )

        key_files = enforceai_gateway_key_files
        token = mint_gateway_token(
            keyring=original_load(
                private_key_path=key_files.private_key_path,
                public_keys_dir=key_files.public_keys_dir,
                active_kid=key_files.active_kid,
            ),
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

        client = TestClient(auth_server_module.app)
        first = client.get(
            "/validate",
            headers=_headers_for(
                method="tools/list",
                extra={"X-Gateway-Token": token},
            ),
        )
        assert first.status_code == 200

        second = client.get(
            "/validate",
            headers=_headers_for(
                method="tools/list",
                extra={"X-Gateway-Token": token},
            ),
        )
        assert second.status_code == 200

        assert len(keyring_loads) == 1
