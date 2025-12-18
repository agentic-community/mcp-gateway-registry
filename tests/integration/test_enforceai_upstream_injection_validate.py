"""
Phase 4 integration tests: request-time upstream credential resolution + /validate headers.

These tests exercise the auth_server `/validate` endpoint directly (FastAPI TestClient)
and avoid any external network dependencies.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
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
from auth_server.enforceai.secrets.upstream_kek import (
    load_upstream_kek,
)
from auth_server.enforceai.tokens.mint import (
    mint_gateway_token,
)


def _reset_enforcement_caches() -> None:
    enforceai_dependency.clear_enforceai_dependency_caches()
    clear_scope_catalog_cache()
    auth_server_module._load_enforceai_runtime.cache_clear()


def _write_scope_catalog(
    *,
    path: Path,
) -> Path:
    content = "\n".join(
        [
            "UI-Scopes: {}",
            "group_mappings: {}",
            "scope-good:",
            "  - server: fininfo",
            "    methods: [tools/list, tools/call]",
            "    tools: [good_tool]",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")
    return path


def _headers_for(
    *,
    server_name: str = "fininfo",
    method: str = "tools/list",
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    payload: dict[str, object] = {
        "jsonrpc": "2.0",
        "id": "req-1",
        "method": method,
        "params": {},
    }
    headers = {
        "X-Original-URL": f"http://localhost/{server_name}/",
        "X-Body": json.dumps(payload),
    }
    if extra:
        headers.update(extra)
    return headers


@pytest.mark.integration
class TestEnforceAIUpstreamInjectionValidate:
    def test_validate_sets_mcp_identity_and_api_key_injection_headers(
        self,
        tmp_path: Path,
        enforceai_env,
        enforceai_sqlite_db_path: Path,
        enforceai_gateway_key_files,
    ) -> None:
        catalog_path = _write_scope_catalog(path=tmp_path / "scopes.yml")
        upstream_kek_path = tmp_path / "upstream_kek"
        upstream_kek_path.write_text("11" * 32)
        upstream_kek = load_upstream_kek(upstream_kek_path)

        data_layer = EnforceAIDataLayer(db_path=enforceai_sqlite_db_path)
        data_layer.initialize()
        stores = data_layer.build_stores(upstream_kek=upstream_kek)
        assert stores.upstream_credential_store is not None

        user_id = "https://issuer.example|user-1"
        agent_id = str(uuid.uuid4())
        stores.agent_store.create_agent(
            user_id=user_id,
            agent_id=agent_id,
            scopes=["scope-good"],
        )
        stores.upstream_credential_store.create_credential(
            server_path="/fininfo",
            credential_type="api-key",
            credential_binding="user",
            user_id=user_id,
            secret_payload={"api_key": "super-secret"},
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
            jti="jti-upstream-1",
        )

        enforceai_env(
            {
                "ENFORCEAI_DB_PATH": str(enforceai_sqlite_db_path),
                "ENFORCEAI_AUTH_PROVIDER": "gateway-token",
                "ENFORCEAI_SCOPES_CATALOG_PATH": str(catalog_path),
                "ENFORCEAI_UPSTREAM_KEK_PATH": str(upstream_kek_path),
                "ENFORCEAI_GATEWAY_PRIVATE_KEY_PATH": str(key_files.private_key_path),
                "ENFORCEAI_GATEWAY_PUBLIC_KEYS_DIR": str(key_files.public_keys_dir),
                "ENFORCEAI_GATEWAY_ACTIVE_KID": key_files.active_kid,
                "ENFORCEAI_GATEWAY_ISSUER": "enforceai-gateway",
            }
        )
        _reset_enforcement_caches()
        load_gateway_keyring_cached.cache_clear()

        client = TestClient(auth_server_module.app)
        response = client.get(
            "/validate",
            headers=_headers_for(
                extra={
                    "X-Gateway-Token": token,
                    "X-EnforceAI-Server-Path": "/fininfo",
                    "X-EnforceAI-Upstream-Auth-Type": "api-key",
                    "X-EnforceAI-Upstream-Credential-Binding": "user",
                    "X-EnforceAI-Upstream-Header-Name": "X-API-Key",
                }
            ),
        )
        assert response.status_code == 200
        assert response.headers.get("X-MCP-Principal") == f"user:{user_id}"
        assert response.headers.get("X-MCP-Auth-Type") == "gateway-token"
        assert response.headers.get("X-MCP-Scopes") == "scope-good"
        assert response.headers.get("X-MCP-Claims") is not None
        assert response.headers.get("X-EnforceAI-Upstream-Mode") == "api-key"
        assert response.headers.get("X-EnforceAI-Upstream-Api-Key") == "super-secret"
        assert response.headers.get("X-EnforceAI-Upstream-Api-Key-Header") == "X-API-Key"

    def test_validate_missing_upstream_credential_returns_424_with_error_code(
        self,
        tmp_path: Path,
        enforceai_env,
        enforceai_sqlite_db_path: Path,
        enforceai_gateway_key_files,
    ) -> None:
        catalog_path = _write_scope_catalog(path=tmp_path / "scopes.yml")
        upstream_kek_path = tmp_path / "upstream_kek"
        upstream_kek_path.write_text("22" * 32)
        upstream_kek = load_upstream_kek(upstream_kek_path)

        data_layer = EnforceAIDataLayer(db_path=enforceai_sqlite_db_path)
        data_layer.initialize()
        stores = data_layer.build_stores(upstream_kek=upstream_kek)

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
            jti="jti-upstream-2",
        )

        enforceai_env(
            {
                "ENFORCEAI_DB_PATH": str(enforceai_sqlite_db_path),
                "ENFORCEAI_AUTH_PROVIDER": "gateway-token",
                "ENFORCEAI_SCOPES_CATALOG_PATH": str(catalog_path),
                "ENFORCEAI_UPSTREAM_KEK_PATH": str(upstream_kek_path),
                "ENFORCEAI_GATEWAY_PRIVATE_KEY_PATH": str(key_files.private_key_path),
                "ENFORCEAI_GATEWAY_PUBLIC_KEYS_DIR": str(key_files.public_keys_dir),
                "ENFORCEAI_GATEWAY_ACTIVE_KID": key_files.active_kid,
                "ENFORCEAI_GATEWAY_ISSUER": "enforceai-gateway",
            }
        )
        _reset_enforcement_caches()
        load_gateway_keyring_cached.cache_clear()

        client = TestClient(auth_server_module.app)
        response = client.get(
            "/validate",
            headers=_headers_for(
                extra={
                    "X-Gateway-Token": token,
                    "X-EnforceAI-Server-Path": "/fininfo",
                    "X-EnforceAI-Upstream-Auth-Type": "api-key",
                    "X-EnforceAI-Upstream-Credential-Binding": "user",
                    "X-EnforceAI-Upstream-Header-Name": "X-API-Key",
                }
            ),
        )
        assert response.status_code == 424
        assert response.headers.get("X-EnforceAI-Error-Code") == "UPSTREAM_CREDENTIALS_REQUIRED"

    def test_validate_sets_bearer_injection_headers_for_jwt(
        self,
        tmp_path: Path,
        enforceai_env,
        enforceai_sqlite_db_path: Path,
        enforceai_gateway_key_files,
    ) -> None:
        catalog_path = _write_scope_catalog(path=tmp_path / "scopes.yml")
        upstream_kek_path = tmp_path / "upstream_kek"
        upstream_kek_path.write_text("33" * 32)
        upstream_kek = load_upstream_kek(upstream_kek_path)

        data_layer = EnforceAIDataLayer(db_path=enforceai_sqlite_db_path)
        data_layer.initialize()
        stores = data_layer.build_stores(upstream_kek=upstream_kek)
        assert stores.upstream_credential_store is not None

        user_id = "https://issuer.example|user-1"
        agent_id = str(uuid.uuid4())
        stores.agent_store.create_agent(
            user_id=user_id,
            agent_id=agent_id,
            scopes=["scope-good"],
        )
        stores.upstream_credential_store.create_credential(
            server_path="/fininfo",
            credential_type="jwt",
            credential_binding="service",
            secret_payload={"token": "jwt-1"},
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
            jti="jti-upstream-3",
        )

        enforceai_env(
            {
                "ENFORCEAI_DB_PATH": str(enforceai_sqlite_db_path),
                "ENFORCEAI_AUTH_PROVIDER": "gateway-token",
                "ENFORCEAI_SCOPES_CATALOG_PATH": str(catalog_path),
                "ENFORCEAI_UPSTREAM_KEK_PATH": str(upstream_kek_path),
                "ENFORCEAI_GATEWAY_PRIVATE_KEY_PATH": str(key_files.private_key_path),
                "ENFORCEAI_GATEWAY_PUBLIC_KEYS_DIR": str(key_files.public_keys_dir),
                "ENFORCEAI_GATEWAY_ACTIVE_KID": key_files.active_kid,
                "ENFORCEAI_GATEWAY_ISSUER": "enforceai-gateway",
            }
        )
        _reset_enforcement_caches()
        load_gateway_keyring_cached.cache_clear()

        client = TestClient(auth_server_module.app)
        response = client.get(
            "/validate",
            headers=_headers_for(
                extra={
                    "X-Gateway-Token": token,
                    "X-EnforceAI-Server-Path": "/fininfo",
                    "X-EnforceAI-Upstream-Auth-Type": "jwt",
                    "X-EnforceAI-Upstream-Credential-Binding": "service",
                    "X-EnforceAI-Upstream-Header-Name": "Authorization",
                    "X-EnforceAI-Upstream-Scheme": "Bearer",
                }
            ),
        )
        assert response.status_code == 200
        assert response.headers.get("X-EnforceAI-Upstream-Mode") == "bearer"
        assert response.headers.get("X-EnforceAI-Upstream-Authorization") == "Bearer jwt-1"
