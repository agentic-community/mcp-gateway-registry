"""
Stage 6.2 integration tests: management API routes.

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
from datetime import (
    datetime,
    timedelta,
    timezone,
)
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
)
from auth_server.enforceai.fgac.catalog import (
    clear_scope_catalog_cache,
)
from auth_server.enforceai.oidc.jwks import JWKSCache
from auth_server.enforceai.tokens.mint import (
    mint_gateway_token,
)
from gateway_csrf import (
    mint_csrf_token,
)
from gateway_session import (
    build_session_cookie_payload,
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
class TestEnforceAIManagementRoutes:
    def test_cookie_session_can_manage_agents_and_enforces_admin(
        self,
        tmp_path: Path,
        enforceai_env,
        enforceai_sqlite_db_path: Path,
        enforceai_gateway_key_files,
    ) -> None:
        catalog_path = _write_scope_catalog(path=tmp_path / "scopes.yml")

        enforceai_env(
            {
                "ENFORCEAI_DB_PATH": str(enforceai_sqlite_db_path),
                "ENFORCEAI_AUTH_PROVIDER": "gateway-token",
                "ENFORCEAI_SCOPES_CATALOG_PATH": str(catalog_path),
                "ENFORCEAI_GATEWAY_PRIVATE_KEY_PATH": str(
                    enforceai_gateway_key_files.private_key_path
                ),
                "ENFORCEAI_GATEWAY_PUBLIC_KEYS_DIR": str(
                    enforceai_gateway_key_files.public_keys_dir
                ),
                "ENFORCEAI_GATEWAY_ACTIVE_KID": enforceai_gateway_key_files.active_kid,
                "ENFORCEAI_GATEWAY_ISSUER": "enforceai-gateway",
            }
        )
        _reset_enforcement_caches()

        data_layer = EnforceAIDataLayer(db_path=enforceai_sqlite_db_path)
        data_layer.initialize()
        stores = data_layer.build_stores()

        session_id = str(uuid.uuid4())
        user_id = "https://issuer.example|cookie-user-1"
        stores.session_store.create_session(
            session_id=session_id,
            user_id=user_id,
            auth_method="oidc",
            expires_at=datetime.now(timezone.utc).replace(microsecond=0) + timedelta(hours=1),
        )

        cookie_payload = build_session_cookie_payload(
            username="cookie-user-1",
            email="cookie-user-1@example.com",
            name=None,
            groups=["enforceai-admin"],
            provider="keycloak",
            legacy_auth_method="oauth2",
            max_age_seconds=28800,
            session_id=session_id,
            user_id=user_id,
        )
        cookie_value = auth_server_module.signer.dumps(cookie_payload)

        client = TestClient(auth_server_module.app)
        client.cookies.set("mcp_gateway_session", cookie_value)

        list_response = client.get("/enforceai/agents")
        assert list_response.status_code == 200
        assert list_response.json() == []

        csrf_token = mint_csrf_token(
            secret_key=auth_server_module.SECRET_KEY,
            session_id=session_id,
        )
        create_response = client.post(
            "/enforceai/agents",
            headers={"X-CSRF-Token": csrf_token},
            json={"scopes": ["scope-mgmt"]},
        )
        assert create_response.status_code == 200
        assert create_response.json()["user_id"] == user_id

        admin_ping = client.get("/enforceai/admin/ping")
        assert admin_ping.status_code == 200
        assert admin_ping.json() == {"ok": True}

        non_admin_session_id = str(uuid.uuid4())
        non_admin_user_id = "https://issuer.example|cookie-user-2"
        stores.session_store.create_session(
            session_id=non_admin_session_id,
            user_id=non_admin_user_id,
            auth_method="oidc",
            expires_at=datetime.now(timezone.utc).replace(microsecond=0) + timedelta(hours=1),
        )
        non_admin_cookie_payload = build_session_cookie_payload(
            username="cookie-user-2",
            email="cookie-user-2@example.com",
            name=None,
            groups=[],
            provider="keycloak",
            legacy_auth_method="oauth2",
            max_age_seconds=28800,
            session_id=non_admin_session_id,
            user_id=non_admin_user_id,
        )
        non_admin_cookie_value = auth_server_module.signer.dumps(non_admin_cookie_payload)
        client.cookies.set("mcp_gateway_session", non_admin_cookie_value)

        denied = client.get("/enforceai/admin/ping")
        assert denied.status_code == 403
        assert denied.json()["detail"] == "Admin required"

    def test_admin_user_directory_and_cross_user_operations(
        self,
        tmp_path: Path,
        enforceai_env,
        enforceai_sqlite_db_path: Path,
        enforceai_gateway_key_files,
    ) -> None:
        catalog_path = _write_scope_catalog(path=tmp_path / "scopes.yml")
        pepper_path = tmp_path / "pepper"
        pepper_path.write_bytes(b"pepper-1")

        enforceai_env(
            {
                "ENFORCEAI_DB_PATH": str(enforceai_sqlite_db_path),
                "ENFORCEAI_AUTH_PROVIDER": "gateway-token",
                "ENFORCEAI_SCOPES_CATALOG_PATH": str(catalog_path),
                "ENFORCEAI_API_KEY_PEPPER_PATH": str(pepper_path),
                "ENFORCEAI_GATEWAY_PRIVATE_KEY_PATH": str(
                    enforceai_gateway_key_files.private_key_path
                ),
                "ENFORCEAI_GATEWAY_PUBLIC_KEYS_DIR": str(
                    enforceai_gateway_key_files.public_keys_dir
                ),
                "ENFORCEAI_GATEWAY_ACTIVE_KID": enforceai_gateway_key_files.active_kid,
                "ENFORCEAI_GATEWAY_ISSUER": "enforceai-gateway",
            }
        )
        _reset_enforcement_caches()

        data_layer = EnforceAIDataLayer(db_path=enforceai_sqlite_db_path)
        data_layer.initialize()
        stores = data_layer.build_stores()

        target_user_id = "https://issuer.example|target-user-1"
        stores.user_store.upsert_oidc_user(
            user_id=target_user_id,
            email="target-user-1@example.com",
            role="user",
        )

        session_id = str(uuid.uuid4())
        admin_user_id = "https://issuer.example|admin-user-1"
        stores.session_store.create_session(
            session_id=session_id,
            user_id=admin_user_id,
            auth_method="oidc",
            expires_at=datetime.now(timezone.utc).replace(microsecond=0) + timedelta(hours=1),
        )
        cookie_payload = build_session_cookie_payload(
            username="admin-user-1",
            email="admin-user-1@example.com",
            name=None,
            groups=["enforceai-admin"],
            provider="keycloak",
            legacy_auth_method="oauth2",
            max_age_seconds=28800,
            session_id=session_id,
            user_id=admin_user_id,
        )
        cookie_value = auth_server_module.signer.dumps(cookie_payload)

        client = TestClient(auth_server_module.app)
        client.cookies.set("mcp_gateway_session", cookie_value)
        csrf_token = mint_csrf_token(
            secret_key=auth_server_module.SECRET_KEY,
            session_id=session_id,
        )

        search = client.get("/enforceai/admin/users", params={"query": "target-user-1"})
        assert search.status_code == 200
        results = search.json()
        assert any(item["user_id"] == target_user_id for item in results)

        detail = client.get(f"/enforceai/admin/users/{target_user_id}")
        assert detail.status_code == 200
        assert detail.json()["email"] == "target-user-1@example.com"

        list_agents = client.get(f"/enforceai/admin/users/{target_user_id}/agents")
        assert list_agents.status_code == 200
        assert list_agents.json() == []

        created = client.post(
            f"/enforceai/admin/users/{target_user_id}/agents",
            headers={"X-CSRF-Token": csrf_token},
            json={"scopes": ["scope-mgmt"], "alias": "target-agent"},
        )
        assert created.status_code == 200
        created_agent_id = created.json()["agent_id"]
        assert created.json()["user_id"] == target_user_id

        key = client.post(
            f"/enforceai/admin/users/{target_user_id}/agents/{created_agent_id}/api-keys",
            headers={"X-CSRF-Token": csrf_token},
            json={"scopes": ["scope-mgmt"]},
        )
        assert key.status_code == 200
        key_id = key.json()["key_id"]

        revoke_key = client.post(
            f"/enforceai/admin/users/{target_user_id}/api-keys/{key_id}/revoke",
            headers={"X-CSRF-Token": csrf_token},
        )
        assert revoke_key.status_code == 200
        assert revoke_key.json()["key_id"] == key_id

        revoke_token = client.post(
            f"/enforceai/admin/users/{target_user_id}/tokens/revoke",
            headers={"X-CSRF-Token": csrf_token},
            json={"agent_id": created_agent_id, "jti": "jti-1", "reason": "test"},
        )
        assert revoke_token.status_code == 200
        assert revoke_token.json()["jti"] == "jti-1"

        revoke_agent = client.post(
            f"/enforceai/admin/users/{target_user_id}/agents/{created_agent_id}/revoke",
            headers={"X-CSRF-Token": csrf_token},
        )
        assert revoke_agent.status_code == 200
        assert revoke_agent.json()["revoked_at"] is not None

    def test_oidc_management_happy_path_and_cross_user_denied(
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

        other_user_agent_id = str(uuid.uuid4())
        stores.agent_store.create_agent(
            user_id=f"{issuer}|user-2",
            agent_id=other_user_agent_id,
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
                "ENFORCEAI_GATEWAY_PRIVATE_KEY_PATH": str(
                    enforceai_gateway_key_files.private_key_path
                ),
                "ENFORCEAI_GATEWAY_PUBLIC_KEYS_DIR": str(
                    enforceai_gateway_key_files.public_keys_dir
                ),
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

        client = TestClient(auth_server_module.app)

        list_response = client.get(
            "/enforceai/agents",
            headers={
                "Authorization": f"Bearer {oidc_token}",
                "X-Agent-Id": bootstrap_agent_id,
            },
        )
        assert list_response.status_code == 200
        assert len(list_response.json()) == 1

        create_response = client.post(
            "/enforceai/agents",
            headers={
                "Authorization": f"Bearer {oidc_token}",
                "X-Agent-Id": bootstrap_agent_id,
            },
            json={"scopes": ["scope-mgmt"], "alias": "agent-2"},
        )
        assert create_response.status_code == 200
        created_agent_id = create_response.json()["agent_id"]
        assert created_agent_id

        create_key_response = client.post(
            f"/enforceai/agents/{created_agent_id}/api-keys",
            headers={
                "Authorization": f"Bearer {oidc_token}",
                "X-Agent-Id": bootstrap_agent_id,
            },
            json={"scopes": ["scope-mgmt"]},
        )
        assert create_key_response.status_code == 200
        api_key_value = create_key_response.json()["api_key_value"]
        assert api_key_value.startswith("eak_")

        mint_response = client.post(
            f"/enforceai/agents/{created_agent_id}/tokens/mint",
            headers={
                "Authorization": f"Bearer {oidc_token}",
                "X-Agent-Id": bootstrap_agent_id,
            },
            json={"scopes": ["scope-mgmt"], "ttl_seconds": 60},
        )
        assert mint_response.status_code == 200
        gateway_token = mint_response.json()["token"]
        assert isinstance(gateway_token, str)
        assert gateway_token

        revoke_token_response = client.post(
            "/enforceai/tokens/revoke",
            headers={
                "Authorization": f"Bearer {oidc_token}",
                "X-Agent-Id": bootstrap_agent_id,
            },
            json={"gateway_token": gateway_token},
        )
        assert revoke_token_response.status_code == 200
        assert revoke_token_response.json()["agent_id"] == created_agent_id
        assert revoke_token_response.json()["jti"]

        revoke_all_response = client.post(
            f"/enforceai/agents/{created_agent_id}/tokens/revoke-all",
            headers={
                "Authorization": f"Bearer {oidc_token}",
                "X-Agent-Id": bootstrap_agent_id,
            },
        )
        assert revoke_all_response.status_code == 200
        assert revoke_all_response.json()["tokens_valid_after"] is not None

        forbidden = client.get(
            f"/enforceai/agents/{other_user_agent_id}",
            headers={
                "Authorization": f"Bearer {oidc_token}",
                "X-Agent-Id": bootstrap_agent_id,
            },
        )
        assert forbidden.status_code == 403

        list_keys_response = client.get(
            f"/enforceai/agents/{created_agent_id}/api-keys",
            headers={
                "Authorization": f"Bearer {oidc_token}",
                "X-Agent-Id": bootstrap_agent_id,
            },
        )
        assert list_keys_response.status_code == 200
        keys_payload = list_keys_response.json()
        assert isinstance(keys_payload, list)
        assert len(keys_payload) == 1
        assert "secret" not in keys_payload[0]
        assert "api_key_value" not in keys_payload[0]

        revoke_key_response = client.post(
            f"/enforceai/api-keys/{create_key_response.json()['key_id']}/revoke",
            headers={
                "Authorization": f"Bearer {oidc_token}",
                "X-Agent-Id": bootstrap_agent_id,
            },
        )
        assert revoke_key_response.status_code == 200
        revoked_payload = revoke_key_response.json()
        assert "secret" not in revoked_payload
        assert "api_key_value" not in revoked_payload

    def test_gateway_token_auth_can_manage_agents(
        self,
        tmp_path: Path,
        enforceai_env,
        enforceai_sqlite_db_path: Path,
        enforceai_gateway_key_files,
    ) -> None:
        catalog_path = _write_scope_catalog(path=tmp_path / "scopes.yml")
        pepper_path = tmp_path / "pepper"
        pepper_path.write_bytes(b"pepper-1")

        data_layer = EnforceAIDataLayer(db_path=enforceai_sqlite_db_path)
        data_layer.initialize()
        stores = data_layer.build_stores()

        user_id = "https://issuer.example|user-1"
        bootstrap_agent_id = str(uuid.uuid4())
        stores.agent_store.create_agent(
            user_id=user_id,
            agent_id=bootstrap_agent_id,
            scopes=["scope-mgmt"],
        )

        keyring = GatewayKeyring.load(
            private_key_path=enforceai_gateway_key_files.private_key_path,
            public_keys_dir=enforceai_gateway_key_files.public_keys_dir,
            active_kid=enforceai_gateway_key_files.active_kid,
        )

        gateway_token = mint_gateway_token(
            keyring=keyring,
            issuer="enforceai-gateway",
            user_id=user_id,
            agent_id=bootstrap_agent_id,
            scopes=["scope-mgmt"],
            ttl_seconds=3600,
        )

        enforceai_env(
            {
                "ENFORCEAI_DB_PATH": str(enforceai_sqlite_db_path),
                "ENFORCEAI_AUTH_PROVIDER": "gateway-token",
                "ENFORCEAI_SCOPES_CATALOG_PATH": str(catalog_path),
                "ENFORCEAI_API_KEY_PEPPER_PATH": str(pepper_path),
                "ENFORCEAI_GATEWAY_PRIVATE_KEY_PATH": str(
                    enforceai_gateway_key_files.private_key_path
                ),
                "ENFORCEAI_GATEWAY_PUBLIC_KEYS_DIR": str(
                    enforceai_gateway_key_files.public_keys_dir
                ),
                "ENFORCEAI_GATEWAY_ACTIVE_KID": enforceai_gateway_key_files.active_kid,
                "ENFORCEAI_GATEWAY_ISSUER": "enforceai-gateway",
            }
        )
        _reset_enforcement_caches()

        client = TestClient(auth_server_module.app)

        create_response = client.post(
            "/enforceai/agents",
            headers={"Authorization": f"Bearer {gateway_token}"},
            json={"scopes": ["scope-mgmt"], "alias": "agent-2"},
        )
        assert create_response.status_code == 200
        assert create_response.json()["user_id"] == user_id

    def test_api_key_auth_dependency_failure_returns_503(
        self,
        tmp_path: Path,
        enforceai_env,
        enforceai_sqlite_db_path: Path,
        enforceai_gateway_key_files,
    ) -> None:
        catalog_path = _write_scope_catalog(path=tmp_path / "scopes.yml")
        pepper = b"pepper-1"
        pepper_path = tmp_path / "pepper"
        pepper_path.write_bytes(pepper)

        data_layer = EnforceAIDataLayer(db_path=enforceai_sqlite_db_path)
        data_layer.initialize()
        stores = data_layer.build_stores()

        user_id = "https://issuer.example|user-1"
        bootstrap_agent_id = str(uuid.uuid4())
        stores.agent_store.create_agent(
            user_id=user_id,
            agent_id=bootstrap_agent_id,
            scopes=["scope-mgmt"],
        )
        key_secret = "secret-1"
        stores.api_key_store.create_key(
            key_id="key-1",
            secret_hash=_compute_api_key_hash(
                pepper=pepper,
                secret=key_secret,
            ),
            user_id=user_id,
            agent_id=bootstrap_agent_id,
            scopes=["scope-mgmt"],
        )

        enforceai_env(
            {
                "ENFORCEAI_DB_PATH": str(enforceai_sqlite_db_path),
                "ENFORCEAI_AUTH_PROVIDER": "api-key",
                "ENFORCEAI_SCOPES_CATALOG_PATH": str(catalog_path),
                "ENFORCEAI_API_KEY_PEPPER_PATH": str(pepper_path),
                "ENFORCEAI_GATEWAY_PRIVATE_KEY_PATH": str(
                    enforceai_gateway_key_files.private_key_path
                ),
                "ENFORCEAI_GATEWAY_PUBLIC_KEYS_DIR": str(
                    enforceai_gateway_key_files.public_keys_dir
                ),
                "ENFORCEAI_GATEWAY_ACTIVE_KID": enforceai_gateway_key_files.active_kid,
                "ENFORCEAI_GATEWAY_ISSUER": "enforceai-gateway",
            }
        )
        _reset_enforcement_caches()

        client = TestClient(auth_server_module.app)

        warm = client.get(
            "/enforceai/agents",
            headers={"X-API-Key": f"eak_key-1.{key_secret}"},
        )
        assert warm.status_code == 200

        enforceai_sqlite_db_path.unlink()

        after_delete = client.get(
            "/enforceai/agents",
            headers={"X-API-Key": f"eak_key-1.{key_secret}"},
        )
        assert after_delete.status_code == 503

    def test_audit_failure_is_best_effort_does_not_block_management(
        self,
        tmp_path: Path,
        enforceai_env,
        enforceai_sqlite_db_path: Path,
        enforceai_gateway_key_files,
    ) -> None:
        catalog_path = _write_scope_catalog(path=tmp_path / "scopes.yml")
        pepper_path = tmp_path / "pepper"
        pepper_path.write_bytes(b"pepper-1")

        data_layer = EnforceAIDataLayer(db_path=enforceai_sqlite_db_path)
        data_layer.initialize()
        stores = data_layer.build_stores()

        user_id = "https://issuer.example|user-1"
        bootstrap_agent_id = str(uuid.uuid4())
        stores.agent_store.create_agent(
            user_id=user_id,
            agent_id=bootstrap_agent_id,
            scopes=["scope-mgmt"],
        )

        keyring = GatewayKeyring.load(
            private_key_path=enforceai_gateway_key_files.private_key_path,
            public_keys_dir=enforceai_gateway_key_files.public_keys_dir,
            active_kid=enforceai_gateway_key_files.active_kid,
        )

        gateway_token = mint_gateway_token(
            keyring=keyring,
            issuer="enforceai-gateway",
            user_id=user_id,
            agent_id=bootstrap_agent_id,
            scopes=["scope-mgmt"],
            ttl_seconds=3600,
        )

        enforceai_env(
            {
                "ENFORCEAI_DB_PATH": str(enforceai_sqlite_db_path),
                "ENFORCEAI_AUTH_PROVIDER": "gateway-token",
                "ENFORCEAI_SCOPES_CATALOG_PATH": str(catalog_path),
                "ENFORCEAI_API_KEY_PEPPER_PATH": str(pepper_path),
                "ENFORCEAI_GATEWAY_PRIVATE_KEY_PATH": str(
                    enforceai_gateway_key_files.private_key_path
                ),
                "ENFORCEAI_GATEWAY_PUBLIC_KEYS_DIR": str(
                    enforceai_gateway_key_files.public_keys_dir
                ),
                "ENFORCEAI_GATEWAY_ACTIVE_KID": enforceai_gateway_key_files.active_kid,
                "ENFORCEAI_GATEWAY_ISSUER": "enforceai-gateway",
            }
        )
        _reset_enforcement_caches()

        class ExplodingAuditStore:
            def append_event(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                raise RuntimeError("audit down")

        overriding_stores = type(stores)(
            agent_store=stores.agent_store,
            api_key_store=stores.api_key_store,
            revocation_store=stores.revocation_store,
            audit_store=ExplodingAuditStore(),
            user_store=stores.user_store,
            session_store=stores.session_store,
        )

        auth_server_module.app.dependency_overrides[
            enforceai_dependency.get_enforceai_stores
        ] = lambda: overriding_stores
        try:
            client = TestClient(auth_server_module.app)
            response = client.get(
                "/enforceai/agents",
                headers={"Authorization": f"Bearer {gateway_token}"},
            )
            assert response.status_code == 200
        finally:
            auth_server_module.app.dependency_overrides.pop(
                enforceai_dependency.get_enforceai_stores,
                None,
            )
