"""
Integration tests for EnforceAI Audit Events API.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import pytest
from starlette.testclient import TestClient

import auth_server.server as auth_server_module
from gateway_session import build_session_cookie_payload
from auth_server.enforceai.auth import dependency as enforceai_dependency
from auth_server.enforceai.crypto.keyring import load_gateway_keyring_cached
from auth_server.enforceai.db.data_layer import EnforceAIDataLayer, EnforceAIStores
from auth_server.enforceai.db.migrations import upgrade_to_latest
from auth_server.enforceai.fgac.catalog import clear_scope_catalog_cache


def _migrate_db(
    db_path: Path,
) -> None:
    connection = sqlite3.connect(db_path)
    try:
        upgrade_to_latest(connection)
    finally:
        connection.close()


def _reset_enforcement_caches() -> None:
    enforceai_dependency.clear_enforceai_dependency_caches()
    clear_scope_catalog_cache()
    load_gateway_keyring_cached.cache_clear()
    auth_server_module._load_enforceai_runtime.cache_clear()


def _write_scope_catalog(
    *,
    path: Path,
    scope_name: str = "test-scope",
    tool_name: str = "test_tool",
) -> Path:
    content = "\n".join(
        [
            "UI-Scopes: {}",
            "group_mappings: {}",
            f"{scope_name}:",
            "  - server: test-server",
            "    methods: [tools/list, tools/call]",
            f"    tools: [{tool_name}]",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _make_cookie_client(
    *,
    stores: EnforceAIStores,
    session_id: str,
    user_id: str,
    groups: list[str],
    username: str,
    email: str,
) -> TestClient:
    stores.session_store.create_session(
        session_id=session_id,
        user_id=user_id,
        auth_method="oidc",
        expires_at=datetime.now(timezone.utc).replace(microsecond=0)
        + timedelta(hours=1),
    )

    cookie_payload = build_session_cookie_payload(
        username=username,
        email=email,
        name=None,
        groups=groups,
        provider="keycloak",
        legacy_auth_method="oauth2",
        max_age_seconds=28800,
        session_id=session_id,
        user_id=user_id,
    )
    cookie_value = auth_server_module.signer.dumps(cookie_payload)

    client = TestClient(auth_server_module.app)
    client.cookies.set("mcp_gateway_session", cookie_value)
    return client


@pytest.mark.integration
class TestEnforceAIAuditEventsAPI:
    def test_list_audit_events_requires_authentication(
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
        _migrate_db(enforceai_sqlite_db_path)

        client = TestClient(auth_server_module.app)
        response = client.get("/enforceai/audit/events")

        assert response.status_code == 401

    def test_list_audit_events_returns_user_events(
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
        user_id = "https://issuer.example|test-user-1"
        agent_id = str(uuid.uuid4())

        stores.audit_store.append_event(
            occurred_at=datetime.now(timezone.utc),
            user_id=user_id,
            agent_id=agent_id,
            action="tools/list",
            outcome="allow",
            request_id="req-1",
            details={"server": "sqlite"},
        )

        other_user_id = "https://issuer.example|other-user"
        other_agent_id = str(uuid.uuid4())
        stores.audit_store.append_event(
            occurred_at=datetime.now(timezone.utc),
            user_id=other_user_id,
            agent_id=other_agent_id,
            action="tools/call",
            outcome="deny",
            request_id="req-2",
        )

        client = _make_cookie_client(
            stores=stores,
            session_id=session_id,
            user_id=user_id,
            groups=[],
            username="test-user-1",
            email="test-user-1@example.com",
        )

        response = client.get("/enforceai/audit/events")

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "next_cursor" in data
        assert "server_time" in data

        assert len(data["items"]) == 1
        assert data["items"][0]["user_id"] == user_id
        assert data["items"][0]["action"] == "tools/list"
        assert data["items"][0]["outcome"] == "allow"
        assert data["items"][0]["request_id"] == "req-1"

    def test_list_audit_events_filters_by_action(
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
        user_id = "https://issuer.example|test-user-2"
        agent_id = str(uuid.uuid4())

        stores.audit_store.append_event(
            occurred_at=datetime.now(timezone.utc),
            user_id=user_id,
            agent_id=agent_id,
            action="tools/list",
            outcome="allow",
        )
        stores.audit_store.append_event(
            occurred_at=datetime.now(timezone.utc),
            user_id=user_id,
            agent_id=agent_id,
            action="tools/call",
            outcome="deny",
        )
        stores.audit_store.append_event(
            occurred_at=datetime.now(timezone.utc),
            user_id=user_id,
            agent_id=agent_id,
            action="management/agents/create",
            outcome="allow",
        )

        client = _make_cookie_client(
            stores=stores,
            session_id=session_id,
            user_id=user_id,
            groups=[],
            username="test-user-2",
            email="test-user-2@example.com",
        )

        response = client.get("/enforceai/audit/events?action=tools/list&action=tools/call")

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 2
        actions = {item["action"] for item in data["items"]}
        assert actions == {"tools/list", "tools/call"}

    def test_list_audit_events_filters_by_outcome(
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
        user_id = "https://issuer.example|test-user-3"
        agent_id = str(uuid.uuid4())

        stores.audit_store.append_event(
            occurred_at=datetime.now(timezone.utc),
            user_id=user_id,
            agent_id=agent_id,
            action="tools/call",
            outcome="allow",
        )
        stores.audit_store.append_event(
            occurred_at=datetime.now(timezone.utc),
            user_id=user_id,
            agent_id=agent_id,
            action="tools/call",
            outcome="deny",
        )

        client = _make_cookie_client(
            stores=stores,
            session_id=session_id,
            user_id=user_id,
            groups=[],
            username="test-user-3",
            email="test-user-3@example.com",
        )

        response = client.get("/enforceai/audit/events?outcome=deny")

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1
        assert data["items"][0]["outcome"] == "deny"

    def test_list_audit_events_pagination(
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
        user_id = "https://issuer.example|test-user-4"
        agent_id = str(uuid.uuid4())

        for i in range(5):
            stores.audit_store.append_event(
                occurred_at=datetime.now(timezone.utc) - timedelta(seconds=i),
                user_id=user_id,
                agent_id=agent_id,
                action="tools/list",
                outcome="allow",
                request_id=f"req-{i}",
            )

        client = _make_cookie_client(
            stores=stores,
            session_id=session_id,
            user_id=user_id,
            groups=[],
            username="test-user-4",
            email="test-user-4@example.com",
        )

        page1 = client.get("/enforceai/audit/events?limit=2")
        assert page1.status_code == 200
        data1 = page1.json()
        assert len(data1["items"]) == 2
        assert data1["next_cursor"] is not None

        page2 = client.get(f"/enforceai/audit/events?limit=2&cursor={data1['next_cursor']}")
        assert page2.status_code == 200
        data2 = page2.json()
        assert len(data2["items"]) == 2
        assert data2["next_cursor"] is not None

        page1_ids = {item["event_id"] for item in data1["items"]}
        page2_ids = {item["event_id"] for item in data2["items"]}
        assert page1_ids.isdisjoint(page2_ids)

    def test_list_audit_events_allows_large_lookback_windows(
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
        user_id = "https://issuer.example|test-user-5"

        client = _make_cookie_client(
            stores=stores,
            session_id=session_id,
            user_id=user_id,
            groups=[],
            username="test-user-5",
            email="test-user-5@example.com",
        )

        old_since = quote((datetime.now(timezone.utc) - timedelta(days=10)).isoformat())
        response = client.get(f"/enforceai/audit/events?since={old_since}")

        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert "server_time" in data

    def test_list_audit_events_rejects_invalid_time_range(
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
        user_id = "https://issuer.example|test-user-6"

        client = _make_cookie_client(
            stores=stores,
            session_id=session_id,
            user_id=user_id,
            groups=[],
            username="test-user-6",
            email="test-user-6@example.com",
        )

        now = datetime.now(timezone.utc)
        since = quote((now + timedelta(hours=1)).isoformat())
        until = quote(now.isoformat())

        response = client.get(f"/enforceai/audit/events?since={since}&until={until}")

        assert response.status_code == 400
        assert "since" in response.json()["detail"].lower()

    def test_list_audit_events_filters_by_request_id(
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
        user_id = "https://issuer.example|test-user-7"
        agent_id = str(uuid.uuid4())
        target_request_id = "req-target-123"

        stores.audit_store.append_event(
            occurred_at=datetime.now(timezone.utc),
            user_id=user_id,
            agent_id=agent_id,
            action="tools/list",
            outcome="allow",
            request_id=target_request_id,
        )
        stores.audit_store.append_event(
            occurred_at=datetime.now(timezone.utc),
            user_id=user_id,
            agent_id=agent_id,
            action="tools/call",
            outcome="allow",
            request_id=target_request_id,
        )
        stores.audit_store.append_event(
            occurred_at=datetime.now(timezone.utc),
            user_id=user_id,
            agent_id=agent_id,
            action="tools/call",
            outcome="deny",
            request_id="req-other",
        )

        client = _make_cookie_client(
            stores=stores,
            session_id=session_id,
            user_id=user_id,
            groups=[],
            username="test-user-7",
            email="test-user-7@example.com",
        )

        response = client.get(f"/enforceai/audit/events?request_id={target_request_id}")

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 2
        for item in data["items"]:
            assert item["request_id"] == target_request_id

    def test_list_audit_events_filters_by_server_and_tool(
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
        user_id = "https://issuer.example|test-user-8"
        agent_id = str(uuid.uuid4())

        stores.audit_store.append_event(
            occurred_at=datetime.now(timezone.utc),
            user_id=user_id,
            agent_id=agent_id,
            action="tools/call",
            outcome="allow",
            details={"server": "sqlite", "tool": "query"},
        )
        stores.audit_store.append_event(
            occurred_at=datetime.now(timezone.utc),
            user_id=user_id,
            agent_id=agent_id,
            action="tools/call",
            outcome="allow",
            details={"server": "sqlite", "tool": "execute"},
        )
        stores.audit_store.append_event(
            occurred_at=datetime.now(timezone.utc),
            user_id=user_id,
            agent_id=agent_id,
            action="tools/call",
            outcome="allow",
            details={"server": "github", "tool": "create_issue"},
        )

        client = _make_cookie_client(
            stores=stores,
            session_id=session_id,
            user_id=user_id,
            groups=[],
            username="test-user-8",
            email="test-user-8@example.com",
        )

        response = client.get("/enforceai/audit/events?server=sqlite&tool=query")

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1
        assert data["items"][0]["details"]["server"] == "sqlite"
        assert data["items"][0]["details"]["tool"] == "query"


@pytest.mark.integration
class TestEnforceAIAdminAuditEventsAPI:
    """Tests for the admin audit events API."""

    def test_admin_audit_events_requires_admin(
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
        user_id = "https://issuer.example|non-admin-user"

        client = _make_cookie_client(
            stores=stores,
            session_id=session_id,
            user_id=user_id,
            groups=[],  # No admin group
            username="non-admin",
            email="non-admin@example.com",
        )

        response = client.get("/enforceai/admin/audit/events")

        assert response.status_code == 403

    def test_admin_audit_events_returns_all_users_events(
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

        # Create events for multiple users
        user1_id = "https://issuer.example|user-1"
        user2_id = "https://issuer.example|user-2"
        agent_id = str(uuid.uuid4())

        stores.audit_store.append_event(
            occurred_at=datetime.now(timezone.utc),
            user_id=user1_id,
            agent_id=agent_id,
            action="tools/list",
            outcome="allow",
        )
        stores.audit_store.append_event(
            occurred_at=datetime.now(timezone.utc),
            user_id=user2_id,
            agent_id=agent_id,
            action="tools/call",
            outcome="deny",
        )

        # Admin user
        admin_user_id = "https://issuer.example|admin-user"
        session_id = str(uuid.uuid4())

        client = _make_cookie_client(
            stores=stores,
            session_id=session_id,
            user_id=admin_user_id,
            groups=["enforceai-admin"],  # Admin group
            username="admin",
            email="admin@example.com",
        )

        response = client.get("/enforceai/admin/audit/events")

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) >= 2  # Both users' events
        user_ids = {item["user_id"] for item in data["items"]}
        assert user1_id in user_ids
        assert user2_id in user_ids

    def test_admin_audit_events_filters_by_user_id(
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

        # Create events for multiple users
        user1_id = "https://issuer.example|filter-user-1"
        user2_id = "https://issuer.example|filter-user-2"
        agent_id = str(uuid.uuid4())

        stores.audit_store.append_event(
            occurred_at=datetime.now(timezone.utc),
            user_id=user1_id,
            agent_id=agent_id,
            action="tools/list",
            outcome="allow",
        )
        stores.audit_store.append_event(
            occurred_at=datetime.now(timezone.utc),
            user_id=user2_id,
            agent_id=agent_id,
            action="tools/call",
            outcome="deny",
        )

        # Admin user
        admin_user_id = "https://issuer.example|admin-filter-user"
        session_id = str(uuid.uuid4())

        client = _make_cookie_client(
            stores=stores,
            session_id=session_id,
            user_id=admin_user_id,
            groups=["enforceai-admin"],
            username="admin",
            email="admin@example.com",
        )

        # Filter by user1
        encoded_user_id = quote(user1_id, safe="")
        response = client.get(f"/enforceai/admin/audit/events?user_id={encoded_user_id}")

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1
        assert data["items"][0]["user_id"] == user1_id


@pytest.mark.integration
class TestEnforceAIAuditExportAPI:
    """Tests for the admin audit CSV export API."""

    def test_export_requires_admin(
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
        user_id = "https://issuer.example|non-admin-export"

        client = _make_cookie_client(
            stores=stores,
            session_id=session_id,
            user_id=user_id,
            groups=[],  # No admin group
            username="non-admin",
            email="non-admin@example.com",
        )

        response = client.get("/enforceai/admin/audit/events/export")

        assert response.status_code == 403

    def test_export_returns_csv(
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

        # Create test events
        user_id = "https://issuer.example|export-user"
        agent_id = str(uuid.uuid4())

        stores.audit_store.append_event(
            occurred_at=datetime.now(timezone.utc),
            user_id=user_id,
            agent_id=agent_id,
            action="tools/list",
            outcome="allow",
        )
        stores.audit_store.append_event(
            occurred_at=datetime.now(timezone.utc),
            user_id=user_id,
            agent_id=agent_id,
            action="tools/call",
            outcome="deny",
        )

        # Admin user
        admin_user_id = "https://issuer.example|admin-export"
        session_id = str(uuid.uuid4())

        client = _make_cookie_client(
            stores=stores,
            session_id=session_id,
            user_id=admin_user_id,
            groups=["enforceai-admin"],
            username="admin",
            email="admin@example.com",
        )

        response = client.get("/enforceai/admin/audit/events/export")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/csv; charset=utf-8"
        assert "attachment" in response.headers["content-disposition"]
        assert ".csv" in response.headers["content-disposition"]

        # Parse CSV content
        import csv
        import io
        content = response.text
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)

        # First row is header
        assert rows[0] == [
            "event_id",
            "occurred_at",
            "user_id",
            "agent_id",
            "action",
            "outcome",
            "request_id",
            "server",
            "tool",
            "reason",
            "matched_scope",
            "provider",
            "details_json",
        ]

        # Should have header + at least 2 data rows
        assert len(rows) >= 3
