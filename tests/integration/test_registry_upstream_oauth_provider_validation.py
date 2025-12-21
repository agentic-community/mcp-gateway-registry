"""
Integration tests for registry server registration validation against the upstream OAuth provider registry.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from auth_server.enforceai.db.data_layer import (
    EnforceAIDataLayer,
)
from registry.auth import dependencies as auth_dependencies
from registry.main import app


@pytest.mark.integration
class TestRegistryUpstreamOAuthProviderValidation:
    def setup_method(self) -> None:
        app.dependency_overrides[auth_dependencies.nginx_proxied_auth] = (
            lambda: {
                "username": "operator",
                "is_admin": True,
                "accessible_servers": ["all"],
                "ui_permissions": {"register_service": ["*"]},
            }
        )

    def teardown_method(self) -> None:
        app.dependency_overrides.pop(auth_dependencies.nginx_proxied_auth, None)

    def test_create_server_rejects_unknown_upstream_oauth_provider(
        self,
        test_client: TestClient,
        enforceai_sqlite_db_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        layer = EnforceAIDataLayer(db_path=enforceai_sqlite_db_path)
        layer.initialize()
        stores = layer.build_stores()
        stores.egress_allowlist_store.create_entry(
            kind="hostname",
            value="example.com",
            comment="test",
        )

        monkeypatch.setenv("ENFORCEAI_DB_PATH", str(enforceai_sqlite_db_path))

        response = test_client.post(
            "/api/servers",
            json={
                "name": "OAuth Server",
                "path": "oauth-server",
                "proxy_pass_url": "https://example.com/mcp",
                "description": "test",
                "tags": [],
                "upstream_auth": {
                    "mode": "gateway-managed",
                    "type": "provider-oauth",
                    "provider": "missing-provider",
                    "credential_binding": "user",
                    "injection": None,
                },
            },
        )

        assert response.status_code == 400
        assert "unknown upstream OAuth provider" in response.json()["detail"]

