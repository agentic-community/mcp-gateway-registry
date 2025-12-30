"""
Integration tests for registry server registration egress allowlist enforcement (Phase 2).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import (
    AsyncMock,
    patch,
)

import pytest
from fastapi.testclient import TestClient

from auth_server.enforceai.db.data_layer import (
    EnforceAIDataLayer,
)
from registry.auth import dependencies as auth_dependencies
from registry.main import app


@pytest.mark.integration
class TestRegistryEgressAllowlistEnforcement:
    def setup_method(self) -> None:
        app.dependency_overrides[auth_dependencies.nginx_proxied_auth] = (
            lambda: {
                "username": "admin-user",
                "is_admin": True,
                "accessible_servers": ["all"],
                "can_modify_servers": True,
            }
        )

    def teardown_method(self) -> None:
        app.dependency_overrides.pop(auth_dependencies.nginx_proxied_auth, None)

    def test_servers_register_rejects_when_allowlist_empty(
        self,
        test_client: TestClient,
        enforceai_sqlite_db_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        layer = EnforceAIDataLayer(db_path=enforceai_sqlite_db_path)
        layer.initialize()

        monkeypatch.setenv("ENFORCEAI_DB_PATH", str(enforceai_sqlite_db_path))

        response = test_client.post(
            "/api/servers/register",
            data={
                "name": "svc",
                "description": "svc",
                "path": "/svc",
                "proxy_pass_url": "https://example.com/mcp",
            },
        )
        assert response.status_code == 400
        assert "egress allowlist is empty" in response.json()["detail"]

    def test_servers_register_accepts_allowlisted_proxy_pass_url(
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

        def _close_coroutine(
            coro,
        ):
            coro.close()
            return None

        with patch("registry.api.server_external_routes.server_service") as mock_service, patch(
            "registry.api.server_external_routes.asyncio.create_task",
            side_effect=_close_coroutine,
        ), patch(
            "registry.health.service.health_service.perform_immediate_health_check",
            new=AsyncMock(),
        ), patch(
            "registry.search.service.faiss_service.save_data",
            new=AsyncMock(),
        ), patch(
            "registry.search.service.faiss_service.add_or_update_service",
            new=AsyncMock(),
        ), patch(
            "registry.core.nginx_service.nginx_service.generate_config_async",
            new=AsyncMock(),
        ), patch(
            "registry.health.service.health_service.broadcast_health_update",
            new=AsyncMock(),
        ), patch(
            "registry.utils.scopes_manager.update_server_scopes",
            new=AsyncMock(),
        ):
            mock_service.get_server_info.return_value = None
            mock_service.register_server.return_value = True
            mock_service.update_server.return_value = True
            mock_service.toggle_service.return_value = True
            mock_service.is_service_enabled.return_value = True
            mock_service.get_enabled_services.return_value = ["/svc"]

            response = test_client.post(
                "/api/servers/register",
                data={
                    "name": "svc",
                    "description": "svc",
                    "path": "/svc",
                    "proxy_pass_url": "https://example.com/mcp",
                },
            )

        assert response.status_code == 201
