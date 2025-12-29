"""
Integration tests for `/api/servers/toggle` contract compatibility.
"""

from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)
from unittest.mock import (
    AsyncMock,
    patch,
)

import pytest
from fastapi.testclient import TestClient

from registry.auth import dependencies as auth_dependencies
from registry.main import app


@pytest.mark.integration
class TestRegistryServersToggleContract:
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

    def test_servers_toggle_accepts_service_path_and_flips_state(
        self,
        test_client: TestClient,
    ) -> None:
        server_info = {
            "server_name": "svc",
            "num_tools": 1,
        }

        with patch("registry.api.server_external_routes.server_service") as mock_service, patch(
            "registry.search.service.faiss_service"
        ) as mock_faiss, patch(
            "registry.core.nginx_service.nginx_service"
        ) as mock_nginx, patch(
            "registry.health.service.health_service"
        ) as mock_health:
            mock_service.get_server_info.return_value = server_info
            mock_service.is_service_enabled.return_value = False
            mock_service.toggle_service.return_value = True
            mock_service.get_enabled_services.return_value = []

            mock_faiss.add_or_update_service = AsyncMock()
            mock_nginx.generate_config_async = AsyncMock()
            mock_health.broadcast_health_update = AsyncMock()
            mock_health.perform_immediate_health_check = AsyncMock(
                return_value=("healthy", datetime.now(timezone.utc)),
            )

            response = test_client.post(
                "/api/servers/toggle",
                data={
                    "service_path": "/svc",
                },
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["path"] == "/svc"
        assert payload["service_path"] == "/svc"
        assert payload["is_enabled"] is True
        assert payload["new_enabled_state"] is True

        mock_service.toggle_service.assert_called_once_with("/svc", True)

    def test_servers_toggle_accepts_path_and_new_state(
        self,
        test_client: TestClient,
    ) -> None:
        server_info = {
            "server_name": "svc",
            "num_tools": 1,
        }

        with patch("registry.api.server_external_routes.server_service") as mock_service, patch(
            "registry.search.service.faiss_service"
        ) as mock_faiss, patch(
            "registry.core.nginx_service.nginx_service"
        ) as mock_nginx, patch(
            "registry.health.service.health_service"
        ) as mock_health:
            mock_service.get_server_info.return_value = server_info
            mock_service.toggle_service.return_value = True
            mock_service.get_enabled_services.return_value = []

            mock_faiss.add_or_update_service = AsyncMock()
            mock_nginx.generate_config_async = AsyncMock()
            mock_health.broadcast_health_update = AsyncMock()
            mock_health.perform_immediate_health_check = AsyncMock(
                return_value=("healthy", datetime.now(timezone.utc)),
            )

            response = test_client.post(
                "/api/servers/toggle",
                data={
                    "path": "/svc",
                    "new_state": "false",
                },
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["path"] == "/svc"
        assert payload["service_path"] == "/svc"
        assert payload["is_enabled"] is False
        assert payload["new_enabled_state"] is False

        mock_service.toggle_service.assert_called_once_with("/svc", False)
