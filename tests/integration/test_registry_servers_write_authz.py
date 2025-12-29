"""
Integration tests for `/api/servers/*` write authorization.
"""

from __future__ import annotations

from unittest.mock import (
    patch,
)

import pytest
from fastapi.testclient import TestClient

from registry.auth import dependencies as auth_dependencies
from registry.main import app


@pytest.mark.integration
class TestRegistryServersWriteAuthz:
    def teardown_method(self) -> None:
        app.dependency_overrides.pop(auth_dependencies.nginx_proxied_auth, None)

    def test_servers_register_denies_without_can_modify_servers(
        self,
        test_client: TestClient,
    ) -> None:
        app.dependency_overrides[auth_dependencies.nginx_proxied_auth] = (
            lambda: {
                "username": "readonly-user",
                "is_admin": False,
                "accessible_servers": ["all"],
                "can_modify_servers": False,
            }
        )

        with patch("registry.api.server_external_routes.server_service"):
            response = test_client.post(
                "/api/servers/register",
                data={
                    "name": "svc",
                    "description": "svc",
                    "path": "/svc",
                    "proxy_pass_url": "http://example.invalid",
                },
            )

        assert response.status_code == 403
        assert response.json()["detail"] == "Insufficient privileges to modify servers"

    def test_servers_toggle_denies_without_can_modify_servers(
        self,
        test_client: TestClient,
    ) -> None:
        app.dependency_overrides[auth_dependencies.nginx_proxied_auth] = (
            lambda: {
                "username": "readonly-user",
                "is_admin": False,
                "accessible_servers": ["all"],
                "can_modify_servers": False,
            }
        )

        with patch("registry.api.server_external_routes.server_service"):
            response = test_client.post(
                "/api/servers/toggle",
                data={
                    "path": "/svc",
                    "new_state": "true",
                },
            )

        assert response.status_code == 403
        assert response.json()["detail"] == "Insufficient privileges to modify servers"

    def test_servers_remove_denies_without_can_modify_servers(
        self,
        test_client: TestClient,
    ) -> None:
        app.dependency_overrides[auth_dependencies.nginx_proxied_auth] = (
            lambda: {
                "username": "readonly-user",
                "is_admin": False,
                "accessible_servers": ["all"],
                "can_modify_servers": False,
            }
        )

        with patch("registry.api.server_external_routes.server_service"):
            response = test_client.post(
                "/api/servers/remove",
                data={
                    "path": "/svc",
                },
            )

        assert response.status_code == 403
        assert response.json()["detail"] == "Insufficient privileges to modify servers"
