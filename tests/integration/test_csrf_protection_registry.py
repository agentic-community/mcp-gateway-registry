"""
Integration tests for registry CSRF protection when authenticated via session cookies.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from registry.auth.dependencies import (
    create_session_cookie,
)
from registry.core.config import (
    settings,
)
from tests.fixtures.factories import (
    ServerInfoFactory,
)


@pytest.mark.integration
@pytest.mark.auth
class TestRegistryCsrfProtection:
    def test_missing_csrf_header_rejected_for_cookie_request(
        self,
        test_client: TestClient,
    ) -> None:
        cookie_value = create_session_cookie("testuser")
        server_data = ServerInfoFactory()

        with patch("registry.api.server_routes.server_service") as mock_service:
            response = test_client.post(
                "/api/register",
                cookies={settings.session_cookie_name: cookie_value},
                data={
                    "name": server_data["server_name"],
                    "description": server_data["description"],
                    "path": server_data["path"],
                    "proxy_pass_url": server_data["proxy_pass_url"],
                },
            )

        assert response.status_code == 403
        assert response.json()["detail"] == "Missing X-CSRF-Token"
        mock_service.register_server.assert_not_called()

    def test_valid_csrf_header_allows_request(
        self,
        test_client: TestClient,
    ) -> None:
        cookie_value = create_session_cookie("testuser")

        csrf_response = test_client.get(
            "/api/auth/csrf",
            cookies={settings.session_cookie_name: cookie_value},
        )
        assert csrf_response.status_code == 200
        csrf_token = csrf_response.json()["csrf_token"]
        assert csrf_token

        server_data = ServerInfoFactory()

        with (
            patch("registry.api.server_routes.server_service") as mock_service,
            patch("registry.search.service.faiss_service") as mock_faiss,
            patch("registry.core.nginx_service.nginx_service") as mock_nginx,
            patch("registry.health.service.health_service") as mock_health,
        ):
            mock_service.register_server.return_value = True
            mock_faiss.add_or_update_service = AsyncMock()
            mock_nginx.generate_config_async = AsyncMock()
            mock_health.broadcast_health_update = AsyncMock()
            mock_service.get_enabled_services.return_value = []
            mock_service.get_server_info.return_value = None

            response = test_client.post(
                "/api/register",
                cookies={settings.session_cookie_name: cookie_value},
                headers={"X-CSRF-Token": csrf_token},
                data={
                    "name": server_data["server_name"],
                    "description": server_data["description"],
                    "path": server_data["path"],
                    "proxy_pass_url": server_data["proxy_pass_url"],
                    "tags": ",".join(server_data["tags"]),
                    "num_tools": server_data["num_tools"],
                    "num_stars": server_data["num_stars"],
                    "is_python": server_data["is_python"],
                    "license": server_data["license"],
                },
            )

        assert response.status_code == 201

