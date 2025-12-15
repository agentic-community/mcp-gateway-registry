"""
Integration tests for registry `/api/internal/*` endpoints authentication.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from registry.auth.dependencies import (
    create_session_cookie,
)
from registry.core.config import (
    settings,
)


@pytest.mark.integration
@pytest.mark.auth
class TestRegistryInternalEndpointsAuth:
    def test_internal_list_accepts_admin_session_cookie(
        self,
        test_client: TestClient,
    ) -> None:
        cookie_value = create_session_cookie("testuser")
        response = test_client.get(
            "/api/internal/list",
            cookies={settings.session_cookie_name: cookie_value},
        )
        assert response.status_code == 200
        payload = response.json()
        assert "services" in payload
        assert "total_count" in payload
        assert payload["total_count"] == len(payload["services"])

    def test_internal_list_denies_non_admin_session_cookie(
        self,
        test_client: TestClient,
    ) -> None:
        cookie_value = create_session_cookie(
            "testuser",
            auth_method="oauth2",
            provider="keycloak",
        )
        response = test_client.get(
            "/api/internal/list",
            cookies={settings.session_cookie_name: cookie_value},
        )
        assert response.status_code == 403
        assert response.json()["detail"] == "Admin required"

