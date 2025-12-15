"""
Integration tests for the registry `/api/auth/me` session contract.
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
class TestAuthMeSessionContract:
    def test_me_includes_user_id_and_session_id(
        self,
        test_client: TestClient,
    ) -> None:
        cookie_value = create_session_cookie("testuser")
        response = test_client.get(
            "/api/auth/me",
            cookies={settings.session_cookie_name: cookie_value},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "testuser"
        assert data["user_id"] == "local|testuser"
        assert data["auth_method"] == "password"
        assert data["legacy_auth_method"] == "traditional"
        assert data["session_id"]

