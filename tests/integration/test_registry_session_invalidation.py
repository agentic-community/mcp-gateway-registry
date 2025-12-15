"""
Integration tests for server-side session invalidation.
"""

from __future__ import annotations

from pathlib import Path

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
class TestRegistrySessionInvalidation:
    def test_logout_revokes_server_side_session(
        self,
        test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        enforceai_db_path = tmp_path / "enforceai.db"
        monkeypatch.setattr(settings, "enforceai_db_path", enforceai_db_path)

        cookie_value = create_session_cookie("testuser")
        cookies = {settings.session_cookie_name: cookie_value}

        response = test_client.get(
            "/api/auth/me",
            cookies=cookies,
        )
        assert response.status_code == 200

        csrf_response = test_client.get(
            "/api/auth/csrf",
            cookies=cookies,
        )
        assert csrf_response.status_code == 200
        csrf_token = csrf_response.json()["csrf_token"]

        logout_response = test_client.post(
            "/api/auth/logout",
            cookies=cookies,
            headers={"X-CSRF-Token": csrf_token},
            follow_redirects=False,
        )
        assert logout_response.status_code == 303

        response = test_client.get(
            "/api/auth/me",
            cookies=cookies,
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "Session invalidated"
