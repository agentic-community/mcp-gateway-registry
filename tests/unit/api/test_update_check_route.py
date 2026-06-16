"""Tests for the admin-gated /api/system/update-check route (issue #1218)."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from registry.auth.dependencies import nginx_proxied_auth
from registry.main import app


def _mock_admin_auth():
    return {
        "username": "admin-user",
        "groups": ["mcp-registry-admin"],
        "scopes": [],
        "auth_method": "oauth2",
        "provider": "keycloak",
        "accessible_servers": ["all"],
        "accessible_services": ["all"],
        "can_modify_servers": True,
        "is_admin": True,
        "ui_permissions": {},
    }


def _mock_non_admin_auth():
    return {
        "username": "regular-user",
        "groups": ["engineering"],
        "scopes": [],
        "auth_method": "oauth2",
        "provider": "keycloak",
        "accessible_servers": [],
        "accessible_services": ["all"],
        "can_modify_servers": False,
        "is_admin": False,
        "ui_permissions": {},
    }


@pytest.mark.unit
class TestUpdateCheckRoute:
    def test_admin_receives_payload(self) -> None:
        app.dependency_overrides[nginx_proxied_auth] = _mock_admin_auth
        try:
            client = TestClient(app)
            response = client.get("/api/system/update-check")
            assert response.status_code == status.HTTP_200_OK
            body = response.json()
            assert "current" in body
            assert "latest" in body
            assert "update_available" in body
            assert "release_notes_url" in body
            assert "checked_at" in body
            assert "check_enabled" in body
        finally:
            app.dependency_overrides.clear()

    def test_non_admin_forbidden(self) -> None:
        app.dependency_overrides[nginx_proxied_auth] = _mock_non_admin_auth
        try:
            client = TestClient(app)
            response = client.get("/api/system/update-check")
            assert response.status_code == status.HTTP_403_FORBIDDEN
        finally:
            app.dependency_overrides.clear()

    def test_unauthenticated_rejected(self) -> None:
        # No override → real nginx_proxied_auth runs and rejects a request with
        # neither a session cookie nor nginx-forwarded identity headers.
        client = TestClient(app)
        response = client.get("/api/system/update-check")
        assert response.status_code in (
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_403_FORBIDDEN,
        )
