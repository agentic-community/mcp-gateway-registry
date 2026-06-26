"""
Unit tests for admin authorization on federation config/sync routes.

These routes mutate federation configuration or trigger syncs and must be
restricted to admin users, matching the sibling pattern in server_routes.py
(e.g. ``if not user_context.get("is_admin"): raise HTTPException(403)``).
"""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from registry.auth.dependencies import nginx_proxied_auth
from registry.main import app


def _mock_non_admin_auth(
    request=None, session=None, x_user=None, x_username=None, x_scopes=None, x_auth_method=None
):
    """Mock nginx_proxied_auth returning an authenticated NON-admin user."""
    return {
        "username": "non-admin-user",
        "groups": ["engineering"],
        "scopes": ["mcp-servers-restricted/read"],
        "auth_method": "oauth2",
        "provider": "keycloak",
        "accessible_servers": [],
        "accessible_services": ["all"],
        "can_modify_servers": False,
        "is_admin": False,
    }


@pytest.mark.unit
class TestFederationConfigAdminAuthz:
    """Non-admin authenticated users must be rejected with 403 on mutating routes."""

    def test_save_federation_config_non_admin_forbidden(self) -> None:
        """POST /api/federation/config returns 403 for non-admin users."""
        app.dependency_overrides[nginx_proxied_auth] = _mock_non_admin_auth
        try:
            client = TestClient(app)
            response = client.post(
                "/api/federation/config",
                json={
                    "anthropic": {"enabled": False, "endpoint": "", "servers": []},
                    "asor": {"enabled": False, "endpoint": "", "agents": []},
                },
            )
            assert response.status_code == status.HTTP_403_FORBIDDEN
        finally:
            app.dependency_overrides.clear()

    def test_sync_federation_non_admin_forbidden(self) -> None:
        """POST /api/federation/sync returns 403 for non-admin users."""
        app.dependency_overrides[nginx_proxied_auth] = _mock_non_admin_auth
        try:
            client = TestClient(app)
            response = client.post("/api/federation/sync")
            assert response.status_code == status.HTTP_403_FORBIDDEN
        finally:
            app.dependency_overrides.clear()

    def test_all_mutating_routes_non_admin_forbidden(self) -> None:
        """Every mutating federation route returns 403 for non-admin users.

        Guards against a regression where one mutating handler is added or
        edited without the admin check that its siblings enforce.
        """
        config_body = {
            "anthropic": {"enabled": False, "endpoint": "", "servers": []},
            "asor": {"enabled": False, "endpoint": "", "agents": []},
        }
        # (method, path, json_body)
        mutating_routes = [
            ("post", "/api/federation/config", config_body),
            ("put", "/api/federation/config/default", config_body),
            ("delete", "/api/federation/config/default", None),
            ("post", "/api/federation/config/default/anthropic/servers?server_name=x", None),
            ("delete", "/api/federation/config/default/anthropic/servers/x", None),
            ("post", "/api/federation/config/default/asor/agents?agent_id=x", None),
            ("delete", "/api/federation/config/default/asor/agents/x", None),
            (
                "post",
                "/api/federation/config/default/aws_registry/registries",
                {"registry_id": "arn:x", "aws_region": "us-east-1"},
            ),
            ("delete", "/api/federation/config/default/aws_registry/registries/arn:x", None),
            ("post", "/api/federation/sync", None),
        ]
        app.dependency_overrides[nginx_proxied_auth] = _mock_non_admin_auth
        try:
            client = TestClient(app)
            for method, path, body in mutating_routes:
                request = getattr(client, method)
                response = request(path, json=body) if body is not None else request(path)
                assert response.status_code == status.HTTP_403_FORBIDDEN, (
                    f"{method.upper()} {path} did not return 403 for non-admin "
                    f"(got {response.status_code})"
                )
        finally:
            app.dependency_overrides.clear()
