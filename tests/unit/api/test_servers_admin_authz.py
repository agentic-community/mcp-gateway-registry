"""Authorization regression tests for mutating /api/servers/* endpoints.

These endpoints authenticate via ``nginx_proxied_auth`` but historically did
not authorize: any authenticated low-privilege user could call
``POST /api/servers/groups/import`` to map their IdP group to the
``mcp-registry-admin`` scope and self-escalate to admin (coordinated
disclosure, 2026-06-24).

Two gate styles are asserted, matching the sibling pattern in this router:

- Scope/group mutation (import, groups/add|remove|create|delete) is admin-only.
- Server-op mutation (toggle, auth-credential, version delete/default) requires
  the fine-grained ``toggle_service`` / ``modify_service`` UI permission.

A non-admin user with NO server UI permissions must be rejected with 403 on
every endpoint below before any state mutation occurs.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from registry.auth.dependencies import nginx_proxied_auth
from registry.main import app


def _mock_non_admin_auth(
    request=None,
    session=None,
    x_user=None,
    x_username=None,
    x_scopes=None,
    x_auth_method=None,
    x_client_id=None,
):
    """Mock nginx_proxied_auth returning an authenticated NON-admin user.

    The user holds no server UI permissions, so both the admin-only and the
    fine-grained ``modify_service``/``toggle_service`` gates must reject them.
    """
    return {
        "username": "non-admin-user",
        "groups": ["engineering"],
        "scopes": ["mcp-servers-restricted/read"],
        "auth_method": "oauth2",
        "provider": "keycloak",
        "accessible_servers": [],
        "accessible_services": ["all"],
        "can_modify_servers": False,
        "ui_permissions": {},
        "is_admin": False,
    }


@pytest.mark.unit
class TestServersAdminAuthz:
    """Non-admin authenticated users must be rejected with 403 on mutations."""

    def test_groups_import_non_admin_forbidden(self) -> None:
        """The reported privesc vector: POST /api/servers/groups/import is admin-only."""
        app.dependency_overrides[nginx_proxied_auth] = _mock_non_admin_auth
        try:
            client = TestClient(app)
            response = client.post(
                "/api/servers/groups/import",
                json={
                    "scope_name": "engineering",
                    "group_mappings": ["engineering", "mcp-registry-admin"],
                    "ui_permissions": {"list_service": ["all"]},
                },
            )
            assert response.status_code == status.HTTP_403_FORBIDDEN
        finally:
            app.dependency_overrides.clear()

    def test_all_mutating_routes_non_admin_forbidden(self) -> None:
        """Every mutating /api/servers/* route returns 403 for a non-admin user.

        Guards against a regression where a mutating handler is added or edited
        without the authorization check its siblings enforce. Each route must
        fail closed with 403 before performing any state mutation.
        """
        # (method, path, json_body, data_body)
        mutating_routes = [
            # Scope/group mutation -> admin-only.
            (
                "post",
                "/api/servers/groups/import",
                {"scope_name": "x", "group_mappings": ["mcp-registry-admin"]},
                None,
            ),
            ("post", "/api/servers/groups/add", None, {"server_name": "x", "group_names": "y"}),
            ("post", "/api/servers/groups/remove", None, {"server_name": "x", "group_names": "y"}),
            ("post", "/api/servers/groups/create", None, {"group_name": "x"}),
            ("post", "/api/servers/groups/delete", None, {"group_name": "x"}),
            # Server-op mutation -> fine-grained UI permission.
            ("post", "/api/servers/toggle", None, {"path": "/x", "new_state": "true"}),
            ("delete", "/api/servers/x/versions/v1", None, None),
            ("put", "/api/servers/x/versions/default", {"version": "v1"}, None),
        ]
        # Server-op endpoints look the server up first (and 404 if missing),
        # then run the permission check -- mirroring the sibling routes. Mock the
        # lookup so the request reaches the authorization gate, not the 404.
        server_info = {"server_name": "x", "path": "/x"}
        app.dependency_overrides[nginx_proxied_auth] = _mock_non_admin_auth
        try:
            client = TestClient(app)
            with patch(
                "registry.api.server_routes.server_service.get_server_info",
                new=AsyncMock(return_value=server_info),
            ):
                for method, path, json_body, data_body in mutating_routes:
                    request = getattr(client, method)
                    if json_body is not None:
                        response = request(path, json=json_body)
                    elif data_body is not None:
                        response = request(path, data=data_body)
                    else:
                        response = request(path)
                    assert response.status_code == status.HTTP_403_FORBIDDEN, (
                        f"{method.upper()} {path} did not return 403 for non-admin "
                        f"(got {response.status_code})"
                    )
        finally:
            app.dependency_overrides.clear()

    def test_auth_credential_non_admin_forbidden(self) -> None:
        """PATCH /api/servers/{path}/auth-credential requires modify_service."""
        server_info = {"server_name": "some-server", "path": "/some-server"}
        app.dependency_overrides[nginx_proxied_auth] = _mock_non_admin_auth
        try:
            client = TestClient(app)
            with patch(
                "registry.api.server_routes.server_service.get_server_info",
                new=AsyncMock(return_value=server_info),
            ):
                response = client.patch(
                    "/api/servers/some-server/auth-credential",
                    json={"auth_scheme": "bearer", "auth_credential": "secret"},
                )
            assert response.status_code == status.HTTP_403_FORBIDDEN
        finally:
            app.dependency_overrides.clear()
