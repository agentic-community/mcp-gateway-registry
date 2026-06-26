"""Authorization regression tests for the registry-wide authz hardening pass.

Covers the gaps closed across IAM/M2M read endpoints, virtual-server
sub-resource scoping, skill registration, ANS status, and health endpoints.
Each test asserts a non-admin / unauthorized caller is rejected on a surface
that previously leaked data or accepted an unauthorized action.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from registry.auth.dependencies import nginx_proxied_auth
from registry.main import app


def _non_admin():
    """A logged-in user with no admin rights and no UI permissions.

    Must take NO parameters: FastAPI introspects a dependency-override's
    signature, so any args (even **kwargs) would be turned into request
    parameters and cause spurious 422s.
    """
    return {
        "username": "non-admin",
        "groups": ["engineering"],
        "scopes": [],
        "is_admin": False,
        "ui_permissions": {},
        "accessible_servers": [],
        "accessible_agents": [],
        "accessible_services": [],
    }


def _override_auth():
    app.dependency_overrides[nginx_proxied_auth] = _non_admin


def _clear():
    app.dependency_overrides.clear()


@pytest.mark.unit
class TestIamM2MReadsAdminOnly:
    """IAM/M2M list+get endpoints must reject non-admins (IAM metadata leak)."""

    @pytest.mark.parametrize(
        "path",
        [
            "/api/iam/auth0/m2m/clients",
            "/api/iam/auth0/m2m/clients/abc/groups",
            "/api/iam/okta/m2m/clients",
            "/api/iam/okta/m2m/clients/abc/groups",
            "/api/iam/m2m-clients",
            "/api/iam/m2m-clients/abc",
            "/api/iam/user-groups",
            "/api/iam/user-groups/someuser",
        ],
    )
    def test_read_endpoint_forbidden_for_non_admin(self, path) -> None:
        _override_auth()
        try:
            client = TestClient(app)
            response = client.get(path)
            assert (
                response.status_code == status.HTTP_403_FORBIDDEN
            ), f"GET {path} did not return 403 (got {response.status_code})"
        finally:
            _clear()


@pytest.mark.unit
class TestVirtualServerScoping:
    """Virtual-server sub-resource reads/writes respect list_virtual_server."""

    def _mock_service(self):
        # Service returns data, but the route-level access check should 404
        # before the user (no list_virtual_server perm) ever sees it.
        svc = AsyncMock()
        svc.resolve_tools = AsyncMock(return_value=[])
        svc.get_virtual_server_rating = AsyncMock(return_value={"num_stars": 0})
        svc.get_virtual_server = AsyncMock(return_value={"path": "/virtual/secret"})
        svc.rate_virtual_server = AsyncMock(return_value={"num_stars": 5})
        return svc

    @pytest.mark.parametrize(
        "method,path,body",
        [
            ("get", "/api/virtual-servers/virtual/secret/tools", None),
            ("get", "/api/virtual-servers/virtual/secret/rating", None),
            ("get", "/api/virtual-servers/virtual/secret", None),
            ("post", "/api/virtual-servers/virtual/secret/rate", {"rating": 5}),
        ],
    )
    def test_unscoped_access_returns_404(self, method, path, body) -> None:
        _override_auth()
        try:
            with patch(
                "registry.api.virtual_server_routes.get_virtual_server_service",
                return_value=self._mock_service(),
            ):
                client = TestClient(app)
                req = getattr(client, method)
                response = req(path, json=body) if body else req(path)
            assert response.status_code == status.HTTP_404_NOT_FOUND, (
                f"{method.upper()} {path} did not 404 for unscoped user "
                f"(got {response.status_code})"
            )
        finally:
            _clear()


@pytest.mark.unit
class TestSkillRegisterRequiresPublish:
    """register_skill / parse-skill-md require the publish_skill permission."""

    def test_register_skill_forbidden_without_publish(self) -> None:
        _override_auth()
        try:
            client = TestClient(app)
            response = client.post(
                "/api/skills",
                json={
                    "name": "x",
                    "description": "y",
                    "skill_md_url": "https://example.com/SKILL.md",
                },
            )
            assert response.status_code == status.HTTP_403_FORBIDDEN
        finally:
            _clear()

    def test_parse_skill_md_forbidden_without_publish(self) -> None:
        _override_auth()
        try:
            client = TestClient(app)
            response = client.post("/api/skills/parse-skill-md?url=https://example.com/SKILL.md")
            assert response.status_code == status.HTTP_403_FORBIDDEN
        finally:
            _clear()


@pytest.mark.unit
class TestHealthHttpRequiresAuth:
    """The HTTP health/stats endpoints are no longer anonymous."""

    def test_health_status_http_requires_auth(self) -> None:
        # No auth override: nginx_proxied_auth runs for real and rejects the
        # unauthenticated request (no session cookie / token).
        client = TestClient(app)
        response = client.get("/api/health/ws/health_status")
        assert response.status_code in (
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_403_FORBIDDEN,
        )

    def test_ws_stats_forbidden_for_non_admin(self) -> None:
        _override_auth()
        try:
            client = TestClient(app)
            response = client.get("/api/health/ws/stats")
            assert response.status_code == status.HTTP_403_FORBIDDEN
        finally:
            _clear()
