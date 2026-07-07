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


def _non_admin_with_execute():
    """A non-admin who holds a per-server execute scope.

    This is the privilege-escalation shape for virtual-server CRUD: a user with
    an ``/execute`` scope must NOT be treated as an administrator.
    """
    return {
        "username": "execute-user",
        "groups": ["engineering"],
        "scopes": ["some-server/execute"],
        "is_admin": False,
        "can_modify_servers": True,  # what the old, broken check accepted
        "ui_permissions": {"publish_skill": ["all"]},
        "accessible_servers": [],
        "accessible_agents": [],
        "accessible_services": [],
    }


def _admin():
    """A genuine administrator."""
    return {
        "username": "admin",
        "groups": ["mcp-registry-admin"],
        "scopes": [],
        "is_admin": True,
        "ui_permissions": {"publish_skill": ["all"]},
        "accessible_servers": ["all"],
        "accessible_agents": ["all"],
        "accessible_services": ["all"],
    }


def _override_auth(factory=_non_admin):
    app.dependency_overrides[nginx_proxied_auth] = factory


def _clear():
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _uninstrument_otel():
    """Remove OpenTelemetry FastAPI auto-instrumentation for these tests.

    ``registry.main`` auto-instruments the app at import time. The installed
    ``opentelemetry-instrumentation-fastapi`` raises ``AttributeError`` on
    ``_IncludedRouter`` when matching a route added via ``include_router`` for a
    *successful* request, which is unrelated to the authorization behavior under
    test. Uninstrumenting keeps these route-level authz assertions deterministic
    regardless of the installed OTel version; the app is re-instrumented after
    the test so global state is restored.
    """
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except ImportError:
        yield
        return

    was_instrumented = getattr(app, "_is_instrumented_by_opentelemetry", False)
    if was_instrumented:
        FastAPIInstrumentor.uninstrument_app(app)
    try:
        yield
    finally:
        if was_instrumented and not getattr(app, "_is_instrumented_by_opentelemetry", False):
            FastAPIInstrumentor.instrument_app(app)


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
            assert response.status_code == status.HTTP_403_FORBIDDEN, (
                f"GET {path} did not return 403 (got {response.status_code})"
            )
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


@pytest.mark.unit
class TestVirtualServerCrudAdminOnly:
    """Virtual-server CRUD must require an actual admin, not an execute scope."""

    _CREATE_BODY = {"server_name": "aggregator", "description": "d"}
    _UPDATE_BODY = {"server_name": "aggregator", "description": "d"}

    def _mock_service(self):
        svc = AsyncMock()
        svc.create_virtual_server = AsyncMock(return_value={"path": "/virtual/x"})
        svc.update_virtual_server = AsyncMock(return_value={"path": "/virtual/x"})
        svc.delete_virtual_server = AsyncMock(return_value=True)
        return svc

    @pytest.mark.parametrize(
        "method,path,body",
        [
            ("post", "/api/virtual-servers", _CREATE_BODY),
            ("put", "/api/virtual-servers/virtual/x", _UPDATE_BODY),
            ("delete", "/api/virtual-servers/virtual/x", None),
        ],
    )
    def test_execute_scope_user_forbidden(self, method, path, body) -> None:
        """A non-admin with only an execute scope is denied vserver CRUD."""
        _override_auth(_non_admin_with_execute)
        try:
            with patch(
                "registry.api.virtual_server_routes.get_virtual_server_service",
                return_value=self._mock_service(),
            ):
                client = TestClient(app)
                req = getattr(client, method)
                response = req(path, json=body) if body else req(path)
            assert response.status_code == status.HTTP_403_FORBIDDEN, (
                f"{method.upper()} {path} should be admin-only (got {response.status_code})"
            )
        finally:
            _clear()

    def test_admin_allowed_to_create(self) -> None:
        """A genuine admin can create a virtual server (not 403)."""
        from registry.schemas.virtual_server_models import VirtualServerConfig

        _override_auth(_admin)
        try:
            svc = self._mock_service()
            svc.create_virtual_server = AsyncMock(
                return_value=VirtualServerConfig(
                    path="/virtual/aggregator",
                    server_name="aggregator",
                    description="d",
                )
            )
            with patch(
                "registry.api.virtual_server_routes.get_virtual_server_service",
                return_value=svc,
            ):
                client = TestClient(app)
                response = client.post("/api/virtual-servers", json=self._CREATE_BODY)
            assert response.status_code != status.HTTP_403_FORBIDDEN
        finally:
            _clear()


@pytest.mark.unit
class TestSkillGlobalCredentialsAdminOnly:
    """The global_credentials auth scheme is restricted to admins."""

    def test_parse_skill_md_global_credentials_forbidden_for_non_admin(self) -> None:
        """A publish-capable non-admin cannot use global_credentials on parse."""
        _override_auth(_non_admin_with_execute)  # has publish_skill but not admin
        try:
            client = TestClient(app)
            response = client.post(
                "/api/skills/parse-skill-md"
                "?url=https://example.com/SKILL.md&auth_scheme=global_credentials"
            )
            assert response.status_code == status.HTTP_403_FORBIDDEN
        finally:
            _clear()

    def test_register_skill_global_credentials_forbidden_for_non_admin(self) -> None:
        """A publish-capable non-admin cannot register a global_credentials skill."""
        _override_auth(_non_admin_with_execute)
        try:
            client = TestClient(app)
            response = client.post(
                "/api/skills",
                json={
                    "name": "x",
                    "description": "y",
                    "skill_md_url": "https://example.com/SKILL.md",
                    "auth_scheme": "global_credentials",
                },
            )
            assert response.status_code == status.HTTP_403_FORBIDDEN
        finally:
            _clear()


@pytest.mark.unit
class TestParseSkillMdCredentialViaHeader:
    """The parse credential is accepted only via header, never a query parameter.

    Query parameters land in access/proxy logs, browser history, and the audit
    log's captured query params, so a plaintext credential must not travel in
    the URL.
    """

    _SECRET = "super-secret-bearer-token-value"

    def _mock_service(self):
        svc = AsyncMock()
        svc.parse_skill_md = AsyncMock(
            return_value={
                "name": "Demo",
                "name_slug": "demo",
                "description": "d",
                "version": "1.0",
                "tags": [],
            }
        )
        return svc

    def test_header_credential_reaches_service(self) -> None:
        """A credential supplied via X-Auth-Credential is forwarded to parsing."""
        _override_auth(_non_admin_with_execute)  # has publish_skill
        try:
            svc = self._mock_service()
            with patch(
                "registry.api.skill_routes.get_skill_service",
                return_value=svc,
            ):
                client = TestClient(app)
                response = client.post(
                    "/api/skills/parse-skill-md"
                    "?url=https://example.com/SKILL.md&auth_scheme=bearer",
                    headers={"X-Auth-Credential": self._SECRET},
                )
            assert response.status_code == status.HTTP_200_OK, response.text
            svc.parse_skill_md.assert_awaited_once()
            _, kwargs = svc.parse_skill_md.await_args
            assert kwargs["auth_credential"] == self._SECRET
            assert kwargs["auth_scheme"] == "bearer"
        finally:
            _clear()

    def test_credential_in_query_param_is_ignored(self) -> None:
        """A credential passed as a query parameter is NOT consumed as the credential.

        The endpoint only binds ``auth_credential`` from the header, so a
        ``?auth_credential=...`` in the URL is treated as an unrelated query
        param and never reaches the parsing service as the credential. This
        proves the credential cannot be smuggled through the (logged) query
        string.
        """
        _override_auth(_non_admin_with_execute)
        try:
            svc = self._mock_service()
            with patch(
                "registry.api.skill_routes.get_skill_service",
                return_value=svc,
            ):
                client = TestClient(app)
                response = client.post(
                    "/api/skills/parse-skill-md"
                    "?url=https://example.com/SKILL.md&auth_scheme=bearer"
                    f"&auth_credential={self._SECRET}",
                )
            assert response.status_code == status.HTTP_200_OK, response.text
            _, kwargs = svc.parse_skill_md.await_args
            assert kwargs["auth_credential"] is None
        finally:
            _clear()


@pytest.mark.unit
class TestAnsServerLinkOwnership:
    """ANS server link/unlink must be owner-or-admin only."""

    def _mock_repo(self, owner: str):
        repo = AsyncMock()
        repo.get = AsyncMock(return_value={"path": "/servers/victim", "registered_by": owner})
        return repo

    def _enable_ans(self):
        # Ensure the ANS feature is enabled so the endpoints don't 404 early.
        return patch("registry.api.ans_routes._check_ans_enabled", lambda: None)

    def test_link_forbidden_for_non_owner(self) -> None:
        _override_auth()  # non-admin "non-admin"
        try:
            with (
                self._enable_ans(),
                patch(
                    "registry.repositories.factory.get_server_repository",
                    return_value=self._mock_repo(owner="someone-else"),
                ),
                patch("registry.api.ans_routes.verify_csrf_token_flexible", AsyncMock()),
            ):
                client = TestClient(app)
                response = client.post(
                    "/api/servers/victim/ans/link",
                    json={"ans_agent_id": "ans://v1.0.acme-agent"},
                )
            assert response.status_code == status.HTTP_403_FORBIDDEN
        finally:
            _clear()

    def test_unlink_forbidden_for_non_owner(self) -> None:
        _override_auth()
        try:
            with (
                self._enable_ans(),
                patch(
                    "registry.repositories.factory.get_server_repository",
                    return_value=self._mock_repo(owner="someone-else"),
                ),
                patch("registry.api.ans_routes.verify_csrf_token_flexible", AsyncMock()),
            ):
                client = TestClient(app)
                response = client.request("DELETE", "/api/servers/victim/ans/link")
            assert response.status_code == status.HTTP_403_FORBIDDEN
        finally:
            _clear()

    def test_unlink_allowed_for_owner(self) -> None:
        """The server owner may unlink ANS (not 403)."""
        _override_auth()  # username "non-admin"
        try:
            with (
                self._enable_ans(),
                patch(
                    "registry.repositories.factory.get_server_repository",
                    return_value=self._mock_repo(owner="non-admin"),
                ),
                patch(
                    "registry.services.ans_service.get_server_repository",
                    return_value=self._mock_repo(owner="non-admin"),
                ),
                patch("registry.api.ans_routes.verify_csrf_token_flexible", AsyncMock()),
            ):
                client = TestClient(app)
                response = client.request("DELETE", "/api/servers/victim/ans/link")
            assert response.status_code != status.HTTP_403_FORBIDDEN
        finally:
            _clear()

    def test_unlink_allowed_for_admin_non_owner(self) -> None:
        """An admin may unlink ANS on a server they don't own (not 403)."""
        _override_auth(_admin)  # is_admin True, username "admin"
        try:
            with (
                self._enable_ans(),
                patch(
                    "registry.repositories.factory.get_server_repository",
                    return_value=self._mock_repo(owner="someone-else"),
                ),
                patch(
                    "registry.services.ans_service.get_server_repository",
                    return_value=self._mock_repo(owner="someone-else"),
                ),
                patch("registry.api.ans_routes.verify_csrf_token_flexible", AsyncMock()),
            ):
                client = TestClient(app)
                response = client.request("DELETE", "/api/servers/victim/ans/link")
            assert response.status_code != status.HTTP_403_FORBIDDEN
        finally:
            _clear()


@pytest.mark.unit
class TestRemoveServiceDeletePermissionKey:
    """POST /servers/remove must key delete_service on the stored server_name.

    The fine-grained delete_service permission check must use the stored
    ``server_info["server_name"]`` as the trust key, matching every other
    mutation handler, not the raw URL path token. A grant for the path token
    that differs from the server_name must NOT authorize deletion, and a grant
    for the actual server_name must.
    """

    def _server_info(self):
        """Server whose stored name intentionally differs from the URL path token."""
        return {
            "path": "/victim",
            "server_name": "victim-server",
            "sync_metadata": {},
        }

    def _mock_service(self):
        service = AsyncMock()
        service.get_server_info = AsyncMock(return_value=self._server_info())
        service.remove_server = AsyncMock(return_value=True)
        return service

    def _non_admin_with_delete(self, allowed: list[str]):
        def factory():
            return {
                "username": "deleter",
                "groups": ["engineering"],
                "scopes": [],
                "is_admin": False,
                "ui_permissions": {"delete_service": allowed},
                "accessible_servers": [],
                "accessible_agents": [],
                "accessible_services": [],
            }

        return factory

    def test_delete_forbidden_when_grant_matches_path_token_not_server_name(self) -> None:
        """A delete_service grant for the path token must NOT authorize deletion."""
        # Grant keyed on the URL path token "victim" (the old, broken trust key).
        _override_auth(self._non_admin_with_delete(["victim"]))
        try:
            with patch(
                "registry.api.server_routes.server_service",
                self._mock_service(),
            ):
                client = TestClient(app)
                response = client.post("/api/servers/remove", data={"path": "/victim"})
            assert response.status_code == status.HTTP_403_FORBIDDEN
        finally:
            _clear()

    def test_delete_allowed_when_grant_matches_server_name(self) -> None:
        """A delete_service grant for the stored server_name authorizes deletion."""
        _override_auth(self._non_admin_with_delete(["victim-server"]))
        service = self._mock_service()
        try:
            with (
                patch("registry.api.server_routes.server_service", service),
                patch(
                    "registry.core.nginx_service.nginx_reload_scheduler.flush_now",
                    AsyncMock(),
                ),
                patch(
                    "registry.health.service.health_service.broadcast_health_update",
                    AsyncMock(),
                ),
                patch(
                    "registry.services.scope_service.remove_server_scopes",
                    AsyncMock(),
                ),
            ):
                client = TestClient(app)
                response = client.post("/api/servers/remove", data={"path": "/victim"})
            assert response.status_code != status.HTTP_403_FORBIDDEN
            service.remove_server.assert_awaited_once_with("/victim")
        finally:
            _clear()

    def test_delete_forbidden_without_any_grant(self) -> None:
        """A non-admin with no delete_service grant is denied (fail closed)."""
        _override_auth(self._non_admin_with_delete([]))
        try:
            with patch(
                "registry.api.server_routes.server_service",
                self._mock_service(),
            ):
                client = TestClient(app)
                response = client.post("/api/servers/remove", data={"path": "/victim"})
            assert response.status_code == status.HTTP_403_FORBIDDEN
        finally:
            _clear()

    def test_delete_federated_server_blocked_before_permission_check(self) -> None:
        """A federated (read-only) server cannot be deleted even with a valid grant.

        The federated-server guard runs before the delete_service check and
        applies to every caller, so it must reject the deletion regardless of a
        matching server_name grant.
        """
        _override_auth(self._non_admin_with_delete(["victim-server"]))
        service = AsyncMock()
        service.get_server_info = AsyncMock(
            return_value={
                "path": "/victim",
                "server_name": "victim-server",
                "sync_metadata": {"is_federated": True, "source_peer_id": "peer-a"},
            }
        )
        service.remove_server = AsyncMock(return_value=True)
        try:
            with patch("registry.api.server_routes.server_service", service):
                client = TestClient(app)
                response = client.post("/api/servers/remove", data={"path": "/victim"})
            assert response.status_code == status.HTTP_403_FORBIDDEN
            service.remove_server.assert_not_awaited()
        finally:
            _clear()

    def test_delete_allowed_for_admin(self) -> None:
        """An administrator may delete regardless of delete_service grants."""
        _override_auth(_admin)
        service = self._mock_service()
        try:
            with (
                patch("registry.api.server_routes.server_service", service),
                patch(
                    "registry.core.nginx_service.nginx_reload_scheduler.flush_now",
                    AsyncMock(),
                ),
                patch(
                    "registry.health.service.health_service.broadcast_health_update",
                    AsyncMock(),
                ),
                patch(
                    "registry.services.scope_service.remove_server_scopes",
                    AsyncMock(),
                ),
            ):
                client = TestClient(app)
                response = client.post("/api/servers/remove", data={"path": "/victim"})
            assert response.status_code != status.HTTP_403_FORBIDDEN
            service.remove_server.assert_awaited_once_with("/victim")
        finally:
            _clear()
