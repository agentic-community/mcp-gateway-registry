"""Guard and audit tests for the IdP M2M and user-group demotion routes.

Each route can strip a principal's admin-conferring groups, so each must
refuse with 409 when the target is the only administrator, and still allow
the change when another administrator remains. The Okta and Auth0 group
routes must also record an audit action naming the target.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from registry.auth.dependencies import nginx_proxied_auth
from registry.main import app
from registry.schemas.idp_user_group import IdPUserGroup

SOLE_ADMIN: list[frozenset[str]] = [frozenset({"target"})]
TWO_ADMINS: list[frozenset[str]] = [frozenset({"target"}), frozenset({"other-admin"})]


def _admin() -> dict:
    """A genuine administrator (takes no args: FastAPI introspects overrides)."""
    return {
        "username": "admin",
        "groups": ["mcp-registry-admin"],
        "scopes": [],
        "is_admin": True,
        "ui_permissions": {"register_service": ["all"]},
        "accessible_servers": ["all"],
        "accessible_agents": ["all"],
        "accessible_services": ["all"],
    }


@pytest.fixture
def admin_population():
    """Mock the admin population; tests set ``.return_value``."""
    identities = AsyncMock(return_value=TWO_ADMINS)
    with (
        patch(
            "registry.services.admin_safety.resolve_admin_group_names",
            new=AsyncMock(return_value={"mcp-registry-admin"}),
        ),
        patch("registry.services.admin_safety.list_admin_identities", new=identities),
    ):
        yield identities


@pytest.fixture
def client(mock_settings, admin_population):
    app.dependency_overrides[nginx_proxied_auth] = _admin
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def idp_sync():
    """Mock Okta and Auth0 sync services whose group update always succeeds."""
    sync = MagicMock()
    sync.update_client_groups = AsyncMock(return_value=True)
    with (
        patch("registry.api.okta_m2m_routes.get_documentdb_client", new=AsyncMock()),
        patch("registry.api.okta_m2m_routes.get_okta_m2m_sync", return_value=sync),
        patch("registry.api.auth0_m2m_routes.get_documentdb_client", new=AsyncMock()),
        patch("registry.api.auth0_m2m_routes.get_auth0_m2m_sync", return_value=sync),
    ):
        yield sync


@pytest.fixture
def user_group_service():
    """Mock the user-group service with the fallback store active."""
    service = MagicMock()
    service.update_user_group = AsyncMock(
        return_value=IdPUserGroup(username="target", groups=[], provider="pingfederate")
    )
    service.delete_user_group = AsyncMock(return_value=None)
    with (
        patch(
            "registry.api.iam_user_groups_routes._get_service",
            new=AsyncMock(return_value=service),
        ),
        patch(
            "registry.services.admin_safety.user_group_fallback_active",
            return_value=True,
        ),
    ):
        yield service


IDP_GROUP_ROUTES = [
    "/api/iam/okta/m2m/clients/target/groups",
    "/api/iam/auth0/m2m/clients/target/groups",
]


@pytest.mark.unit
@pytest.mark.api
class TestIdPM2MGroupRoutes:
    """PATCH /api/iam/{okta,auth0}/m2m/clients/{client_id}/groups."""

    @pytest.mark.parametrize("path", IDP_GROUP_ROUTES)
    def test_demoting_last_admin_returns_409(self, client, admin_population, idp_sync, path):
        admin_population.return_value = SOLE_ADMIN

        response = client.patch(path, json={"groups": ["readers"]})

        assert response.status_code == 409
        idp_sync.update_client_groups.assert_not_called()

    @pytest.mark.parametrize("path", IDP_GROUP_ROUTES)
    def test_demoting_non_last_admin_is_allowed(self, client, idp_sync, path):
        response = client.patch(path, json={"groups": ["readers"]})

        assert response.status_code == 200
        idp_sync.update_client_groups.assert_awaited_once_with("target", ["readers"])

    @pytest.mark.parametrize("idp", ["okta", "auth0"])
    def test_group_change_records_audit_action(self, client, idp_sync, idp):
        # The audit record names the target client and the groups granted.
        with patch(f"registry.api.{idp}_m2m_routes.set_audit_action") as audit:
            response = client.patch(
                f"/api/iam/{idp}/m2m/clients/target/groups", json={"groups": ["readers"]}
            )

        assert response.status_code == 200
        args, kwargs = audit.call_args
        assert args[1:3] == ("update", "m2m_client")
        assert kwargs["resource_id"] == "target"
        assert kwargs["metadata"] == {"groups": ["readers"]}

    @pytest.mark.parametrize("path", IDP_GROUP_ROUTES)
    def test_fails_closed_when_population_unknown(self, client, admin_population, idp_sync, path):
        from registry.services.admin_safety import AdminSafetyError

        admin_population.side_effect = AdminSafetyError(503, "unknown")

        response = client.patch(path, json={"groups": ["readers"]})

        assert response.status_code == 503
        idp_sync.update_client_groups.assert_not_called()


@pytest.mark.unit
@pytest.mark.api
class TestUserGroupRoutes:
    """PATCH and DELETE /api/iam/user-groups/{username}."""

    def test_deleting_last_admin_returns_409(self, client, admin_population, user_group_service):
        admin_population.return_value = SOLE_ADMIN

        response = client.delete("/api/iam/user-groups/target")

        assert response.status_code == 409
        user_group_service.delete_user_group.assert_not_called()

    def test_deleting_non_last_admin_is_allowed(self, client, user_group_service):
        response = client.delete("/api/iam/user-groups/target")

        assert response.status_code == 204

    @pytest.mark.parametrize("body", [{"groups": []}, {"enabled": False}])
    def test_demoting_last_admin_returns_409(
        self, client, admin_population, user_group_service, body
    ):
        admin_population.return_value = SOLE_ADMIN

        response = client.patch("/api/iam/user-groups/target", json=body)

        assert response.status_code == 409
        user_group_service.update_user_group.assert_not_called()

    def test_demoting_non_last_admin_is_allowed(self, client, user_group_service):
        response = client.patch("/api/iam/user-groups/target", json={"groups": []})

        assert response.status_code == 200

    def test_guard_skipped_when_fallback_store_unused(self, client, admin_population):
        # When the auth server never reads idp_user_groups, the record confers
        # nothing, so removing it cannot demote anyone.
        admin_population.return_value = SOLE_ADMIN
        service = MagicMock()
        service.delete_user_group = AsyncMock(return_value=None)
        with (
            patch(
                "registry.api.iam_user_groups_routes._get_service",
                new=AsyncMock(return_value=service),
            ),
            patch(
                "registry.services.admin_safety.user_group_fallback_active",
                return_value=False,
            ),
        ):
            response = client.delete("/api/iam/user-groups/target")

        assert response.status_code == 204
        admin_population.assert_not_called()


_ADMIN_GRANT_AUDIT = "registry.api._admin_grant_audit.set_audit_action"


def _admin_grant_recorded(audit: MagicMock) -> bool:
    """Return True if any set_audit_action call marked an admin-tier grant."""
    return any(
        call.kwargs.get("metadata", {}).get("admin_grant") is True for call in audit.call_args_list
    )


@pytest.fixture
def m2m_service():
    """Mock the generic M2M management service."""
    from registry.schemas.idp_m2m_client import IdPM2MClient

    record = IdPM2MClient(
        client_id="target", name="target", groups=["mcp-registry-admin"], provider="manual"
    )
    service = MagicMock()
    service.create = AsyncMock(return_value=record)
    service.patch = AsyncMock(return_value=record)
    with patch(
        "registry.api.m2m_management_routes._get_service",
        new=AsyncMock(return_value=service),
    ):
        yield service


@pytest.mark.unit
@pytest.mark.api
class TestAdminGrantAudit:
    """Every route that sets a principal's groups records an admin-tier grant."""

    @pytest.mark.parametrize("idp", ["okta", "auth0"])
    @pytest.mark.parametrize(
        ("groups", "granted"), [(["mcp-registry-admin"], True), (["x"], False)]
    )
    def test_idp_m2m_group_update(self, client, idp_sync, idp, groups, granted):
        with patch(_ADMIN_GRANT_AUDIT) as audit:
            response = client.patch(
                f"/api/iam/{idp}/m2m/clients/target/groups", json={"groups": groups}
            )

        assert response.status_code == 200
        assert _admin_grant_recorded(audit) is granted

    def test_generic_m2m_create(self, client, m2m_service):
        with patch(_ADMIN_GRANT_AUDIT) as audit:
            response = client.post(
                "/api/iam/m2m-clients",
                json={
                    "client_id": "target",
                    "client_name": "target",
                    "groups": ["mcp-registry-admin"],
                },
            )

        assert response.status_code == 201
        assert _admin_grant_recorded(audit) is True

    @pytest.mark.parametrize(
        ("body", "granted"),
        [({"groups": ["mcp-registry-admin"]}, True), ({"description": "d"}, False)],
    )
    def test_generic_m2m_patch(self, client, m2m_service, body, granted):
        # A description-only edit grants nothing, even on an admin record.
        with patch(_ADMIN_GRANT_AUDIT) as audit:
            response = client.patch("/api/iam/m2m-clients/target", json=body)

        assert response.status_code == 200
        assert _admin_grant_recorded(audit) is granted

    @pytest.mark.parametrize(("fallback_active", "granted"), [(True, True), (False, False)])
    def test_user_group_create(self, client, fallback_active, granted):
        # The record confers groups only when the auth server reads the store.
        service = MagicMock()
        service.register_user_group = AsyncMock(
            return_value=IdPUserGroup(
                username="target", groups=["mcp-registry-admin"], provider="pingfederate"
            )
        )
        with (
            patch(
                "registry.api.iam_user_groups_routes._get_service",
                new=AsyncMock(return_value=service),
            ),
            patch(
                "registry.services.admin_safety.user_group_fallback_active",
                return_value=fallback_active,
            ),
            patch(_ADMIN_GRANT_AUDIT) as audit,
        ):
            response = client.post(
                "/api/iam/user-groups",
                json={"username": "target", "groups": ["mcp-registry-admin"]},
            )

        assert response.status_code == 201
        assert _admin_grant_recorded(audit) is granted

    def test_user_group_patch(self, client, user_group_service):
        user_group_service.update_user_group.return_value = IdPUserGroup(
            username="target", groups=["mcp-registry-admin"], provider="pingfederate"
        )
        with patch(_ADMIN_GRANT_AUDIT) as audit:
            response = client.patch(
                "/api/iam/user-groups/target", json={"groups": ["mcp-registry-admin"]}
            )

        assert response.status_code == 200
        assert _admin_grant_recorded(audit) is True
