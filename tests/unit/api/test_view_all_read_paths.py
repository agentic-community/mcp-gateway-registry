"""Read paths honor ``view_all`` (issue #2099); writes, binding and admin routes do not."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from registry.api.agent_routes import (
    _filter_agents_by_access,
    _filter_agents_for_read,
)
from registry.api.custom_entity_visibility import (
    _build_visibility_filter,
    _user_can_read_record,
    _user_can_view,
)
from registry.api.skill_routes import (
    _user_can_access_skill,
    _user_can_read_skill,
)
from registry.api.virtual_server_routes import _require_virtual_server_access
from registry.auth.resource_binding import (
    ResourceType,
    validate_user_can_bind_resource,
)
from registry.schemas.custom_entity_models import CustomEntityRecord
from registry.schemas.skill_models import VisibilityEnum

OTHER_OWNER: str = "someone-else"
OTHER_GROUPS: list[str] = ["other-group"]
VIRTUAL_PATH: str = "/virtual/hidden"


def _context(
    view_all: bool,
) -> dict[str, Any]:
    """A non-admin context with no grants except, optionally, view_all."""
    ui_permissions = {"view_all": ["all"]} if view_all else {}
    return {
        "username": "ro-admin",
        "is_admin": False,
        "groups": ["registry-read-only-admin"],
        "scopes": ["registry-read-only-admin"],
        "accessible_servers": [],
        "accessible_services": [],
        "accessible_agents": [],
        "accessible_skills": [],
        "accessible_tools": {},
        "ui_permissions": ui_permissions,
        "auth_method": "session",
    }


def _agent(
    visibility: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        path="/hidden-agent",
        visibility=visibility,
        registered_by=OTHER_OWNER,
        allowed_groups=OTHER_GROUPS,
    )


def _skill(
    visibility: VisibilityEnum,
) -> SimpleNamespace:
    return SimpleNamespace(
        name="hidden-skill",
        visibility=visibility,
        owner=OTHER_OWNER,
        allowed_groups=OTHER_GROUPS,
    )


def _record(
    visibility: str,
) -> CustomEntityRecord:
    record = CustomEntityRecord(
        entity_type="dataset",
        name="r",
        owner=OTHER_OWNER,
        visibility=visibility,
        allowed_groups=OTHER_GROUPS,
    )
    record.path = "/dataset/abc"
    return record


@pytest.mark.unit
class TestAgentReadAndWrite:
    @pytest.mark.parametrize("visibility", ["private", "group-restricted"])
    def test_read_filter_keeps_hidden_agent_for_holder(self, visibility):
        agents = [_agent(visibility)]
        assert _filter_agents_for_read(agents, _context(True)) == agents

    @pytest.mark.parametrize("visibility", ["private", "group-restricted"])
    def test_write_filter_still_drops_it(self, visibility):
        assert _filter_agents_by_access([_agent(visibility)], _context(True)) == []

    @pytest.mark.parametrize("visibility", ["private", "group-restricted", "public"])
    def test_non_holder_read_equals_write(self, visibility):
        agents = [_agent(visibility)]
        context = _context(False)
        context["accessible_agents"] = ["all"]
        assert _filter_agents_for_read(agents, context) == _filter_agents_by_access(agents, context)


@pytest.mark.unit
class TestSkillReadAndWrite:
    @pytest.mark.parametrize("visibility", [VisibilityEnum.PRIVATE, VisibilityEnum.GROUP])
    def test_holder_reads_hidden_skill(self, visibility):
        assert _user_can_read_skill(_skill(visibility), _context(True)) is True

    @pytest.mark.parametrize("visibility", [VisibilityEnum.PRIVATE, VisibilityEnum.GROUP])
    def test_write_check_still_refuses(self, visibility):
        assert _user_can_access_skill(_skill(visibility), _context(True)) is False


@pytest.mark.unit
class TestCustomEntityReadAndWrite:
    def test_list_filter_is_open_for_holder(self):
        assert _build_visibility_filter(_context(True)) is None

    def test_list_filter_still_restricts_others(self):
        assert _build_visibility_filter(_context(False)) is not None

    @pytest.mark.parametrize("visibility", ["private", "group-restricted"])
    def test_holder_reads_hidden_record(self, visibility):
        assert _user_can_read_record(_record(visibility), _context(True)) is True

    @pytest.mark.parametrize("visibility", ["private", "group-restricted"])
    def test_write_gate_still_refuses(self, visibility):
        assert _user_can_view(_record(visibility), _context(True)) is False


@pytest.mark.unit
class TestVirtualServerReadAndWrite:
    def test_read_gate_passes_holder(self):
        _require_virtual_server_access(_context(True), VIRTUAL_PATH, honor_view_all=True)

    def test_default_gate_refuses_holder(self):
        with pytest.raises(HTTPException) as excinfo:
            _require_virtual_server_access(_context(True), VIRTUAL_PATH)
        assert excinfo.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
class TestTokenBindingUnchanged:
    async def test_holder_cannot_bind_token_to_agent(self):
        allowed = await validate_user_can_bind_resource(
            ResourceType.AGENT, "hidden-agent", _context(True)
        )
        assert allowed is False

    async def test_holder_cannot_bind_token_to_virtual_server(self):
        allowed = await validate_user_can_bind_resource(
            ResourceType.VIRTUAL_SERVER, VIRTUAL_PATH, _context(True)
        )
        assert allowed is False


@pytest.fixture
def mock_server_service():
    """Server service that knows one server the holder has no grant for."""
    service = MagicMock()
    service.get_server_info = AsyncMock(
        return_value={
            "server_name": "hidden",
            "path": "/hidden",
            "proxy_pass_url": "http://internal-backend:8080",
            "tool_list": [],
        }
    )
    service.user_can_access_server_path = AsyncMock(return_value=False)
    return service


@pytest.fixture
def holder_client(mock_settings, mock_server_service):
    """A TestClient authenticated as a view_all holder."""
    from registry.auth.csrf import (
        verify_csrf_token_flexible,
        verify_csrf_token_header_only,
    )
    from registry.auth.dependencies import enhanced_auth, nginx_proxied_auth
    from registry.main import app

    context = _context(True)
    app.dependency_overrides[nginx_proxied_auth] = lambda: context
    app.dependency_overrides[enhanced_auth] = lambda: context
    # Skip CSRF so write routes are refused by authorization, the thing under test.
    app.dependency_overrides[verify_csrf_token_flexible] = lambda: None
    app.dependency_overrides[verify_csrf_token_header_only] = lambda: None
    with (
        patch("registry.api.server_routes.server_service", mock_server_service),
        patch("registry.api.server_routes._apply_tool_visibility", AsyncMock()),
        patch("registry.health.service.health_service", MagicMock()),
        patch("registry.core.nginx_service.nginx_service", MagicMock()),
        patch("registry.api.server_routes.security_scanner_service", MagicMock()),
    ):
        yield TestClient(app, cookies={"mcp_gateway_session": "test-session"})
    app.dependency_overrides.clear()


@pytest.mark.unit
class TestHolderOverHttp:
    def test_auth_me_reports_view_all_without_admin(self, holder_client):
        data = holder_client.get("/api/auth/me").json()
        assert data["view_all"] is True
        assert data["is_admin"] is False

    def test_server_detail_readable_and_redacted(self, holder_client):
        from registry.core.config import DeploymentMode

        with patch(
            "registry.services.visibility.settings.deployment_mode",
            DeploymentMode.WITH_GATEWAY,
        ):
            response = holder_client.get("/api/servers/hidden")
        assert response.status_code == 200
        assert "proxy_pass_url" not in response.json()

    def test_server_rate_still_refused(self, holder_client):
        response = holder_client.post("/api/servers/hidden/rate", json={"rating": 5})
        assert response.status_code == 403
        assert "csrf" not in response.text.lower()

    @pytest.mark.parametrize(
        "method,url",
        [
            ("get", "/api/config/full"),
            ("get", "/api/audit/statistics"),
            ("post", "/api/servers/groups/import"),
            ("get", "/api/export/scopes"),
        ],
    )
    def test_admin_routes_refused(self, holder_client, method, url):
        response = getattr(holder_client, method)(url)
        assert response.status_code == 403, f"{url} returned {response.status_code}"
        assert "admin" in response.text.lower(), response.text


@pytest.fixture
def scan_result() -> dict[str, Any]:
    """A stored server scan result, with the fields only admins may see."""
    return {
        "server_path": "/hidden",
        "server_url": "http://internal-backend:8080/mcp",
        "output_file": "/app/security_scans/hidden.json",
        "raw_output": {"analysis_results": {"yara": {"findings": []}}},
        "is_safe": True,
    }


@pytest.mark.unit
class TestHolderSensitiveRoutes:
    """Routes that touch credentials, backend URLs or stored state (review findings 1-4)."""

    def test_connect_config_refused(self, holder_client):
        """Decrypted custom headers never reach a view_all holder."""
        response = holder_client.get("/api/servers/hidden/connect-config")
        assert response.status_code == 403
        assert "csrf" not in response.text.lower()

    def test_security_scan_redacts_backend_url(self, holder_client, scan_result):
        scanner = MagicMock()
        scanner.get_scan_result = AsyncMock(return_value=scan_result)
        with patch("registry.api.server_routes.security_scanner_service", scanner):
            response = holder_client.get("/api/servers/hidden/security-scan")
        assert response.status_code == 200
        body = response.json()
        assert "server_url" not in body
        assert "output_file" not in body
        assert body["raw_output"] == scan_result["raw_output"]

    def test_tool_list_served_from_storage_without_live_fetch(
        self, holder_client, mock_server_service
    ):
        mock_server_service.get_server_info.return_value = {
            "server_name": "hidden",
            "path": "/hidden",
            "proxy_pass_url": "http://internal-backend:8080",
            "tool_list": [{"name": "read_note"}],
        }
        mcp = MagicMock()
        mcp.get_tools_from_server_with_server_info = AsyncMock()
        with (
            patch("registry.core.mcp_client.mcp_client_service", mcp),
            patch(
                "registry.api.server_routes.annotate_blocked_tools",
                AsyncMock(side_effect=lambda _path, tools: tools),
            ),
        ):
            response = holder_client.get("/api/servers/tools/hidden")
        assert response.status_code == 200
        assert response.json()["cached"] is True
        assert [tool["name"] for tool in response.json()["tools"]] == ["read_note"]
        mcp.get_tools_from_server_with_server_info.assert_not_called()
        mock_server_service.update_server.assert_not_called()

    def test_skill_content_refused(self, holder_client):
        """Fetching content uses the owner's credential and writes health state."""
        service = MagicMock()
        service.get_skill = AsyncMock(return_value=_skill(VisibilityEnum.PRIVATE))
        with patch("registry.api.skill_routes.get_skill_service", return_value=service):
            response = holder_client.get("/api/skills/hidden-skill/content")
        assert response.status_code == 403


@pytest.mark.unit
class TestHolderNotesAndAggregates:
    """Clear refusal messages for holders, and the "all" aggregates (PR review items 2-3)."""

    def test_connect_config_refusal_names_read_only_admin(self, holder_client):
        from registry.auth.asset_permissions import VIEW_ALL_DENIED_PREFIX

        response = holder_client.get("/api/servers/hidden/connect-config")
        assert response.status_code == 403
        assert response.json()["detail"].startswith(VIEW_ALL_DENIED_PREFIX)

    def test_skill_content_refusal_names_read_only_admin(self, holder_client):
        from registry.auth.asset_permissions import VIEW_ALL_DENIED_PREFIX

        service = MagicMock()
        service.get_skill = AsyncMock(return_value=_skill(VisibilityEnum.PRIVATE))
        with patch("registry.api.skill_routes.get_skill_service", return_value=service):
            response = holder_client.get("/api/skills/hidden-skill/content")
        assert response.status_code == 403
        assert response.json()["detail"].startswith(VIEW_ALL_DENIED_PREFIX)

    def test_all_server_details_include_every_server_redacted(
        self, holder_client, mock_server_service
    ):
        from registry.core.config import DeploymentMode

        mock_server_service.get_all_servers = AsyncMock(
            return_value={
                "/hidden": {
                    "server_name": "hidden",
                    "path": "/hidden",
                    "proxy_pass_url": "http://internal-backend:8080",
                }
            }
        )
        mock_server_service.get_all_servers_with_permissions = AsyncMock(return_value={})
        with patch(
            "registry.services.visibility.settings.deployment_mode",
            DeploymentMode.WITH_GATEWAY,
        ):
            response = holder_client.get("/api/server_details/all")
        assert response.status_code == 200
        body = response.json()
        assert "/hidden" in body
        assert "proxy_pass_url" not in body["/hidden"]
        mock_server_service.get_all_servers_with_permissions.assert_not_called()

    def test_all_tools_uses_stored_lists_for_every_server(self, holder_client, mock_server_service):
        mock_server_service.get_all_servers = AsyncMock(
            return_value={
                "/hidden": {
                    "server_name": "hidden",
                    "path": "/hidden",
                    "tool_list": [{"name": "read_note"}],
                }
            }
        )
        mock_server_service.get_all_servers_with_permissions = AsyncMock(return_value={})
        mock_server_service.is_service_enabled = AsyncMock(return_value=True)
        mcp = MagicMock()
        mcp.get_tools_from_server_with_server_info = AsyncMock()
        with (
            patch("registry.core.mcp_client.mcp_client_service", mcp),
            patch(
                "registry.api.server_routes.annotate_blocked_tools",
                AsyncMock(side_effect=lambda _path, tools: tools),
            ),
        ):
            response = holder_client.get("/api/servers/tools/all")
        assert response.status_code == 200
        assert [tool["name"] for tool in response.json()["tools"]] == ["read_note"]
        mcp.get_tools_from_server_with_server_info.assert_not_called()
