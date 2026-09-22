"""Per-tool visibility on the single-server read endpoints.

CR2-21 scoped the tool CATALOG to the caller's server/tool access. The same
disclosure exists on the single-server detail reads: ``GET /api/servers/{path}``
(``get_server``) and ``GET /api/server_details/{path}`` (``get_server_details``)
returned the server's full ``tool_list`` to any caller with server access, even
when that caller's tool allowlist was a strict subset. These tests assert that
both endpoints now prune ``tool_list`` through the canonical
``filter_tools_for_user`` helper (via ``_apply_tool_visibility``), matching the
server-listing and tool-catalog paths, and that ``num_tools`` tracks the pruned
list.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from registry.api import server_routes


def _server_doc() -> dict[str, Any]:
    """A remote server document with three tools."""
    return {
        "server_name": "test-server",
        "path": "/test-server",
        "deployment": "remote",
        "num_tools": 3,
        "tool_list": [
            {"name": "read_item"},
            {"name": "list_items"},
            {"name": "delete_item"},
        ],
        "visibility": "public",
    }


# --- Unit tests of the shared helper --------------------------------------


class TestApplyToolVisibility:
    """Direct coverage of the in-place pruning helper."""

    async def test_restricted_user_sees_only_allowed_tools(self):
        server_info = _server_doc()
        user_context = {
            "username": "u",
            "is_admin": False,
            "accessible_servers": ["test-server"],
            "accessible_tools": {"/test-server": {"read_item"}},
        }
        await server_routes._apply_tool_visibility(
            server_info, "/test-server", user_context, endpoint="server_detail"
        )
        names = [t["name"] for t in server_info["tool_list"]]
        assert names == ["read_item"]
        assert server_info["num_tools"] == 1

    async def test_admin_sees_all_tools(self):
        server_info = _server_doc()
        await server_routes._apply_tool_visibility(
            server_info, "/test-server", {"is_admin": True}, endpoint="server_detail"
        )
        assert len(server_info["tool_list"]) == 3
        assert server_info["num_tools"] == 3

    async def test_missing_allowlist_fails_closed(self):
        server_info = _server_doc()
        user_context = {
            "username": "u",
            "is_admin": False,
            "accessible_servers": ["test-server"],
            "accessible_tools": {},  # no entry for this server
        }
        await server_routes._apply_tool_visibility(
            server_info, "/test-server", user_context, endpoint="server_detail"
        )
        assert server_info["tool_list"] == []
        assert server_info["num_tools"] == 0

    async def test_non_list_tool_list_left_untouched(self):
        server_info = {"server_name": "s", "tool_list": None}
        await server_routes._apply_tool_visibility(
            server_info, "/s", {"is_admin": False}, endpoint="server_detail"
        )
        assert server_info["tool_list"] is None


# --- Endpoint tests --------------------------------------------------------


@pytest.fixture
def restricted_user_context() -> dict[str, Any]:
    """Non-admin with server access but only one allowed tool."""
    return {
        "username": "testuser",
        "is_admin": False,
        "groups": ["test-group"],
        "scopes": ["test-server/read"],
        "accessible_servers": ["test-server"],
        "accessible_services": ["test-server"],
        "accessible_tools": {"/test-server": {"read_item"}},
        "ui_permissions": {"list_service": ["test-server"]},
        "auth_method": "session",
    }


@pytest.fixture
def admin_user_context() -> dict[str, Any]:
    return {
        "username": "admin",
        "is_admin": True,
        "groups": ["mcp-registry-admin"],
        "accessible_servers": ["all"],
        "accessible_tools": {"*": {"*"}},
        "auth_method": "session",
    }


@pytest.fixture
def mock_server_service():
    mock_service = MagicMock()
    mock_service.get_server_info = AsyncMock(return_value=_server_doc())
    mock_service.user_can_access_server_path = AsyncMock(return_value=True)
    return mock_service


def _make_client(user_context: dict[str, Any], mock_server_service: MagicMock):
    from registry.auth.dependencies import enhanced_auth, nginx_proxied_auth
    from registry.main import app

    app.dependency_overrides[nginx_proxied_auth] = lambda: user_context
    app.dependency_overrides[enhanced_auth] = lambda: user_context

    # Registry-only mode keeps backend URLs (irrelevant here) and avoids the
    # redaction branch; the tool-visibility prune runs regardless of mode.
    with (
        patch("registry.api.server_routes.server_service", mock_server_service),
        patch("registry.health.service.health_service", MagicMock()),
    ):
        client = TestClient(app, cookies={"mcp_gateway_session": "test-session"})
        yield client

    app.dependency_overrides.clear()


@pytest.fixture
def client_restricted(restricted_user_context, mock_server_service):
    yield from _make_client(restricted_user_context, mock_server_service)


@pytest.fixture
def client_admin(admin_user_context, mock_server_service):
    yield from _make_client(admin_user_context, mock_server_service)


class TestGetServerToolVisibility:
    """GET /api/servers/{path} must prune tool_list to the caller's allowlist."""

    def test_restricted_user_sees_only_allowed_tools(self, client_restricted):
        resp = client_restricted.get("/api/servers/test-server")
        assert resp.status_code == 200
        body = resp.json()
        names = [t["name"] for t in body["tool_list"]]
        assert names == ["read_item"]
        assert "delete_item" not in names
        assert body["num_tools"] == 1

    def test_admin_sees_all_tools(self, client_admin):
        resp = client_admin.get("/api/servers/test-server")
        assert resp.status_code == 200
        assert len(resp.json()["tool_list"]) == 3


class TestGetServerDetailsToolVisibility:
    """GET /api/server_details/{path} must prune tool_list to the caller's allowlist."""

    def test_restricted_user_sees_only_allowed_tools(self, client_restricted):
        resp = client_restricted.get("/api/server_details/test-server")
        assert resp.status_code == 200
        body = resp.json()
        names = [t["name"] for t in body["tool_list"]]
        assert names == ["read_item"]
        assert "delete_item" not in names
        assert body["num_tools"] == 1

    def test_admin_sees_all_tools(self, client_admin):
        resp = client_admin.get("/api/server_details/test-server")
        assert resp.status_code == 200
        assert len(resp.json()["tool_list"]) == 3
